import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

PRICING_PER_1M = {"input": 0.15, "output": 0.60}

FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|ATTACH|DETACH|PRAGMA|CREATE|REPLACE)\b|;",
    re.IGNORECASE,
)


@dataclass
class SpanRecord:
    invocation_id: str
    span_type: str
    agent_name: str
    target_name: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_sec: float = 0.0
    cost_usd: float = 0.0
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)


class TelemetryAndGuardrailPlugin(BasePlugin):
    """Global ADK plugin for LLM-native observability, OTLP export, and tool guardrails."""

    def __init__(self) -> None:
        super().__init__(name="aegis_telemetry_guardrail")
        self._timers: dict[str, float] = {}
        self.traces: dict[str, list[dict[str, Any]]] = {}

    def _record(self, session_id: str, span: SpanRecord) -> None:
        self.traces.setdefault(session_id, []).append(asdict(span))

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        key = f"model:{callback_context.invocation_id}:{callback_context.agent_name}"
        self._timers[key] = time.perf_counter()
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        key = f"model:{callback_context.invocation_id}:{callback_context.agent_name}"
        start = self._timers.pop(key, time.perf_counter())
        elapsed_s = max(time.perf_counter() - start, 1e-4)

        usage = getattr(llm_response, "usage_metadata", None)
        in_tok = (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        out_tok = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0

        tps = round(out_tok / elapsed_s, 2) if out_tok else 0.0
        cost = round(
            (in_tok * PRICING_PER_1M["input"] + out_tok * PRICING_PER_1M["output"]) / 1_000_000,
            6,
        )

        session_id = callback_context._invocation_context.session.id
        self._record(
            session_id,
            SpanRecord(
                invocation_id=callback_context.invocation_id,
                span_type="llm_call",
                agent_name=callback_context.agent_name,
                target_name="gemini-model",
                latency_ms=round(elapsed_s * 1000, 2),
                input_tokens=in_tok,
                output_tokens=out_tok,
                tokens_per_sec=tps,
                cost_usd=cost,
            ),
        )
        return None

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        key = f"tool:{tool_context.invocation_id}:{tool.name}"
        self._timers[key] = time.perf_counter()

        if tool.name == "query_telemetry_sql":
            sql = (tool_args.get("sql_query") or "").strip()
            if not sql.upper().startswith("SELECT") or FORBIDDEN_SQL_PATTERN.search(sql):
                session_id = tool_context._invocation_context.session.id
                self._record(
                    session_id,
                    SpanRecord(
                        invocation_id=tool_context.invocation_id,
                        span_type="guardrail_block",
                        agent_name=tool_context.agent_name,
                        target_name=tool.name,
                        latency_ms=0.0,
                        status="blocked",
                        metadata={"blocked_sql": sql},
                    ),
                )
                return {
                    "status": "blocked",
                    "error": "SECURITY_POLICY_VIOLATION: Only single read-only SELECT queries are permitted.",
                }
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        key = f"tool:{tool_context.invocation_id}:{tool.name}"
        start = self._timers.pop(key, time.perf_counter())
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        session_id = tool_context._invocation_context.session.id
        self._record(
            session_id,
            SpanRecord(
                invocation_id=tool_context.invocation_id,
                span_type="tool_call",
                agent_name=tool_context.agent_name,
                target_name=tool.name,
                latency_ms=elapsed_ms,
                status=str(result.get("status", "ok")) if isinstance(result, dict) else "ok",
                metadata={"args": tool_args},
            ),
        )
        return None

    def summarize_session(self, session_id: str) -> dict[str, Any]:
        spans = self.traces.get(session_id, [])
        llm_spans = [s for s in spans if s["span_type"] == "llm_call"]
        total_cost = round(sum(s["cost_usd"] for s in llm_spans), 6)
        total_latency_ms = round(sum(s["latency_ms"] for s in spans), 2)
        avg_tps = (
            round(sum(s["tokens_per_sec"] for s in llm_spans) / len(llm_spans), 2)
            if llm_spans
            else 0.0
        )
        return {
            "session_id": session_id,
            "total_spans": len(spans),
            "total_input_tokens": sum(s["input_tokens"] for s in llm_spans),
            "total_output_tokens": sum(s["output_tokens"] for s in llm_spans),
            "avg_tokens_per_sec": avg_tps,
            "total_cost_usd": total_cost,
            "total_latency_ms": total_latency_ms,
            "guardrail_blocks": sum(1 for s in spans if s["span_type"] == "guardrail_block"),
            "spans": spans,
        }


def build_otlp_payload(trace_summary: dict[str, Any]) -> dict[str, Any]:
    """Formats session trace spans into OpenTelemetry (OTLP) JSON for Google Cloud Trace."""
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "aegis-ops-adk"}},
                        {"key": "session.id", "value": {"stringValue": trace_summary["session_id"]}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "google.adk.telemetry_plugin", "version": "2.0.0"},
                        "spans": [
                            {
                                "traceId": trace_summary["session_id"],
                                "spanId": f"span-{idx+1:04d}",
                                "name": f"{s['agent_name']}.{s['target_name']}",
                                "kind": "SPAN_KIND_INTERNAL",
                                "attributes": [
                                    {"key": "span.type", "value": {"stringValue": s["span_type"]}},
                                    {"key": "latency.ms", "value": {"doubleValue": s["latency_ms"]}},
                                    {"key": "llm.tokens.input", "value": {"intValue": s.get("input_tokens", 0)}},
                                    {"key": "llm.tokens.output", "value": {"intValue": s.get("output_tokens", 0)}},
                                    {"key": "llm.tokens_per_sec", "value": {"doubleValue": s.get("tokens_per_sec", 0.0)}},
                                    {"key": "llm.cost_usd", "value": {"doubleValue": s.get("cost_usd", 0.0)}},
                                    {"key": "status", "value": {"stringValue": s.get("status", "ok")}},
                                ],
                            }
                            for idx, s in enumerate(trace_summary.get("spans", []))
                        ],
                    }
                ],
            }
        ]
    }