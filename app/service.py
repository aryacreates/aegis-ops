import json
import uuid
from typing import Any, AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents import kb, root_orchestrator
from app.telemetry_plugin import TelemetryAndGuardrailPlugin

telemetry_plugin = TelemetryAndGuardrailPlugin()
adk_app = App(
    name="aegis_ops",
    root_agent=root_orchestrator,
    plugins=[telemetry_plugin],
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=600),
)
session_service = InMemorySessionService()
runner = Runner(app=adk_app, session_service=session_service)


async def stream_investigation(
    query: str, user_id: str = "fde-engineer", session_id: str | None = None
) -> AsyncGenerator[dict[str, Any], None]:
    sid = session_id or f"inc-{uuid.uuid4().hex[:8]}"
    existing = await session_service.get_session(
        app_name="aegis_ops", user_id=user_id, session_id=sid
    )
    if existing is None:
        await session_service.create_session(
            app_name="aegis_ops", user_id=user_id, session_id=sid
        )

    message = types.Content(role="user", parts=[types.Part(text=query)])
    final_responses: list[str] = []

    async for event in runner.run_async(
        user_id=user_id, session_id=sid, new_message=message
    ):
        author = getattr(event, "author", "system")
        fn_calls = [
            {"name": fc.name, "args": dict(fc.args or {})}
            for fc in (event.get_function_calls() or [])
        ]
        fn_resps = [
            {"name": fr.name, "response": fr.response}
            for fr in (event.get_function_responses() or [])
        ]
        actions = getattr(event, "actions", None)
        transfer_to = getattr(actions, "transfer_to_agent", None) if actions else None
        escalate = getattr(actions, "escalate", None) if actions else None

        text = ""
        if event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
        if event.is_final_response() and text:
            final_responses.append(text)

        yield {
            "type": "agent_event",
            "author": author,
            "text": text,
            "function_calls": fn_calls,
            "function_responses": fn_resps,
            "transfer_to": transfer_to,
            "escalate": bool(escalate),
            "is_final": event.is_final_response(),
        }

    session = await session_service.get_session(
        app_name="aegis_ops", user_id=user_id, session_id=sid
    )
    state = dict(session.state) if session else {}
    answer = state.get("rca_draft") or state.get("finops_report") or (
        final_responses[-1] if final_responses else "No response emitted."
    )
    metrics = telemetry_plugin.summarize_session(sid)

    change_request = None
    if isinstance(state.get("proposed_action"), dict):
        change_request = kb.create_change_request(sid, state["proposed_action"])

    kb.save_incident(session_id=sid, query=query, answer=answer, telemetry=metrics)

    yield {
        "type": "complete",
        "payload": {
            "session_id": sid,
            "answer": answer,
            "critic_feedback": state.get("critic_feedback", ""),
            "blast_radius": state.get("blast_radius"),
            "change_request": change_request,
            "session_state": state,
            "telemetry": metrics,
        },
    }


async def execute_investigation(
    query: str, user_id: str = "fde-engineer", session_id: str | None = None
) -> dict[str, Any]:
    final_payload: dict[str, Any] = {}
    async for chunk in stream_investigation(query=query, user_id=user_id, session_id=session_id):
        if chunk["type"] == "complete":
            final_payload = chunk["payload"]
    return final_payload


def sse_encode(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"