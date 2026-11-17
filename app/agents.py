from typing import Any

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.tools import exit_loop
from google.adk.tools.tool_context import ToolContext

from app.data_pipeline import HybridKnowledgeBase
from app.topology import compute_blast_radius

MODEL_ID = "gemini-2.5-flash"
kb = HybridKnowledgeBase()


def retrieve_runbooks(query: str, tool_context: ToolContext) -> dict[str, Any]:
    """Search enterprise SRE & FinOps runbooks using hybrid vector + keyword RRF retrieval."""
    results = kb.search_runbooks(query, top_k=2)
    tool_context.state["retrieved_runbook_ids"] = [r["id"] for r in results]
    return {"status": "ok", "runbooks": results}


def query_telemetry_sql(sql_query: str) -> dict[str, Any]:
    """
    Execute a read-only SELECT SQL query against tables:
    - services(service_name, owner_team, sla_tier, p99_latency_ms, error_rate_pct, active_version, status)
    - deployments(deploy_id, service_name, version, deployed_at, change_summary)
    - billing_daily(date, service_name, gcp_product, cost_usd)
    """
    return kb.run_readonly_sql(sql_query)


def analyze_blast_radius(service_name: str, tool_context: ToolContext) -> dict[str, Any]:
    """Compute downstream customer-facing blast radius using BFS over the service dependency DAG."""
    result = compute_blast_radius(service_name)
    tool_context.state["blast_radius"] = result
    return {"status": "ok", **result}


def propose_remediation_action(
    service_name: str,
    action_type: str,
    target_value: str,
    rationale: str,
    runbook_id: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """
    Register a structured remediation change request for Human-in-the-Loop approval.
    action_type must be one of: ROLLBACK_DEPLOYMENT, SCALE_MEMORY, CAP_BIGQUERY_BYTES, ENABLE_SQL_POOLING.
    """
    proposal = {
        "service_name": service_name,
        "action_type": action_type,
        "target_value": target_value,
        "rationale": rationale,
        "runbook_id": runbook_id,
    }
    tool_context.state["proposed_action"] = proposal
    return {"status": "ok", "queued_for_hitl_approval": proposal}


diagnostics_agent = LlmAgent(
    name="diagnostics_agent",
    model=MODEL_ID,
    description="Investigates outages, latency spikes, and errors using SQL, BFS blast radius, and RAG.",
    instruction="""
    You are an SRE Diagnostics Engineer using the ReAct pattern.
    1. Call `query_telemetry_sql` to inspect `services` and `deployments` for the affected service.
    2. Call `analyze_blast_radius` on the affected service to quantify downstream customer impact.
    3. Call `retrieve_runbooks` to fetch official remediation steps.
    4. Call `propose_remediation_action` to stage a structured safe change (e.g., ROLLBACK_DEPLOYMENT to v2.13.9).
    5. Write a clear Root Cause Analysis (RCA) citing SQL numbers, blast radius severity, and runbook IDs.
    If critic feedback exists below, fix every issue raised:
    Previous Critic Feedback: {critic_feedback?}
    """,
    tools=[retrieve_runbooks, query_telemetry_sql, analyze_blast_radius, propose_remediation_action],
    output_key="rca_draft",
)

safety_critic_agent = LlmAgent(
    name="safety_critic_agent",
    model=MODEL_ID,
    description="Audits proposed RCA & remediation plans for citation grounding, blast radius awareness, and safety.",
    instruction="""
    You are a Principal Production Safety Reviewer. Review the proposed RCA draft:
    ---
    {rca_draft?}
    ---
    Verify ALL four criteria:
    1. Groundedness: Cites actual numbers from SQL and an explicit runbook ID (e.g., rb-gke-oom, rb-sql-pool).
    2. Blast Radius: Mentions downstream impacted services or severity from the dependency graph.
    3. Safety: Avoids destructive actions (dropping DBs, restarting primary DBs, deleting PVCs).
    4. Actionability: A structured remediation action was proposed via `propose_remediation_action`.

    If ALL criteria pass, call `exit_loop` immediately and output a concise approval summary.
    Otherwise, do NOT call `exit_loop`; list the exact missing evidence or violations.
    """,
    tools=[exit_loop],
    output_key="critic_feedback",
)

sre_resolution_loop = LoopAgent(
    name="sre_resolution_loop",
    description="Handles production outages, latency spikes, crashes, and OOM incidents with self-reflection.",
    sub_agents=[diagnostics_agent, safety_critic_agent],
    max_iterations=3,
)

finops_specialist = LlmAgent(
    name="finops_specialist",
    model=MODEL_ID,
    description="Handles GCP billing anomalies, cost spikes, and BigQuery/GKE cost optimization.",
    instruction="""
    You are a Google Cloud FinOps Architect.
    1. Query `billing_daily` via `query_telemetry_sql` to quantify the spend anomaly.
    2. Call `retrieve_runbooks` to fetch FinOps remediation guidelines.
    3. Call `propose_remediation_action` with action_type='CAP_BIGQUERY_BYTES' to stage a cost-control policy for HITL approval.
    4. Return a clear cost breakdown, root cause, and remediation summary citing the runbook ID.
    """,
    tools=[retrieve_runbooks, query_telemetry_sql, propose_remediation_action],
    output_key="finops_report",
)

root_orchestrator = LlmAgent(
    name="aegis_orchestrator",
    model=MODEL_ID,
    description="Enterprise triage coordinator routing incidents to SRE or FinOps sub-agents.",
    instruction="""
    You are the AegisOps Incident Commander.
    - For cost spikes, BigQuery spend, or FinOps questions -> transfer to `finops_specialist`.
    - For outages, OOMKilled, latency spikes, 503 errors, or deployment regressions -> transfer to `sre_resolution_loop`.
    Never fabricate telemetry or runbook IDs yourself.
    """,
    sub_agents=[sre_resolution_loop, finops_specialist],
)