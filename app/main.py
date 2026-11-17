from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.agents import kb
from app.eval_harness import run_eval
from app.service import execute_investigation, sse_encode, stream_investigation, telemetry_plugin
from app.telemetry_plugin import build_otlp_payload

FRONTEND_FILE = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

api = FastAPI(title="AegisOps Enterprise FDE Platform", version="3.0.0")


class InvestigateRequest(BaseModel):
    query: str
    user_id: str = "fde-engineer"
    session_id: str | None = None


class RunbookRequest(BaseModel):
    id: str
    title: str
    content: str


class ChangeDecisionRequest(BaseModel):
    operator: str = "sre-oncall-lead"


@api.get("/")
async def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_FILE)


@api.post("/investigate")
async def investigate(req: InvestigateRequest) -> dict:
    return await execute_investigation(
        query=req.query, user_id=req.user_id, session_id=req.session_id
    )


@api.post("/investigate/stream")
async def investigate_stream(req: InvestigateRequest) -> StreamingResponse:
    async def event_generator():
        async for item in stream_investigation(
            query=req.query, user_id=req.user_id, session_id=req.session_id
        ):
            yield sse_encode(item)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@api.post("/changes/{change_id}/approve")
async def approve_change(change_id: str, req: ChangeDecisionRequest) -> dict:
    try:
        return kb.resolve_change_request(change_id, approve=True, operator=req.operator)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@api.post("/changes/{change_id}/reject")
async def reject_change(change_id: str, req: ChangeDecisionRequest) -> dict:
    try:
        return kb.resolve_change_request(change_id, approve=False, operator=req.operator)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@api.post("/demo/reset")
async def reset_demo() -> dict:
    return kb.reset_demo_state()


@api.get("/incidents")
async def list_incidents() -> dict:
    return {"incidents": kb.list_incidents()}


@api.get("/traces/{session_id}")
async def get_traces(session_id: str) -> dict:
    if session_id in telemetry_plugin.traces:
        return telemetry_plugin.summarize_session(session_id)
    persisted = kb.get_persisted_traces(session_id)
    if not persisted:
        raise HTTPException(status_code=404, detail="Trace session not found")
    return persisted


@api.get("/traces/{session_id}/otlp")
async def export_otlp(session_id: str) -> JSONResponse:
    trace_data = await get_traces(session_id)
    return JSONResponse(content=build_otlp_payload(trace_data))


@api.get("/telemetry/overview")
async def telemetry_overview() -> dict:
    return kb.get_telemetry_overview()


@api.post("/runbooks")
async def add_runbook(req: RunbookRequest) -> dict:
    return kb.add_runbook(runbook_id=req.id, title=req.title, content=req.content)


@api.post("/eval/run")
async def trigger_eval() -> dict:
    return await run_eval()