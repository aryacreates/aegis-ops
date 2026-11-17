from fastapi.testclient import TestClient

from app.agents import kb
from app.main import api
from app.telemetry_plugin import FORBIDDEN_SQL_PATTERN
from app.topology import compute_blast_radius


def test_blast_radius_bfs() -> None:
    res = compute_blast_radius("checkout-api")
    impacted_names = {item["service"] for item in res["impacted_downstream"]}
    assert impacted_names == {"web-storefront", "mobile-bff", "partner-webhook"}
    assert res["severity"] == "SEV-1 CRITICAL"


def test_sql_guardrail_blocks_destructive_queries() -> None:
    assert FORBIDDEN_SQL_PATTERN.search("DROP TABLE services;") is not None
    assert FORBIDDEN_SQL_PATTERN.search("SELECT * FROM services; DELETE FROM services") is not None
    assert FORBIDDEN_SQL_PATTERN.search("SELECT service_name FROM services") is None


def test_hybrid_rrf_search() -> None:
    results = kb.search_runbooks("OOMKilled exit code 137 crash", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "rb-gke-oom"


def test_fastapi_hitl_and_otlp() -> None:
    client = TestClient(api)
    assert client.get("/").status_code == 200

    ov = client.get("/telemetry/overview").json()
    assert len(ov["services"]) >= 3

    cr = kb.create_change_request(
        "test-sess",
        {
            "service_name": "checkout-api",
            "action_type": "ROLLBACK_DEPLOYMENT",
            "target_value": "v2.13.9",
            "rationale": "Automated pytest rollback check",
            "runbook_id": "rb-gke-oom",
        },
    )
    res = client.post(f"/changes/{cr['change_id']}/approve", json={"operator": "pytest"})
    assert res.status_code == 200
    assert res.json()["status"] == "EXECUTED"

    kb.save_incident(
        "test-sess",
        "test query",
        "test answer",
        {
            "total_latency_ms": 100.0,
            "avg_tokens_per_sec": 90.0,
            "total_cost_usd": 0.0001,
            "guardrail_blocks": 0,
            "spans": [
                {
                    "invocation_id": "inv-1",
                    "span_type": "llm_call",
                    "agent_name": "aegis_orchestrator",
                    "target_name": "gemini-model",
                    "latency_ms": 100.0,
                    "input_tokens": 50,
                    "output_tokens": 25,
                    "tokens_per_sec": 90.0,
                    "cost_usd": 0.0001,
                    "status": "ok",
                }
            ],
        },
    )
    otlp = client.get("/traces/test-sess/otlp")
    assert otlp.status_code == 200
    assert "resourceSpans" in otlp.json()
    kb.reset_demo_state()