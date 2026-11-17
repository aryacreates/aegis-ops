import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DB_PATH = Path("aegis_telemetry.db")
CHROMA_DIR = Path(".chroma_store")

SAMPLE_RUNBOOKS = [
    {
        "id": "rb-gke-oom",
        "title": "Runbook: GKE Pod OOMKilled & Memory Pressure",
        "content": (
            "Symptom: CrashLoopBackOff with exit code 137 (OOMKilled) on Cloud Run or GKE. "
            "Root Cause: Container memory limit exceeded due to unbounded in-memory caching or traffic spikes. "
            "Safe Remediation: 1) Verify recent image tag in deployments table. "
            "2) Propose ROLLBACK_DEPLOYMENT to previous stable version (e.g. v2.13.9) or SCALE_MEMORY by 25%. "
            "3) Never delete production PVCs or flush Redis blindly during peak traffic."
        ),
    },
    {
        "id": "rb-sql-pool",
        "title": "Runbook: Cloud SQL Connection Pool Exhaustion",
        "content": (
            "Symptom: HTTP 503 / timeout errors with 'remaining connection slots are reserved'. "
            "Root Cause: Application instances leaking DB connections or missing PgBouncer pooling. "
            "Safe Remediation: 1) Check error_rate and recent deployments. "
            "2) Propose ENABLE_SQL_POOLING via Cloud SQL Auth Proxy and cap max_pool_size per replica. "
            "3) Do NOT restart the primary database instance without DBA approval."
        ),
    },
    {
        "id": "rb-finops-bq",
        "title": "Runbook: BigQuery / Vertex AI Daily Cost Anomaly",
        "content": (
            "Symptom: Daily GCP spend spikes >2x trailing 7-day average. "
            "Root Cause: Unpartitioned SELECT * table scans in BigQuery or uncached LLM loop retries. "
            "Safe Remediation: 1) Query billing_daily to isolate service and team owner. "
            "2) Propose CAP_BIGQUERY_BYTES with maximum_bytes_billed and partition filters. "
            "3) Enable Gemini context caching for repeated system prompts >2048 tokens."
        ),
    },
]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


def _local_deterministic_embedding(text: str, dim: int = 768) -> list[float]:
    """Deterministic offline fallback embedding so unit tests & CI run without API keys."""
    vec = [0.0] * dim
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "big") % dim
        sign = 1.0 if (digest[2] % 2 == 0) else -1.0
        vec[idx] += sign
    return _l2_normalize(vec)


class GeminiEmbedder:
    """Uses gemini-embedding-001 (768-dim MRL) when credentials exist, with safe offline fallback."""

    def __init__(self) -> None:
        self._client = None
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        has_real_key = bool(api_key) and "paste_your" not in api_key and "your_gemini" not in api_key
        if has_real_key or os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
            self._client = genai.Client()

    @property
    def mode(self) -> str:
        return "gemini-embedding-001" if self._client else "offline-hash-fallback"

    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        if not self._client:
            return [_local_deterministic_embedding(t) for t in texts]
        resp = self._client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
        )
        return [_l2_normalize(e.values) for e in resp.embeddings]


class HybridKnowledgeBase:
    """Manages ChromaDB Vector RAG + SQLite Relational Telemetry, Incidents, Traces & Change Requests."""

    def __init__(self) -> None:
        self.embedder = GeminiEmbedder()
        self.chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma.get_or_create_collection(
            name="enterprise_runbooks",
            metadata={"hnsw:space": "cosine"},
        )
        self._seed_runbooks()
        self._init_sql_db()

    def _seed_runbooks(self) -> None:
        if self.collection.count() > 0:
            return
        docs = [rb["content"] for rb in SAMPLE_RUNBOOKS]
        embeddings = self.embedder.embed(docs, task_type="RETRIEVAL_DOCUMENT")
        self.collection.add(
            ids=[rb["id"] for rb in SAMPLE_RUNBOOKS],
            documents=docs,
            metadatas=[{"title": rb["title"]} for rb in SAMPLE_RUNBOOKS],
            embeddings=embeddings,
        )

    def _init_sql_db(self) -> None:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS services (
                service_name TEXT PRIMARY KEY,
                owner_team TEXT,
                sla_tier TEXT,
                p99_latency_ms INTEGER,
                error_rate_pct REAL,
                active_version TEXT,
                status TEXT
            );
            CREATE TABLE IF NOT EXISTS deployments (
                deploy_id TEXT PRIMARY KEY,
                service_name TEXT,
                version TEXT,
                deployed_at TEXT,
                change_summary TEXT
            );
            CREATE TABLE IF NOT EXISTS billing_daily (
                date TEXT,
                service_name TEXT,
                gcp_product TEXT,
                cost_usd REAL
            );
            CREATE TABLE IF NOT EXISTS incidents (
                session_id TEXT PRIMARY KEY,
                query TEXT,
                answer TEXT,
                total_latency_ms REAL,
                avg_tokens_per_sec REAL,
                total_cost_usd REAL,
                guardrail_blocks INTEGER,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS trace_spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                invocation_id TEXT,
                span_type TEXT,
                agent_name TEXT,
                target_name TEXT,
                latency_ms REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                tokens_per_sec REAL,
                cost_usd REAL,
                status TEXT,
                metadata_json TEXT
            );
            CREATE TABLE IF NOT EXISTS change_requests (
                change_id TEXT PRIMARY KEY,
                session_id TEXT,
                service_name TEXT,
                action_type TEXT,
                target_value TEXT,
                rationale TEXT,
                runbook_id TEXT,
                status TEXT,
                approved_by TEXT,
                created_at TEXT,
                resolved_at TEXT
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                change_id TEXT,
                actor TEXT,
                action TEXT,
                details TEXT,
                timestamp TEXT
            );
            """
        )
        if cur.execute("SELECT COUNT(*) FROM services").fetchone()[0] == 0:
            cur.executemany(
                "INSERT INTO services VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    ("checkout-api", "payments-core", "TIER-0", 1850, 4.8, "v2.14.0", "DEGRADED"),
                    ("auth-gateway", "identity", "TIER-0", 120, 0.05, "v4.1.2", "HEALTHY"),
                    ("analytics-etl", "data-eng", "TIER-2", 420, 0.2, "v1.8.0", "COST_ANOMALY"),
                ],
            )
            cur.executemany(
                "INSERT INTO deployments VALUES (?, ?, ?, ?, ?)",
                [
                    ("dep-901", "checkout-api", "v2.14.0", "2026-11-17T09:15:00Z", "Added un-evicted in-memory cart cache"),
                    ("dep-890", "checkout-api", "v2.13.9", "2026-11-12T11:00:00Z", "Stable release with bounded LRU cache"),
                    ("dep-899", "auth-gateway", "v4.1.2", "2026-11-15T14:00:00Z", "Rotated JWT signing key"),
                ],
            )
            cur.executemany(
                "INSERT INTO billing_daily VALUES (?, ?, ?, ?)",
                [
                    ("2026-11-15", "analytics-etl", "BigQuery", 140.50),
                    ("2026-11-16", "analytics-etl", "BigQuery", 148.20),
                    ("2026-11-17", "analytics-etl", "BigQuery", 1980.75),
                    ("2026-11-17", "checkout-api", "GKE", 310.00),
                ],
            )
        conn.commit()
        conn.close()

    def reset_demo_state(self) -> dict[str, str]:
        """Resets services to initial incident state for repeatable live demos."""
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "UPDATE services SET p99_latency_ms=1850, error_rate_pct=4.8, active_version='v2.14.0', status='DEGRADED' WHERE service_name='checkout-api'"
        )
        cur.execute(
            "UPDATE services SET p99_latency_ms=420, error_rate_pct=0.2, active_version='v1.8.0', status='COST_ANOMALY' WHERE service_name='analytics-etl'"
        )
        cur.execute(
            "UPDATE billing_daily SET cost_usd=1980.75 WHERE date='2026-11-17' AND service_name='analytics-etl'"
        )
        conn.commit()
        conn.close()
        return {"status": "demo_state_reset"}

    def add_runbook(self, runbook_id: str, title: str, content: str) -> dict[str, Any]:
        emb = self.embedder.embed([content], task_type="RETRIEVAL_DOCUMENT")[0]
        self.collection.upsert(
            ids=[runbook_id],
            documents=[content],
            metadatas=[{"title": title}],
            embeddings=[emb],
        )
        return {"id": runbook_id, "title": title, "status": "indexed", "embedder": self.embedder.mode}

    def list_runbooks(self) -> list[dict[str, str]]:
        data = self.collection.get(include=["documents", "metadatas"])
        return [
            {"id": doc_id, "title": meta.get("title", doc_id), "content": doc}
            for doc_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]

    def search_runbooks(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        all_rbs = self.list_runbooks()
        if not all_rbs:
            return []
        q_emb = self.embedder.embed([query], task_type="RETRIEVAL_QUERY")[0]
        vec_res = self.collection.query(query_embeddings=[q_emb], n_results=len(all_rbs))

        rrf_scores: dict[str, float] = {}
        doc_map = {rb["id"]: rb for rb in all_rbs}

        for rank, doc_id in enumerate(vec_res["ids"][0]):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank + 1)

        q_tokens = set(re.findall(r"\w+", query.lower()))
        lex_ranked = sorted(
            all_rbs,
            key=lambda rb: len(q_tokens & set(re.findall(r"\w+", (rb["title"] + " " + rb["content"]).lower()))),
            reverse=True,
        )
        for rank, rb in enumerate(lex_ranked):
            rrf_scores[rb["id"]] = rrf_scores.get(rb["id"], 0.0) + 1.0 / (60 + rank + 1)

        top_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:top_k]
        return [{**doc_map[d], "rrf_score": round(rrf_scores[d], 4)} for d in top_ids]

    def run_readonly_sql(self, sql_query: str) -> dict[str, Any]:
        ro_uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(ro_uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in conn.execute(sql_query).fetchmany(50)]
            return {"status": "ok", "row_count": len(rows), "rows": rows}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        finally:
            conn.close()

    def create_change_request(self, session_id: str, proposal: dict[str, Any]) -> dict[str, Any]:
        change_id = f"CR-{uuid.uuid4().hex[:6].upper()}"
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            INSERT INTO change_requests
            (change_id, session_id, service_name, action_type, target_value, rationale, runbook_id, status, approved_by, created_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING_APPROVAL', NULL, ?, NULL)
            """,
            (
                change_id,
                session_id,
                proposal["service_name"],
                proposal["action_type"],
                proposal["target_value"],
                proposal["rationale"],
                proposal["runbook_id"],
                now,
            ),
        )
        conn.execute(
            "INSERT INTO audit_log (change_id, actor, action, details, timestamp) VALUES (?, ?, ?, ?, ?)",
            (change_id, "aegis-ai-agent", "PROPOSED", json.dumps(proposal), now),
        )
        conn.commit()
        conn.close()
        return {**proposal, "change_id": change_id, "status": "PENDING_APPROVAL", "created_at": now}

    def resolve_change_request(self, change_id: str, approve: bool, operator: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cr = conn.execute("SELECT * FROM change_requests WHERE change_id = ?", (change_id,)).fetchone()
        if not cr:
            conn.close()
            raise ValueError(f"Change request {change_id} not found")
        if cr["status"] != "PENDING_APPROVAL":
            conn.close()
            return dict(cr)

        new_status = "EXECUTED" if approve else "REJECTED"
        conn.execute(
            "UPDATE change_requests SET status=?, approved_by=?, resolved_at=? WHERE change_id=?",
            (new_status, operator, now, change_id),
        )

        exec_note = "Rejected by human operator."
        if approve:
            svc = cr["service_name"]
            act = cr["action_type"]
            if svc == "checkout-api" or act in ("ROLLBACK_DEPLOYMENT", "SCALE_MEMORY", "ENABLE_SQL_POOLING"):
                target_ver = cr["target_value"] if "v" in str(cr["target_value"]) else "v2.13.9"
                conn.execute(
                    "UPDATE services SET p99_latency_ms=115, error_rate_pct=0.02, active_version=?, status='HEALTHY' WHERE service_name=?",
                    (target_ver, svc),
                )
                exec_note = f"Applied {act} ({cr['target_value']}) on {svc}. Health restored to p99=115ms, errors=0.02%."
            elif svc == "analytics-etl" or act == "CAP_BIGQUERY_BYTES":
                conn.execute("UPDATE services SET status='HEALTHY' WHERE service_name=?", (svc,))
                conn.execute(
                    "UPDATE billing_daily SET cost_usd=150.00 WHERE date='2026-11-17' AND service_name='analytics-etl'"
                )
                exec_note = f"Applied BigQuery quota cap ({cr['target_value']}) on {svc}. Spend stabilized at $150/day."

        conn.execute(
            "INSERT INTO audit_log (change_id, actor, action, details, timestamp) VALUES (?, ?, ?, ?, ?)",
            (change_id, operator, new_status, exec_note, now),
        )
        updated = dict(conn.execute("SELECT * FROM change_requests WHERE change_id = ?", (change_id,)).fetchone())
        conn.commit()
        conn.close()
        return {**updated, "execution_note": exec_note}

    def save_incident(self, session_id: str, query: str, answer: str, telemetry: dict[str, Any]) -> None:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO incidents
            (session_id, query, answer, total_latency_ms, avg_tokens_per_sec, total_cost_usd, guardrail_blocks, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                query,
                answer,
                telemetry["total_latency_ms"],
                telemetry["avg_tokens_per_sec"],
                telemetry["total_cost_usd"],
                telemetry["guardrail_blocks"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        cur.execute("DELETE FROM trace_spans WHERE session_id = ?", (session_id,))
        for s in telemetry.get("spans", []):
            cur.execute(
                """
                INSERT INTO trace_spans
                (session_id, invocation_id, span_type, agent_name, target_name, latency_ms,
                 input_tokens, output_tokens, tokens_per_sec, cost_usd, status, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    s.get("invocation_id", ""),
                    s.get("span_type", ""),
                    s.get("agent_name", ""),
                    s.get("target_name", ""),
                    s.get("latency_ms", 0.0),
                    s.get("input_tokens", 0),
                    s.get("output_tokens", 0),
                    s.get("tokens_per_sec", 0.0),
                    s.get("cost_usd", 0.0),
                    s.get("status", "ok"),
                    json.dumps(s.get("metadata", {})),
                ),
            )
        conn.commit()
        conn.close()

    def list_incidents(self, limit: int = 20) -> list[dict[str, Any]]:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()]
        conn.close()
        return rows

    def get_persisted_traces(self, session_id: str) -> dict[str, Any] | None:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        inc = conn.execute("SELECT * FROM incidents WHERE session_id = ?", (session_id,)).fetchone()
        spans = conn.execute("SELECT * FROM trace_spans WHERE session_id = ? ORDER BY id ASC", (session_id,)).fetchall()
        conn.close()
        if not inc:
            return None
        span_list = []
        for r in spans:
            d = dict(r)
            d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
            span_list.append(d)
        return {
            "session_id": session_id,
            "total_spans": len(span_list),
            "total_latency_ms": inc["total_latency_ms"],
            "avg_tokens_per_sec": inc["avg_tokens_per_sec"],
            "total_cost_usd": inc["total_cost_usd"],
            "guardrail_blocks": inc["guardrail_blocks"],
            "spans": span_list,
        }

    def get_telemetry_overview(self) -> dict[str, Any]:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        overview = {
            "services": [dict(r) for r in conn.execute("SELECT * FROM services").fetchall()],
            "deployments": [dict(r) for r in conn.execute("SELECT * FROM deployments ORDER BY deployed_at DESC").fetchall()],
            "billing_daily": [dict(r) for r in conn.execute("SELECT * FROM billing_daily ORDER BY date DESC").fetchall()],
            "change_requests": [dict(r) for r in conn.execute("SELECT * FROM change_requests ORDER BY created_at DESC LIMIT 20").fetchall()],
            "audit_log": [dict(r) for r in conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 30").fetchall()],
            "runbooks": self.list_runbooks(),
            "embedder_mode": self.embedder.mode,
        }
        conn.close()
        return overview