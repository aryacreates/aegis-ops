# AegisOps — Autonomous Cloud Incident & FinOps Resolution Engine

[![AegisOps CI/CD Gate](https://github.com/aryacreates/aegis-ops/actions/workflows/ci.yml/badge.svg)](https://github.com/aryacreates/aegis-ops/actions/workflows/ci.yml)
![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)
![Google ADK](https://img.shields.io/badge/Google-ADK-4285F4)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688)
![Terraform](https://img.shields.io/badge/IaC-Terraform-7B42BC)

**AegisOps** is a production-grade multi-agent SRE & FinOps platform built with **Google ADK**, **Gemini 2.5 Flash**, **ChromaDB Hybrid RRF RAG**, **BFS Dependency Graph Blast Radius**, **Human-in-the-Loop (HITL)** change approvals, and **OpenTelemetry (OTLP)** observability.

---

## Architecture

```mermaid
flowchart LR
  UI["Web Console (Live SSE + HITL)"] --> API["FastAPI Gateway"]
  API --> ADK["Google ADK App + ContextCacheConfig + TelemetryPlugin"]
  ADK --> ORCH["aegis_orchestrator (Hierarchical Delegation)"]
  ORCH --> LOOP["sre_resolution_loop (Self-Reflection LoopAgent)"]
  ORCH --> FIN["finops_specialist (ReAct)"]
  LOOP --> DIAG["diagnostics_agent (ReAct)"]
  LOOP --> CRIT["safety_critic_agent (exit_loop)"]
  DIAG --> RAG["ChromaDB + Reciprocal Rank Fusion RAG"]
  DIAG --> SQL["SQLite (Read-Only Guardrailed)"]
  DIAG --> DAG["BFS Blast Radius Graph O(V+E)"]
  API --> HITL["HITL Change Ledger -> Live State Remediation"]
