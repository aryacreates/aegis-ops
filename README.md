# AegisOps — Google Cloud FDE Enterprise Agent Platform

Production-grade multi-agent SRE & FinOps platform built with **Google ADK**, **Gemini**, **ChromaDB Hybrid RRF RAG**, **BFS Dependency Graph Blast Radius**, **HITL Change Approvals**, **SSE Streaming**, **OTLP Trace Export**, and **Terraform IaC**.

## Architecture

```mermaid
flowchart LR
  UI["Web Console (Live SSE + HITL)"] --> API["FastAPI Gateway"]
  API --> ADK["ADK App + Context Cache + Telemetry Plugin"]
  ADK --> ORCH["aegis_orchestrator (Delegation)"]
  ORCH --> LOOP["sre_resolution_loop (Self-Reflection)"]
  ORCH --> FIN["finops_specialist"]
  LOOP --> DIAG["diagnostics_agent (ReAct)"]
  LOOP --> CRIT["safety_critic_agent (exit_loop)"]
  DIAG --> RAG["ChromaDB + RRF Hybrid RAG"]
  DIAG --> SQL["SQLite (Read-Only Guardrailed)"]
  DIAG --> DAG["BFS Blast Radius Graph"]
  API --> HITL["Change Request Ledger -> Live Remediation"]