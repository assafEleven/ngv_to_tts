# 🕵️ Investigative Notebook

This is a production-grade, agent-powered Trust & Safety investigation framework. It runs in Vertex AI, integrates with BigQuery and Stripe, and supports natural language (NL) investigations with real data — no mockups, no hallucinations, no simulation.

## 🔍 Purpose

Designed to detect and investigate abuse of TTS models and related systems, this notebook:
- Connects to ElevenLabs data sources (TTS logs, PostHog, Stripe)
- Supports natural language agent workflows
- Maps user networks using device fingerprints, card fingerprints, API key reuse, and more
- Enables attribution, narrative reconstruction, and pattern discovery

## 🧠 Architecture

- `cell_3_investigation_management.py`: Manages investigation lifecycles and logs
- `cell_4_sql_interface.py`: Executes parameterized BigQuery queries + NL → SQL mapping
- `cell_5_analysis_engine.py`: Content analysis engine (OpenAI + fallback rules)
- `cell_7b_agent_launcher.py`: Natural language agent routing and execution
- `cell_7c_agent_runtime_manager.py`: (New) Tracks agent execution status, supports interruption
- `Agent_001_Stripe_Actor_Identification.ipynb`: Modular agent example
- `investigative_agents_notebook.ipynb`: Full NL + manual execution system

## 🔗 Data Sources

| Dataset | Purpose |
|--------|---------|
| `xi-labs.xi_prod.tts_usage_partitioned` | TTS usage logs |
| `analytics-dev-421514.dbt_jho_staging.stg_stripe__charge` | Stripe charge info |
| `eleven-team-safety.device_fingerprint_dataset.device_fingerprint_cleaned` | Device fingerprints |
| `xi-analytics.dbt_marts.fct_posthog_product_events` | PostHog events |
| `xi-analytics.dbt_marts.dim_users` | Joins user ID → email |
| `xi-analytics.dbt_intermediate.dim_stripe_customers` | Connects user ID ↔ Stripe ID |

## 🧠 Capabilities

- ✅ NL queries like `"Investigate all activity tied to john@example.com"`
- ✅ Autonomous agents (scam, spam, fingerprint, Stripe linkage, etc.)
- ✅ Integrated logging and result tracking
- ✅ Cross-entity analysis: API keys, emails, cards, fingerprints, sessions
- ✅ Secure BigQuery execution with audit trail
- 🧪 In development: agent interruption, parallel execution, agent dashboard UI

## 🛠️ Getting Started

```bash
git clone https://github.com/assafEleven/investigative-notebook.git
cd investigative-notebook
# Use inside your Vertex AI environment with BigQuery access

