# CloudSec Knowledge Graph Assistant

CAI 4505C, Module 10 In-Class Activity. A graph-augmented hybrid RAG assistant for SOC analysts.

- **Multi-hop count questions** ("How many internet-facing prod assets have an unpatched Critical vulnerability that enables T1190?") go to an exact query over the asset, CVE and technique graph.
- **Procedure questions** (time limits, escalation contacts) go to hybrid BM25 + TF-IDF retrieval over 300 runbooks.
- **Retrieved text is data, never instructions.** Runbooks with injected instructions (the planted RB-300) are flagged and never used as a source.
- Every answer is a schema-validated JSON object: `{question, answer, route, graph_facts, sources, flagged_sources, latency_ms}`.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (the deployed product) |
| `kg_rag.py` | Engine: loads data, builds indexes once, `Engine.ask()` returns validated JSON with input checks, a time budget and error handling |
| `cloudsec_assistant_service.py` | Optional FastAPI `/ask` and `/health` endpoints over the same engine |
| `config.json` | Settings exported from the notebook (hybrid, chunk 80/20, top-k 3, injection filter) |
| `requirements.txt` | Pinned dependencies |
| `.env.example` | No secrets are needed; any future keys go in `.env`, which is git-ignored |
| `data/` | Synthetic, seeded knowledge base (RANDOM_STATE = 42) |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Optional API: `pip install fastapi uvicorn` then `uvicorn cloudsec_assistant_service:app --reload`.

Data is synthetic: CVE IDs (CVE-SIM-...) and threat groups (SIM-...) are invented; MITRE ATT&CK technique IDs are real.
