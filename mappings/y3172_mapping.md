# Y.3172 Mapping - Agri AI Readiness Prototype

ITU-T Y.3172 defines a 7-node ML pipeline: SRC -> C -> PP -> M -> P -> D -> SINK, plus MLFO and ML Sandbox.

## Node-by-node mapping

| Node | Role | Prototype component |
|---|---|---|
| SRC | Data source | Soil sensors: N, P, K, temperature, humidity, pH, rainfall |
| C | Collector | ingest.py reads CSV + UAE PolicyData docs |
| PP | Pre-processor | chunk_text() + _detect_anomaly() (validates ranges) |
| M | Model | RandomForestClassifier -> crop_rf.pkl (99.55% accuracy) |
| P | Policy gate | Juridique agent (PDPL Art. 22/24) + Regulateur agent |
| D | Distributor | scenario.py forwards decision to farmer |
| SINK | Target | Farmer via ADAFSA farm monitoring system (doc 7.2) |
| MLFO | Orchestrator | run_demo.py / scenario.run_scenario() |
| Sandbox | Validation env | ChromaDB KB - evidence retrieved before any permit |

## Node-to-factor mapping

| Y.3172 node | AI Readiness factor | Dimension |
|---|---|---|
| SRC, C, PP | Data | D1 |
| M | Research, Open Source | D1, D12 |
| P, MLFO | Standards | D5, D10 |
| D, SINK | Deployment | D5, D13 |
| Sandbox | Sandbox | D12 |
