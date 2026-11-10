# Policy Gaps Analysis

Domain-level analysis (agriculture AI). UAE = host country; EU/Singapore/OECD = comparative references.

## Documents in the knowledge base

| Doc | Title | Key articles |
|---|---|---|
| 1.1 | UAE National Strategy for AI 2031 | 8 strategic objectives |
| 1.2 | UAE AI Charter 2024 | Ethics principles |
| 1.3 | UAE Cabinet Resolution May 2026 - Agentic AI | Agentic AI framework |
| 2.1 | Federal Decree-Law 45 of 2021 - PDPL | Art. 5-9, 12, 20, 22, 24 |
| 2.2 | Dubai Data Law 26 of 2015 | Data exchange |
| 7.1 | National Food Security Strategy 2051 | Pillar 2 = tech production |
| 7.2 | ADAFSA Farm Registration & Monitoring SOPs | Farm registration, traceability |
| 7.3 | National Framework for Sustainable Development SDGs | SDG alignment |

## Gap 1: No sector-specific AI regulation for agriculture
No UAE rule specifically governing AI in agriculture. Ambiguous if crop recommendation is high-risk or advisory.
- **EU**: AI Act Annex III lists essential services as high-risk; Art. 14 human oversight
- **Singapore**: Model AI Governance Framework GenAI 2024: human-in-the-loop for significant impact
- **OECD**: AI Principles 1.4: human oversight, no sector-specific rules

## Gap 2: No codified sensor-data integrity requirement
PDPL Art. 24 requires informing data subjects but no data-quality threshold for blocking automated decisions.
- **EU**: AI Act Art. 10 (data governance) + Art. 15 (robustness)
- **Singapore**: Recommended, not mandated
- **OECD**: Implied by accountability

## Gap 3: No right to explanation for AI recommendations
PDPL Art. 22 grants right not to be subject to solely automated decisions, but no explicit right to explanation.
- **EU**: AI Act Art. 13 + Art. 86 (right to explanation)
- **Singapore**: Recommended explainability
- **OECD**: Transparency principle

## Gap 4: No cross-border data transfer rules for agricultural data
PDPL regulates transfers generally but not for agricultural data to cloud AI.
- **EU**: GDPR Ch. V + AI Act extraterritoriality
- **Singapore**: PDPA transfer limitation
- **OECD**: Cross-border data flows framework

## Gap 5: No DPIA obligation for agricultural AI
PDPL Art. 20 requires DPIAs for high-risk processing but no published list; agriculture not listed.
- **EU**: GDPR Art. 35 + AI Act Art. 27 (FRIA)
- **Singapore**: Recommended
- **OECD**: Risk assessment in guidelines

## Prototype mitigations

| Gap | Mitigation |
|---|---|
| 1 | Juridique agent treats crop recommendations as high-risk by default |
| 2 | Agronome anomaly detector blocks on out-of-range readings (Y.3172 P-node gate) |
| 3 | Agent returns narrative, sources_used, confidence, alternatives |
| 4 | Prototype runs entirely locally (no cross-border transfer) |
| 5 | Scenario audit log functions as runtime DPIA |
