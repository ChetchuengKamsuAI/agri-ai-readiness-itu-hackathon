# Agri AI Readiness Prototype

**ITU AI for Good Sandbox Hackathon - UAE**

An AI agent pipeline that recommends crops to farmers from soil-sensor
data while enforcing UAE data-protection law (PDPL) and the ITU-T
Y.3172 ML pipeline architecture.

## Team

| Role | Name |
|---|---|
| Team Lead | CHEKOUNANG PANDJO HERCO FLORE |
| Member | CHETCHUENG KAMSU PATRICK |
| Member | DONGMO WAMBA ERICA |
| Member | THINDA WANO RIBERY |

## What it does

Three AI agents process soil readings in sequence:
1. **Agronome** (Y.3172 M node) - RandomForest on Crop_recommendation.csv (2200 rows, 22 crops, 99.5% accuracy)
2. **Juridique** (Y.3172 P node) - checks UAE PDPL Art. 22/24, compares with EU AI Act and Singapore framework
3. **Regulateur** (Y.3172 P->D gate) - decides PERMIT_AND_DISTRIBUTE or BLOCK_AND_ESCALATE

## Quick start

```bash
git clone https://github.com/ChetchuengKamsuAI/agri-ai-readiness-itu-hackathon
cd agri-ai-readiness-itu-hackathon
pip install -r requirements.txt
python run_demo.py
```

## Structure

```
agri-ai-readiness/
├── config.py, embed.py, llm.py, kb.py, ingest.py, crop_model.py
├── scenario.py, run_demo.py
├── agents/ (base, agronome, juridique, regulateur)
├── mappings/ (y3172_mapping.md, ai_readiness_mapping.md)
├── policy_gaps.md
├── Technical_Report_Agri_AI_Readiness.pdf
└── data/ (chromadb, models, logs)
```
