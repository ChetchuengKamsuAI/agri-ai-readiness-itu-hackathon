"""Error scenario runner - 5 events with full audit trail."""
from __future__ import annotations
import argparse, json, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import config
from agents import AgronomeAgent, JuridiqueAgent, RegulateurAgent
from kb import KnowledgeBase
from embed import get_embedder
from llm import get_llm

logger = logging.getLogger(__name__)

EVENTS: List[Dict[str, Any]] = [
    {"event_id": 1, "title": "Normal baseline reading",
     "description": "A registered farm in Abu Dhabi sends a routine soil reading. All 7 features within physiological ranges.",
     "soil": {"N": 90, "P": 42, "K": 43, "temperature": 20.9, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9},
     "expected_outcome": "PERMIT_AND_DISTRIBUTE"},
    {"event_id": 2, "title": "pH sensor drift detected",
     "description": "The next reading shows pH=11.5, outside valid range [3.0, 10.0]. The ML model flags an anomaly.",
     "soil": {"N": 88, "P": 40, "K": 45, "temperature": 21.2, "humidity": 80.5, "ph": 11.5, "rainfall": 198.0},
     "expected_outcome": "ESCALATE_TO_REGULATOR"},
    {"event_id": 3, "title": "Juridique assesses lawfulness under PDPL Article 24",
     "description": "Juridique receives the escalation. Issuing an automated recommendation on unreliable data would violate PDPL Art. 22 and 24.",
     "soil": {"N": 88, "P": 40, "K": 45, "temperature": 21.2, "humidity": 80.5, "ph": 11.5, "rainfall": 198.0},
     "expected_outcome": "BLOCK_AUTOMATED_DECISION"},
    {"event_id": 4, "title": "Regulateur blocks and escalates to ADAFSA",
     "description": "Regulateur combines agronome + juridique outputs. Y.3172 P node must not forward to D when data integrity is compromised.",
     "soil": {"N": 88, "P": 40, "K": 45, "temperature": 21.2, "humidity": 80.5, "ph": 11.5, "rainfall": 198.0},
     "expected_outcome": "BLOCK_AND_ESCALATE"},
    {"event_id": 5, "title": "Sensor re-calibrated, decision permitted",
     "description": "ADAFSA re-calibrates the sensor. Next reading pH=7.1. Full chain permits and distributes.",
     "soil": {"N": 91, "P": 41, "K": 44, "temperature": 21.0, "humidity": 81.5, "ph": 7.1, "rainfall": 201.0},
     "expected_outcome": "PERMIT_AND_DISTRIBUTE"},
]


def play_event(event, agronome, juridique, regulateur):
    ts = datetime.now(timezone.utc).isoformat()
    soil = event["soil"]
    owner = {2: "agronome", 3: "juridique", 4: "regulateur"}.get(event["event_id"], "full")
    print(f"\n{'='*70}\nEVENT {event['event_id']}: {event['title']}\n{'='*70}")
    print(f"Soil: {soil} | Expected: {event['expected_outcome']}")
    agronome_out = agronome.recommend_crop(soil)
    print(f"[Agronome] {agronome_out.get('decision')} crop={agronome_out.get('recommended_crop')} anomaly={agronome_out.get('anomaly_detected')}")
    juridique_out = juridique.assess_lawfulness(agronome_out, soil)
    print(f"[Juridique] {juridique_out.get('decision')} basis={juridique_out.get('legal_basis')}")
    regulateur_out = regulateur.decide(agronome_out, juridique_out, soil)
    print(f"[Regulateur] {regulateur_out.get('decision')}")
    try:
        emb = regulateur.embedder.embed([f"{regulateur_out.get('decision')} {regulateur_out.get('narrative','')[:200]}"])[0]
        regulateur.kb.add_decision(
            decision_text=json.dumps({"event_id": event["event_id"], "decision": regulateur_out.get("decision")}, ensure_ascii=False),
            embedding=emb,
            metadata={"event_id": str(event["event_id"]), "agent": "regulateur", "decision": str(regulateur_out.get("decision")), "soil_ph": str(soil.get("ph")), "timestamp": ts})
    except Exception as e:
        logger.warning("kb_decisions log failed: %s", e)
    return {"event_id": event["event_id"], "title": event["title"], "description": event["description"],
            "soil": soil, "expected_outcome": event["expected_outcome"], "owner_agent": owner, "timestamp": ts,
            "agronome": _strip(agronome_out), "juridique": _strip(juridique_out), "regulateur": _strip(regulateur_out)}


def _strip(out):
    audit = out.pop("_audit", {})
    return {**out, "_audit": audit}


def run_scenario():
    log_path = config.SCENARIO_LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from crop_model import load_model
        load_model()
    except FileNotFoundError:
        print("Training crop model ...")
        from crop_model import load_data, train_model, save_model
        X, y = load_data()
        save_model(*train_model(X, y))
    kb = KnowledgeBase()
    if kb.count(config.KB_MAIN) == 0:
        print("Ingesting documents ...")
        from ingest import ingest_all, ingest_framework_definitions
        emb = get_embedder()
        ingest_framework_definitions(kb, emb)
        ingest_all(kb, emb)
    emb = get_embedder()
    llm = get_llm()
    print(f"Embedder: {emb.name} | LLM: {llm.name}")
    agronome = AgronomeAgent(kb=kb, embedder=emb, llm=llm)
    juridique = JuridiqueAgent(kb=kb, embedder=emb, llm=llm)
    regulateur = RegulateurAgent(kb=kb, embedder=emb, llm=llm)
    results = []
    with open(log_path, "w", encoding="utf-8") as f:
        for event in EVENTS:
            result = play_event(event, agronome, juridique, regulateur)
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    print(f"\n{'='*70}\nScenario complete. Audit log: {log_path}")
    print(f"  Events: {len(results)} | Decisions in kb_decisions: {kb.count(config.KB_DECISIONS)}")
    for r in results:
        ag = r["agronome"].get("decision", "?")
        ju = r["juridique"].get("decision", "?")
        re_ = r["regulateur"].get("decision", "?")
        owner = r.get("owner_agent", "full")
        actual = {"agronome": ag, "juridique": ju, "regulateur": re_}.get(owner, re_)
        match = "OK" if actual == r["expected_outcome"] else "MISMATCH"
        print(f"  Event {r['event_id']} ({owner}): ag={ag} ju={ju} re={re_} | expected={r['expected_outcome']} [{match}]")
    return log_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_scenario()
    print(f"\nKB stats: {KnowledgeBase().stats()}")


if __name__ == "__main__":
    main()
