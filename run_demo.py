"""One-command demo runner: train model + ingest docs + run scenario."""
from __future__ import annotations
import logging
import config
from crop_model import load_data, train_model, save_model, load_model
from ingest import ingest_all, ingest_framework_definitions
from kb import KnowledgeBase
from embed import get_embedder


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("=" * 70)
    print("Agri AI Readiness - Demo Runner")
    print("=" * 70)
    print(f"Dataset: {config.DATASET_DIR}")

    print("\n[1/3] Training crop model ...")
    try:
        load_model()
        print("  Already trained.")
    except FileNotFoundError:
        X, y = load_data()
        print(f"  {len(X)} rows, {y.nunique()} classes.")
        clf, metrics = train_model(X, y)
        save_model(clf, metrics)
        print(f"  Accuracy={metrics['accuracy']} F1={metrics['macro_f1']}")

    print("\n[2/3] Ingesting UAE documents ...")
    emb = get_embedder()
    print(f"  Embedder: {emb.name}")
    kb = KnowledgeBase()
    if kb.count(config.KB_MAIN) > 0 and kb.count(config.KB_FRAMEWORK) > 0:
        print(f"  KB populated: {kb.stats()}")
    else:
        fw = ingest_framework_definitions(kb, emb)
        results = ingest_all(kb, emb)
        total = sum(sum(v.values()) for v in results.values())
        print(f"  Framework: {fw} | Policy chunks: {total}")
        print(f"  KB stats: {kb.stats()}")

    print("\n[3/3] Running 5-event scenario ...\n")
    from scenario import run_scenario
    run_scenario()


if __name__ == "__main__":
    main()
