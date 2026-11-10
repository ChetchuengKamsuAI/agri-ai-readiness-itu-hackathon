"""Document ingestion pipeline. Adapted from ITUAIReadiness reference."""
from __future__ import annotations
import argparse, logging, re, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import config
from embed import Embedder, get_embedder
from kb import KnowledgeBase

logger = logging.getLogger(__name__)


def parse_file(filepath: Path) -> str:
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(filepath)
    if suffix in (".txt", ".md", ".csv"):
        return filepath.read_text(encoding="utf-8", errors="replace")
    if suffix == ".html":
        return _strip_html(filepath.read_text(encoding="utf-8", errors="replace"))
    return filepath.read_text(encoding="utf-8", errors="replace")


def _parse_pdf(filepath):
    try:
        import fitz
        doc = fitz.open(str(filepath))
        return "\n\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.warning("PDF parse failed for %s: %s", filepath.name, e)
        return ""


def _strip_html(html):
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP, max_chars=config.MAX_CHARS):
    if not text or not text.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], []
    for para in paragraphs:
        for word in para.split():
            current.append(word)
            if len(current) >= chunk_size:
                chunks.append(" ".join(current))
                current = current[-overlap:] if overlap > 0 else []
    if current:
        last = " ".join(current)
        if len(current) > 20 or not chunks:
            chunks.append(last)
        elif chunks:
            chunks[-1] += " " + last
    safe = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            safe.append(chunk)
        else:
            for i in range(0, len(chunk), max_chars):
                piece = chunk[i:i+max_chars].strip()
                if piece:
                    safe.append(piece)
    return safe


_CATEGORY_MAP = {"01_AI_Strategy_and_Governance": "AI_strategy", "02_Data_Protection": "data_protection", "07_Food_Security_and_Agriculture": "agriculture"}
_COUNTRY_RE = re.compile(r"(?i)(UAE|united.arab.emirates|emirates)|(EU|european)|(singapore)|(OECD)")


def build_metadata(filepath, chunk_index, total_chunks, category):
    fname = filepath.name
    m = _COUNTRY_RE.search(fname)
    country = "UAE"
    if m:
        if m.group(1):
            country = "UAE"
        elif m.group(2):
            country = "EU"
        elif m.group(3):
            country = "Singapore"
        elif m.group(4):
            country = "OECD"
    return {"source_file": fname, "document_category": _CATEGORY_MAP.get(category, category), "country": country,
            "chunk_index": str(chunk_index), "total_chunks": str(total_chunks),
            "ingested_at": datetime.now(timezone.utc).isoformat(), "policy_folder": category}


Y3172_NODES = {
    "SRC": "Source of data input to the ML pipeline (sensors, user equipment, application function).",
    "C": "Collector - gathers data from one or more SRC nodes.",
    "PP": "Pre-processor - data cleaning, transformation, normalisation, feature engineering.",
    "M": "Model - where ML models are trained, validated and execute inference.",
    "P": "Policy - enforces policies on ML output before distribution.",
    "D": "Distributor - routes ML output to one or more SINK nodes.",
    "SINK": "Target of ML output - actuators, dashboards, decision systems, feedback loops.",
    "MLFO": "ML Function Orchestrator - manages and orchestrates all pipeline nodes.",
    "SANDBOX": "ML Sandbox - validation environment using simulated underlay networks.",
}

AI_READINESS_DIMENSIONS = {
    1: ("Data/Model Marketplace", ["data", "opensource"]),
    2: ("Generated Content Marketplace", ["data", "opensource"]),
    3: ("Cross-Domain Correlation Analysis", ["data", "standards"]),
    4: ("Contextualization and Regional Impact", ["data", "research", "deployment"]),
    5: ("Level of Integration of AI in Workflows", ["standards"]),
    6: ("Human Interface", ["standards", "deployment"]),
    7: ("Strategy Alignment", ["data", "research", "standards", "deployment"]),
    8: ("Governance and Coordination", ["standards"]),
    9: ("Government Services and Public Sector AI", ["deployment", "standards"]),
    10: ("AI and Policies", ["standards"]),
    11: ("Skills and Education", ["research", "deployment"]),
    12: ("Ecosystem and Innovation", ["opensource", "sandbox"]),
    13: ("Digital Infrastructure", ["deployment"]),
}


def ingest_framework_definitions(kb, embedder):
    texts, metadatas = [], []
    for node_id, desc in Y3172_NODES.items():
        texts.append(f"Y.3172 Node {node_id}: {desc}")
        metadatas.append({"framework": "Y.3172", "entity_type": "pipeline_node", "node_id": node_id, "source_file": "ITU-T_Y.3172"})
    for dim_id, (name, factors) in AI_READINESS_DIMENSIONS.items():
        texts.append(f"AI Readiness Dimension {dim_id}: {name}. Mapped factors: {', '.join(factors)}.")
        metadatas.append({"framework": "AI_Readiness_2.0", "entity_type": "dimension", "dimension_id": str(dim_id), "dimension_name": name, "factors": ",".join(factors), "source_file": "ITU_AI_Readiness_Framework_2.0"})
    texts.append("EU AI Act (Regulation 2024/1689) - risk tiers: unacceptable, high-risk (Art.14, Art.22), limited-risk, minimal-risk. Penalties up to 7% global turnover.")
    metadatas.append({"framework": "EU_AI_Act", "entity_type": "regulation", "country": "EU", "source_file": "EU_AI_Act_summary"})
    embeddings = embedder.embed(texts)
    kb.add_documents(texts=texts, embeddings=embeddings, metadatas=metadatas, collection_name=config.KB_FRAMEWORK)
    return len(texts)


def ingest_policy_folder(kb, embedder, folder_name):
    folder = config.UAE_POLICY_DIR / folder_name
    if not folder.exists():
        return {}
    supported = {".pdf", ".txt", ".md", ".html"}
    results = {}
    files = sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in supported)
    seen_ids = set()
    chosen = []
    for f in files:
        doc_id = f.stem.split("_", 1)[0]
        if doc_id in seen_ids and f.suffix.lower() in (".md", ".pdf"):
            continue
        chosen.append(f)
        seen_ids.add(doc_id)
    for filepath in chosen:
        try:
            text = parse_file(filepath)
            if not text.strip():
                results[filepath.name] = 0
                continue
            chunks = chunk_text(text)
            if not chunks:
                results[filepath.name] = 0
                continue
            embeddings = embedder.embed(chunks)
            metadatas = [build_metadata(filepath, i, len(chunks), folder_name) for i in range(len(chunks))]
            kb.add_documents(texts=chunks, embeddings=embeddings, metadatas=metadatas, collection_name=config.KB_MAIN)
            results[filepath.name] = len(chunks)
        except Exception as e:
            logger.error("Failed to ingest %s: %s", filepath.name, e)
            results[filepath.name] = 0
    return results


def ingest_all(kb, embedder):
    all_results = {}
    for folder in config.POLICY_FOLDERS:
        all_results[folder] = ingest_policy_folder(kb, embedder, folder)
    return all_results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ingest UAE policy documents")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--framework-only", action="store_true")
    args = parser.parse_args()
    if not config.UAE_POLICY_DIR.exists():
        print(f"ERROR: {config.UAE_POLICY_DIR}")
        sys.exit(1)
    embedder = get_embedder()
    print(f"Embedder: {embedder.name}")
    kb = KnowledgeBase()
    if args.reset:
        kb.reset()
    print("Loading framework definitions ...")
    fw = ingest_framework_definitions(kb, embedder)
    print(f"  Framework: {fw} definitions")
    if args.framework_only:
        print(f"\nKB stats: {kb.stats()}")
        return
    print("\nIngesting UAE policy documents ...")
    results = ingest_all(kb, embedder)
    total = sum(sum(v.values()) for v in results.values())
    print(f"\nTotal: {total} policy chunks ingested")
    print(f"KB stats: {kb.stats()}")


if __name__ == "__main__":
    main()
