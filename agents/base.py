"""Base agent class. RAG reasoning loop: embed -> retrieve -> prompt -> LLM -> parse."""
from __future__ import annotations
import json, logging, time
from typing import Any, Dict, List, Optional
import config
from embed import Embedder, get_embedder
from kb import KnowledgeBase
from llm import LLM, get_llm

logger = logging.getLogger(__name__)


class BaseAgent:
    name: str = "base_agent"
    description: str = "Base agent"
    dimensions: List[int] = []

    def __init__(self, kb=None, embedder=None, llm=None):
        self.kb = kb or KnowledgeBase()
        self.embedder = embedder or get_embedder()
        self.llm = llm or get_llm()
        self.system_prompt = self._default_system_prompt()

    def _default_system_prompt(self):
        return (f"You are {self.name}, an AI agent in an Agri AI Readiness pipeline. "
                f"Role: {self.description}. "
                "Operating under UAE law (PDPL Federal Decree-Law 45 of 2021, National Food Security Strategy 2051, ADAFSA SOPs) "
                "and the ITU AI Readiness Framework 2.0. Cite source documents. Respond in JSON.")

    def run(self, query, context=None):
        context = context or {}
        start = time.time()
        query_embedding = self.embedder.embed([query])[0]
        evidence = self._retrieve(query_embedding, context)
        prompt = self._build_prompt(query, evidence, context)
        raw_response = self.llm.chat(self.system_prompt, prompt)
        parsed = self._parse_response(raw_response)
        elapsed = round(time.time() - start, 3)
        audit = {"agent": self.name, "query": query, "context": context, "evidence": evidence,
                 "prompt": prompt, "raw_response": raw_response, "parsed_response": parsed,
                 "llm": self.llm.name, "embedder": self.embedder.name, "elapsed_seconds": elapsed}
        return {**parsed, "_audit": audit}

    def _retrieve(self, query_embedding, context):
        evidence = []
        fw = self.kb.search(query_embedding=query_embedding, n_results=3, collection_name=config.KB_FRAMEWORK)
        evidence.extend(self._flatten(fw, "kb_framework"))
        main = self.kb.search(query_embedding=query_embedding, n_results=config.TOP_K, collection_name=config.KB_MAIN)
        evidence.extend(self._flatten(main, "kb_agri_main"))
        seen, unique = set(), []
        for e in evidence:
            key = e.get("text", "")[:100]
            if key and key not in seen:
                seen.add(key)
                unique.append(e)
        return unique[:12]

    def _flatten(self, results, source):
        out = []
        ids = results.get("ids", [[]])
        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])
        dists = results.get("distances", [[]])
        if ids and isinstance(ids[0], list):
            ids, docs, metas, dists = ids[0], docs[0], metas[0], dists[0]
        for i, doc in enumerate(docs):
            out.append({"text": doc, "metadata": metas[i] if i < len(metas) else {}, "distance": dists[i] if i < len(dists) else None, "source": source})
        return out

    def _build_prompt(self, query, evidence, context):
        ev_text = self._format_evidence(evidence)
        return (f"EVIDENCE FROM KNOWLEDGE BASE:\n{ev_text}\n\n"
                f"CONTEXT:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
                f"QUERY: {query}\n\n"
                f"Respond with a JSON object. Cite source_file names.")

    def _format_evidence(self, evidence):
        if not evidence:
            return "(No relevant evidence found in KB)"
        parts = []
        for i, e in enumerate(evidence[:8], 1):
            meta = e.get("metadata", {})
            source = meta.get("source_file", "unknown")
            country = meta.get("country", "")
            cat = meta.get("document_category", "")
            text = e.get("text", "")[:400]
            parts.append(f"[{i}] Source: {source} | Country: {country} | Category: {cat}\n{text}")
        return "\n\n".join(parts)

    def _parse_response(self, raw):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return {"narrative": raw[:2000], "decision": "PARSE_ERROR", "sources_used": []}
