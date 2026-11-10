"""Knowledge Base backed by ChromaDB. Three collections."""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4
import chromadb
from chromadb.config import Settings as ChromaSettings
import config

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self, persist_dir=None):
        persist_dir = persist_dir or str(config.CHROMADB_DIR)
        self._client = chromadb.PersistentClient(path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False))
        self._collections: Dict[str, chromadb.Collection] = {}

    def _col(self, name):
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        return self._collections[name]

    @property
    def main(self):
        return self._col(config.KB_MAIN)
    @property
    def framework(self):
        return self._col(config.KB_FRAMEWORK)
    @property
    def decisions(self):
        return self._col(config.KB_DECISIONS)

    def add_documents(self, texts, embeddings, metadatas, collection_name=None):
        col = self._col(collection_name) if collection_name else self.main
        ids = [str(uuid4()) for _ in texts]
        for i in range(0, len(texts), 5000):
            end = min(i + 5000, len(texts))
            col.add(ids=ids[i:end], documents=texts[i:end], embeddings=embeddings[i:end], metadatas=metadatas[i:end])
        logger.info("Added %d chunks to '%s'", len(texts), col.name)
        return ids

    def add_decision(self, decision_text, embedding, metadata):
        doc_id = str(uuid4())
        metadata.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        metadata.setdefault("entity_type", "decision")
        self.decisions.add(ids=[doc_id], documents=[decision_text], embeddings=[embedding], metadatas=[metadata])
        return doc_id

    def search(self, query_embedding, n_results=config.TOP_K, collection_name=None, where=None):
        col = self._col(collection_name) if collection_name else self.main
        count = col.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        kwargs = {"query_embeddings": [query_embedding], "n_results": min(n_results, count), "include": ["documents", "metadatas", "distances"]}
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    def count(self, collection_name=None):
        col = self._col(collection_name) if collection_name else self.main
        return col.count()

    def stats(self):
        return {"kb_agri_main": self.main.count(), "kb_framework": self.framework.count(), "kb_decisions": self.decisions.count()}

    def reset(self):
        for name in [config.KB_MAIN, config.KB_FRAMEWORK, config.KB_DECISIONS]:
            try:
                self._client.delete_collection(name)
            except Exception:
                pass
        self._collections.clear()
