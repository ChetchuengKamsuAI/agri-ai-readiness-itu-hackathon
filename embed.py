"""Pluggable text embedder. Ollama nomic-embed-text or TF-IDF fallback."""
from __future__ import annotations
import logging
from typing import List
import numpy as np
import config

logger = logging.getLogger(__name__)


class Embedder:
    @property
    def name(self) -> str:
        return "base"
    @property
    def dim(self) -> int:
        raise NotImplementedError
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    def __init__(self, base_url=config.OLLAMA_BASE_URL, model=config.EMBED_MODEL):
        import ollama
        self._client = ollama.Client(host=base_url)
        self._model = model
    @property
    def name(self):
        return f"ollama:{self._model}"
    @property
    def dim(self):
        return 768
    def embed(self, texts):
        safe = [t[:config.MAX_CHARS] if len(t) > config.MAX_CHARS else t for t in texts]
        embeddings = []
        batch = 16
        for i in range(0, len(safe), batch):
            chunk = safe[i:i+batch]
            try:
                resp = self._client.embed(model=self._model, input=chunk)
                embeddings.extend(resp["embeddings"])
            except Exception:
                for t in chunk:
                    resp = self._client.embed(model=self._model, input=[t])
                    embeddings.extend(resp["embeddings"])
        return embeddings


class TfidfEmbedder(Embedder):
    """Deterministic sklearn fallback using HashingVectorizer (fixed dim)."""
    def __init__(self, n_features=256):
        from sklearn.feature_extraction.text import HashingVectorizer
        self._vectorizer = HashingVectorizer(
            n_features=n_features, stop_words="english", ngram_range=(1, 2),
            lowercase=True, norm="l2", alternate_sign=False, dtype=float)
        self._dim = n_features
    @property
    def name(self):
        return "tfidf-fallback"
    @property
    def dim(self):
        return self._dim
    def embed(self, texts):
        matrix = self._vectorizer.transform(texts)
        dense = matrix.toarray()
        if dense.shape[1] != self._dim:
            if dense.shape[1] < self._dim:
                pad = np.zeros((dense.shape[0], self._dim - dense.shape[1]))
                dense = np.hstack([dense, pad])
            else:
                dense = dense[:, :self._dim]
        return [row.tolist() for row in dense]


_probe_done = False
_available = False


def _ollama_reachable():
    global _probe_done, _available
    if _probe_done:
        return _available
    _probe_done = True
    try:
        import ollama
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        client.list()
        _available = True
        logger.info("Ollama reachable - using nomic-embed-text")
    except Exception as e:
        _available = False
        logger.info("Ollama not reachable (%s) - using TF-IDF fallback", e)
    return _available


def get_embedder():
    if _ollama_reachable():
        try:
            return OllamaEmbedder()
        except Exception as e:
            logger.warning("OllamaEmbedder init failed: %s", e)
    return TfidfEmbedder()
