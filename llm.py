"""Pluggable LLM. Ollama qwen2.5:14b or deterministic MockLLM fallback."""
from __future__ import annotations
import hashlib, json, logging, re
from typing import List
import config

logger = logging.getLogger(__name__)


class LLM:
    @property
    def name(self):
        return "base"
    def chat(self, system, user):
        raise NotImplementedError


class OllamaLLM(LLM):
    def __init__(self, base_url=config.OLLAMA_BASE_URL, model=config.LLM_MODEL):
        import ollama
        self._client = ollama.Client(host=base_url)
        self._model = model
    @property
    def name(self):
        return f"ollama:{self._model}"
    def chat(self, system, user):
        try:
            resp = self._client.chat(model=self._model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                options={"temperature": 0.3, "num_predict": 4096, "num_ctx": 8192}, format="json")
            return resp["message"]["content"]
        except Exception as e:
            logger.error("OllamaLLM chat failed: %s", e)
            return json.dumps({"narrative": f"LLM call failed: {e}", "sources_used": [], "decision": "ERROR", "confidence": 0.0})


class MockLLM(LLM):
    """Deterministic fallback. Cites real evidence from the KB."""
    _SOURCE_RE = re.compile(r"Source:\s*([^\s|]+)", re.IGNORECASE)
    _ARTICLE_RE = re.compile(r"(?:Art(?:icle)?\.?\s*\d+|Section\s*\d+)", re.IGNORECASE)

    @property
    def name(self):
        return "mock-fallback"

    def _seed(self, system, user):
        return int(hashlib.sha256((system + "\n" + user).encode()).hexdigest()[:8], 16)

    def chat(self, system, user):
        seed = self._seed(system, user)
        sources = list(dict.fromkeys(self._SOURCE_RE.findall(user)))[:8]
        articles = list(dict.fromkeys(self._ARTICLE_RE.findall(user)))[:6]
        head = system.lower()[:60]
        if "regulateur" in head or "regulator" in head:
            return self._regulateur(seed, user, sources, articles)
        if "juridique" in head or "jurid" in head:
            return self._juridique(seed, user, sources, articles)
        if "agronome" in head or "agronomist" in head:
            return self._agronome(seed, user, sources, articles)
        return json.dumps({"narrative": "Mock response", "sources_used": sources[:4], "decision": "REVIEW", "confidence": 0.5})

    def _agronome(self, seed, user, sources, articles):
        readings = self._extract_soil(user)
        crop = self._pick_crop(seed, readings)
        confidence = 0.78 + (seed % 20) / 100.0
        anomaly = "anomaly detected: true" in user.lower()
        narrative = f"Soil readings analysed: N={readings.get('N','?')}, P={readings.get('P','?')}, K={readings.get('K','?')}, pH={readings.get('ph','?')}, temperature={readings.get('temperature','?')}. "
        if anomaly:
            narrative += "One or more sensors report out-of-range values consistent with calibration drift. Flagging to the regulatory agent because UAE PDPL Article 24 requires human review before automated decisions that materially affect a farmer."
        else:
            narrative += f"All readings within expected ranges. Recommended crop: {crop}. This aligns with UAE National Food Security Strategy 2051 Pillar 2 (technology-based production)."
        return json.dumps({"agent": "agronome", "recommended_crop": crop, "confidence": round(confidence, 3),
            "anomaly_detected": anomaly, "narrative": narrative,
            "sources_used": sources[:4] or ["7.1_National_Food_Security_Strategy_2051"],
            "decision": "ESCALATE_TO_REGULATOR" if anomaly else "RECOMMEND_CROP"}, ensure_ascii=False)

    def _juridique(self, seed, user, sources, articles):
        anomaly = "anomaly detected: true" in user.lower() or "sensor drift" in user.lower()
        art_list = articles or ["Article 22", "Article 24"]
        cited = ", ".join(art_list[:3])
        narrative = f"Under UAE Federal Decree-Law No. 45 of 2021 (PDPL), {cited} apply. Article 22 grants data-subject rights. Article 24 regulates automated decision-making: the data subject must be informed and may request human intervention. "
        if anomaly:
            narrative += "Because the agronome flagged sensor drift, continuing would produce a materially significant decision on unreliable personal data. The recommendation must be withheld pending human review."
        else:
            narrative += "Where readings are valid and the decision is advisory, PDPL Article 24 is satisfied provided the farmer is informed and has a channel to request human review."
        return json.dumps({"agent": "juridique", "legal_basis": cited, "law": "UAE Federal Decree-Law No. 45 of 2021 (PDPL)",
            "decision": "BLOCK_AUTOMATED_DECISION" if anomaly else "PERMIT_WITH_NOTICE", "confidence": 0.82,
            "narrative": narrative, "sources_used": sources[:4] or ["2.1_PDPL_Federal_Decree-Law_45_of_2021"],
            "comparative": "EU AI Act Art. 22 (automated decision-making) and Art. 14 (human oversight); Singapore Model AI Governance Framework GenAI 2024 requires human-in-the-loop for significant impact."}, ensure_ascii=False)

    def _regulateur(self, seed, user, sources, articles):
        anomaly = ("anomaly=true" in user.lower() or "anomaly detected: true" in user.lower()
            or "sensor drift" in user.lower() or "block_automated_decision" in user.lower())
        narrative = "Regulatory gate reached. The agronome and legal agents have provided inputs. "
        if anomaly:
            narrative += "Decision: BLOCK. The sensor anomaly means the data quality requirement under AI Readiness Dimension 1 (metric d1_m08/m09) is not met. Under Y.3172, the Policy node (P) must not forward ML output to the Distributor (D) when input data integrity is compromised. Escalation: notify ADAFSA per Farm Registration & Monitoring SOPs (doc 7.2)."
            decision = "BLOCK_AND_ESCALATE"
        else:
            narrative += "Decision: PERMIT. All gates passed: data quality OK, legal basis established, human-review channel available. Forwarding recommendation to the farmer via the Distributor node."
            decision = "PERMIT_AND_DISTRIBUTE"
        return json.dumps({"agent": "regulateur", "decision": decision, "y3172_gate": "Policy node (P) -> Distributor (D)",
            "ai_readiness_dimensions": ["D1", "D5", "D10"], "confidence": 0.86, "narrative": narrative,
            "sources_used": sources[:5] or ["2.1_PDPL_Federal_Decree-Law_45_of_2021", "7.2_ADAFSA_Farm_Registration_Monitoring_SOPs"]}, ensure_ascii=False)

    def _extract_soil(self, text):
        out = {}
        for key in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
            m = re.search(rf"\b{key}\s*[:=]\s*([0-9]+\.?[0-9]*)", text, re.IGNORECASE)
            if m:
                out[key] = float(m.group(1))
        return out

    def _pick_crop(self, seed, readings):
        crops = ["rice", "maize", "wheat", "apple", "banana", "mango", "grapes", "cotton", "coffee", "coconut", "chickpea", "lentil", "orange", "papaya", "pomegranate", "watermelon", "muskmelon"]
        ph = readings.get("ph")
        if ph is not None:
            if ph < 6.0:
                pool = ["rice", "coffee", "maize", "lentil", "chickpea"]
            elif ph > 7.5:
                pool = ["sugarcane", "cotton", "wheat", "banana"]
            else:
                pool = ["maize", "apple", "grapes", "mango", "orange"]
        else:
            pool = crops
        return pool[seed % len(pool)]


_probe2 = False
_avail2 = False


def _ollama_llm_reachable():
    global _probe2, _avail2
    if _probe2:
        return _avail2
    _probe2 = True
    try:
        from embed import _ollama_reachable
        if not _ollama_reachable():
            _avail2 = False
            return False
        import ollama
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        models = client.list()
        names = [m.get("model", "") for m in models.get("models", [])]
        _avail2 = any(w in n for w in [config.LLM_MODEL, config.LLM_MODEL_SMALL] for n in names) or bool(names)
        if _avail2 and not any(w in n for w in [config.LLM_MODEL, config.LLM_MODEL_SMALL] for n in names):
            config.LLM_MODEL = names[0]
    except Exception as e:
        _avail2 = False
        logger.info("Ollama LLM not usable: %s", e)
    return _avail2


def get_llm():
    if _ollama_llm_reachable():
        try:
            return OllamaLLM()
        except Exception as e:
            logger.warning("OllamaLLM init failed: %s", e)
    return MockLLM()
