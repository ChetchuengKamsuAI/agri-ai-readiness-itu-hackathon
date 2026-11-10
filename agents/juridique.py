"""Juridique agent - Y.3172 P node. UAE PDPL compliance check."""
from __future__ import annotations
from typing import Any, Dict
from .base import BaseAgent


class JuridiqueAgent(BaseAgent):
    name = "juridique"
    description = "Legal & data-protection agent. Cites UAE PDPL Art. 22/24. Compares with EU AI Act and Singapore framework."
    dimensions = [10]

    def _default_system_prompt(self):
        return ("You are the Juridique (legal) agent in an Agri AI Readiness pipeline. "
                "You are the P (Policy) node in ITU-T Y.3172. "
                "Decide whether UAE PDPL permits the automated crop recommendation. "
                "Key articles: Article 22 (data-subject rights), Article 24 (automated decision-making), Article 12 (sensitive data), Article 20 (DPIAs). "
                "Compare with EU AI Act Art. 14/22 and Singapore Model AI Governance Framework GenAI 2024. "
                "Respond in JSON: agent, legal_basis, law, decision (PERMIT_WITH_NOTICE or BLOCK_AUTOMATED_DECISION), confidence, narrative, sources_used, comparative.")

    def assess_lawfulness(self, agronome_output: Dict[str, Any], soil: Dict[str, float]) -> Dict[str, Any]:
        anomaly = agronome_output.get("ml_result", {}).get("anomaly_detected", False)
        query = (f"The agronome agent recommended: {agronome_output.get('recommended_crop', 'NONE')} "
                 f"with confidence {agronome_output.get('confidence', 0)}. "
                 f"Anomaly detected: {anomaly}. "
                 f"Soil readings tied to a registered farm (identifiable data subject under PDPL). "
                 "Assess lawfulness of issuing this automated recommendation. ")
        if anomaly:
            query += "Because the data is unreliable (sensor drift), continuing would produce a significant automated decision on inaccurate personal data. Is this lawful under PDPL Article 22 and 24? "
        else:
            query += "The decision is advisory (not binding). Is PDPL Article 24 satisfied if the farmer is informed and has a channel to request human review? "
        return self.run(query, context={"agronome_output": agronome_output, "soil": soil, "anomaly": anomaly})
