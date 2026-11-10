"""Regulateur agent - final compliance gate. Y.3172 P->D decision."""
from __future__ import annotations
from typing import Any, Dict
from .base import BaseAgent


class RegulateurAgent(BaseAgent):
    name = "regulateur"
    description = "Regulator / compliance agent. Final gate before Y.3172 Distributor. Decides PERMIT_AND_DISTRIBUTE or BLOCK_AND_ESCALATE."
    dimensions = [1, 5, 10]

    def _default_system_prompt(self):
        return ("You are the Regulateur (regulator) agent in an Agri AI Readiness pipeline. "
                "You sit at the Y.3172 Policy (P) node and decide whether to forward ML output to the Distributor (D) -> SINK (farmer). "
                "Inputs: agronome recommendation + anomaly flag; juridique legal assessment. "
                "Gates: (1) data quality D1; (2) AI integration D5; (3) policy compliance D10. "
                "If any gate fails, BLOCK_AND_ESCALATE to ADAFSA (doc 7.2). Otherwise PERMIT_AND_DISTRIBUTE. "
                "Respond in JSON: agent, decision, y3172_gate, ai_readiness_dimensions, confidence, narrative, sources_used.")

    def decide(self, agronome_output, juridique_output, soil):
        anomaly = agronome_output.get("ml_result", {}).get("anomaly_detected", False)
        legal_decision = juridique_output.get("decision", "UNKNOWN")
        query = (f"Agronome says: recommended_crop={agronome_output.get('recommended_crop')}, "
                 f"confidence={agronome_output.get('confidence')}, anomaly={anomaly}. "
                 f"Juridique says: decision={legal_decision}, legal_basis={juridique_output.get('legal_basis')}. "
                 "Make the final gate decision: PERMIT_AND_DISTRIBUTE or BLOCK_AND_ESCALATE. ")
        return self.run(query, context={"agronome_output": agronome_output, "juridique_output": juridique_output,
                                        "soil": soil, "anomaly": anomaly, "legal_decision": legal_decision})
