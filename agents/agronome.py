"""Agronome agent - Y.3172 M node. Crop recommendation + anomaly escalation."""
from __future__ import annotations
from typing import Any, Dict
from .base import BaseAgent


class AgronomeAgent(BaseAgent):
    name = "agronome"
    description = "Agronomist agent. Runs the crop recommendation ML model. Escalates on anomalous readings."
    dimensions = [1, 5]

    def _default_system_prompt(self):
        return ("You are the Agronome agent in an Agri AI Readiness pipeline. "
                "You run a RandomForest crop recommendation model on 7 soil features. "
                "Your role is the M (Model) node in ITU-T Y.3172. "
                "When readings are valid, recommend a crop and cite UAE National Food Security Strategy 2051 (doc 7.1) and ADAFSA SOPs (doc 7.2). "
                "When readings are anomalous, flag ESCALATE_TO_REGULATOR. "
                "Respond in JSON: agent, recommended_crop, confidence, anomaly_detected, narrative, sources_used, decision.")

    def recommend_crop(self, soil: Dict[str, float]) -> Dict[str, Any]:
        try:
            from crop_model import predict
            ml_result = predict(soil)
        except Exception as e:
            ml_result = {"error": str(e), "recommended_crop": None, "confidence": 0.0}
        query = (f"Soil readings: N={soil.get('N')}, P={soil.get('P')}, K={soil.get('K')}, "
                 f"temperature={soil.get('temperature')}, humidity={soil.get('humidity')}, "
                 f"ph={soil.get('ph')}, rainfall={soil.get('rainfall')}. ")
        if ml_result.get("anomaly_detected"):
            query += ("Anomaly detected: True. Details: " + "; ".join(ml_result.get("anomaly_details", []))
                      + ". Do NOT recommend a crop. Escalate to regulator.")
        else:
            query += (f"Anomaly detected: False. ML model predicts: {ml_result.get('recommended_crop')} "
                      f"(confidence {ml_result.get('confidence')}). Confirm and cite the food security strategy.")
        result = self.run(query, context={"soil": soil, "ml_result": ml_result})
        result["ml_result"] = ml_result
        result["soil"] = soil
        return result
