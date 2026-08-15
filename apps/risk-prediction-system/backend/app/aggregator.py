from __future__ import annotations

import json
from typing import Any

from .llm.llm_service import LLMService
from .ml.ml_service import MLService


class PredictionAggregator:
    """Run ML and LLM independently and return both branches side by side."""

    def __init__(self, ml: MLService, llm: LLMService):
        self.ml = ml
        self.llm = llm

    @staticmethod
    def _patient_id(patient: dict[str, Any], index: int) -> str:
        pid = patient.get("id")
        if pid is None or str(pid).strip() == "":
            return f"row_{index + 1}"
        return str(pid)

    def predict_batch(self, patients: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ml_results = self.ml.predict_batch(patients)
        llm_results = self.llm.predict_batch(patients)

        combined: list[dict[str, Any]] = []
        for index, patient in enumerate(patients):
            combined.append(
                {
                    "id": self._patient_id(patient, index),
                    "ml": ml_results[index],
                    "llm": llm_results[index],
                }
            )
        return combined

    def predict_one(self, patient: dict[str, Any]) -> dict[str, Any]:
        return self.predict_batch([patient])[0]


def flatten_for_csv(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten one combined result into a single CSV-friendly row."""
    ml = result.get("ml", {}) or {}
    llm = result.get("llm", {}) or {}

    def llm_probability(field: str) -> Any:
        value = llm.get(field)
        if value is None and llm.get("status") == "ok":
            return "null"
        return value

    return {
        "id": result.get("id"),
        "ml_status": ml.get("status"),
        "ml_death_probability_1m": ml.get("death_probability_1m"),
        "ml_death_probability_1y": ml.get("death_probability_1y"),
        "ml_death_probability_5y": ml.get("death_probability_5y"),
        "ml_error": ml.get("error"),
        "llm_status": llm.get("status"),
        "llm_death_probability_1m": llm_probability("death_probability_1m"),
        "llm_death_probability_1y": llm_probability("death_probability_1y"),
        "llm_death_probability_5y": llm_probability("death_probability_5y"),
        "llm_rationale": llm.get("rationale"),
        "llm_rationale_source": llm.get("rationale_source"),
        "llm_pattern_output": llm.get("pattern_output"),
        "llm_patterns": json.dumps(llm.get("patterns") or [], ensure_ascii=False),
        "llm_pair_count": llm.get("pair_count"),
        "llm_parse_status": llm.get("parse_status"),
        "llm_parse_warnings": "; ".join(llm.get("parse_warnings") or []),
        "llm_error": llm.get("error"),
    }
