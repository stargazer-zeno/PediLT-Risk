from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

try:
    from .feature_pipeline import FeaturePipeline, default_train_v2_root
    from .feature_validator import FeatureValidationError
except ImportError:  # pragma: no cover - supports direct script execution
    from feature_pipeline import FeaturePipeline, default_train_v2_root
    from feature_validator import FeatureValidationError

LOGGER = logging.getLogger(__name__)

TARGETS = {
    "Label_1m": ("xgb_sequence_Label_1m.json", "death_probability_1m"),
    "Label_1y": ("xgb_sequence_Label_1y.json", "death_probability_1y"),
    "Label_5y": ("xgb_sequence_Label_5y.json", "death_probability_5y"),
}

EXPECTED_FEATURE_COUNT = 22_927
EXPECTED_BOOSTED_ROUNDS = 300
EXPECTED_OBJECTIVE = "binary:logistic"


class ModelRegistryError(RuntimeError):
    """Raised when required model artifacts are missing or incompatible."""


class XGBoostPredictor:
    def __init__(
        self,
        train_v2_root: str | Path | None = None,
        schema_path: str | Path | None = None,
        model_dir: str | Path | None = None,
    ):
        self.train_v2_root = Path(train_v2_root).resolve() if train_v2_root else default_train_v2_root()
        self.model_dir = Path(model_dir).resolve() if model_dir else self.train_v2_root / "xgboost"
        self.pipeline = FeaturePipeline(schema_path=schema_path, train_v2_root=self.train_v2_root)
        if self.pipeline.validator.expected_count != EXPECTED_FEATURE_COUNT:
            raise ModelRegistryError(
                "XGBoost schema feature count mismatch: expected "
                f"{EXPECTED_FEATURE_COUNT}, got {self.pipeline.validator.expected_count}."
            )
        self.models = self._load_models()

    @classmethod
    def from_environment(cls) -> "XGBoostPredictor":
        train_v2_root = os.environ.get("TRAIN_V2_ROOT")
        schema_path = os.environ.get("SCHEMA_PATH")
        model_dir = os.environ.get("MODEL_DIR")
        return cls(train_v2_root=train_v2_root, schema_path=schema_path, model_dir=model_dir)

    def model_paths(self) -> dict[str, Path]:
        return {
            target: self.model_dir / filename
            for target, (filename, _) in TARGETS.items()
        }

    def predict_one(self, raw_patient_data: dict[str, Any]) -> dict[str, float]:
        items = self.pipeline.preprocess(raw_patient_data)
        if len(items) != 1:
            raise ValueError("/predict expects exactly one patient JSON object.")
        return self.predict_batch(items)[0]

    def predict_batch(self, raw_patient_data: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, float]]:
        items = self.pipeline.preprocess(raw_patient_data)
        features = self.pipeline.transform(items, validate=True)
        probabilities = self._predict_features(features)

        outputs: list[dict[str, float]] = []
        for row_idx in range(features.shape[0]):
            outputs.append(
                {
                    output_name: float(probabilities[output_name][row_idx])
                    for _, output_name in TARGETS.values()
                }
            )
        return outputs

    def export_features(self, raw_patient_data: Any, output_csv_path: str | Path):
        return self.pipeline.transform(
            raw_patient_data,
            output_csv_path=output_csv_path,
            as_dataframe=True,
            validate=True,
        )

    def _predict_features(self, features: np.ndarray) -> dict[str, np.ndarray]:
        if features.ndim != 2 or features.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ModelRegistryError(
                "XGBoost input feature shape mismatch: expected "
                f"(*, {EXPECTED_FEATURE_COUNT}), got {features.shape}."
            )
        if np.isinf(features).any():
            raise ModelRegistryError("XGBoost input contains infinite feature values.")

        predictions: dict[str, np.ndarray] = {}
        for target, (_, output_name) in TARGETS.items():
            model = self.models[target]
            proba = model.predict_proba(features)
            if proba.ndim != 2 or proba.shape[1] < 2:
                raise ModelRegistryError(f"Model {target} returned invalid predict_proba shape {proba.shape}.")
            positive_probability = proba[:, 1].astype(float)
            if not np.isfinite(positive_probability).all():
                raise ModelRegistryError(f"Model {target} returned NaN or infinite probabilities.")
            if ((positive_probability < 0.0) | (positive_probability > 1.0)).any():
                raise ModelRegistryError(f"Model {target} returned probability outside [0, 1].")
            predictions[output_name] = positive_probability
        return predictions

    def _load_models(self) -> dict[str, xgb.XGBClassifier]:
        models: dict[str, xgb.XGBClassifier] = {}
        expected_count = self.pipeline.validator.expected_count

        for target, path in self.model_paths().items():
            if not path.exists():
                raise ModelRegistryError(f"Missing XGBoost model for {target}: {path}")
            model = xgb.XGBClassifier()
            model.load_model(str(path))
            booster = model.get_booster()
            feature_count = int(booster.num_features())
            if feature_count != expected_count or feature_count != EXPECTED_FEATURE_COUNT:
                raise ModelRegistryError(
                    f"Model {target} feature count mismatch: expected {expected_count}, "
                    f"model has {feature_count}."
                )
            boosted_rounds = int(booster.num_boosted_rounds())
            if boosted_rounds != EXPECTED_BOOSTED_ROUNDS:
                raise ModelRegistryError(
                    f"Model {target} boosted-round mismatch: expected "
                    f"{EXPECTED_BOOSTED_ROUNDS}, model has {boosted_rounds}."
                )
            objective = str(model.get_xgb_params().get("objective") or "")
            if objective != EXPECTED_OBJECTIVE:
                raise ModelRegistryError(
                    f"Model {target} objective mismatch: expected "
                    f"{EXPECTED_OBJECTIVE!r}, got {objective!r}."
                )
            classes = getattr(model, "classes_", None)
            if classes is not None and len(classes) < 2:
                raise ModelRegistryError(f"Model {target} is not a binary classifier.")
            models[target] = model
            LOGGER.info(
                "Loaded %s from %s (features=%d, boosted_rounds=%d, objective=%s)",
                target,
                path,
                feature_count,
                boosted_rounds,
                objective,
            )
        return models


__all__ = [
    "FeatureValidationError",
    "ModelRegistryError",
    "TARGETS",
    "EXPECTED_FEATURE_COUNT",
    "EXPECTED_BOOSTED_ROUNDS",
    "EXPECTED_OBJECTIVE",
    "XGBoostPredictor",
]
