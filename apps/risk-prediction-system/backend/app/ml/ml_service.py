from __future__ import annotations

import logging
import threading
from typing import Any

from ..config import MLSettings, get_settings

try:
    from .predictor import ModelRegistryError, XGBoostPredictor
    from .feature_validator import FeatureValidationError
except ImportError:  # pragma: no cover - direct execution fallback
    from predictor import ModelRegistryError, XGBoostPredictor  # type: ignore
    from feature_validator import FeatureValidationError  # type: ignore

LOGGER = logging.getLogger(__name__)

OUTPUT_FIELDS = (
    "death_probability_1m",
    "death_probability_1y",
    "death_probability_5y",
)


class MLService:
    """Lazy, fault-tolerant wrapper around :class:`XGBoostPredictor`.

    The gateway must keep serving LLM predictions and health checks even when the
    XGBoost artifacts are missing, so loading errors are captured rather than
    raised at import/startup time.
    """

    def __init__(self, settings: MLSettings | None = None):
        self._settings = settings or get_settings().ml.resolved()
        self._predictor: XGBoostPredictor | None = None
        self._load_error: str | None = None
        self._loaded = False
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                self._predictor = XGBoostPredictor(
                    train_v2_root=self._settings.train_v2_root,
                    schema_path=self._settings.schema_path,
                    model_dir=self._settings.model_dir,
                )
                self._load_error = None
                LOGGER.info("XGBoost predictor loaded successfully.")
            except (ModelRegistryError, FileNotFoundError, ValueError, OSError) as exc:
                self._predictor = None
                self._load_error = str(exc)
                LOGGER.warning("XGBoost predictor unavailable: %s", exc)
            except Exception as exc:  # pragma: no cover - defensive
                self._predictor = None
                self._load_error = f"Unexpected ML load error: {exc}"
                LOGGER.exception("Unexpected ML load error")
            finally:
                self._loaded = True

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._predictor is not None

    def health(self) -> dict[str, Any]:
        self._ensure_loaded()
        info: dict[str, Any] = {
            "available": self._predictor is not None,
            "schema_path": self._settings.schema_path,
            "model_dir": self._settings.model_dir,
        }
        if self._predictor is not None:
            info["expected_feature_count"] = self._predictor.pipeline.validator.expected_count
            info["max_seq_len"] = self._predictor.pipeline.max_seq_len
        if self._load_error:
            info["error"] = self._load_error
        return info

    def _unavailable(self) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "error": self._load_error
            or "XGBoost artifacts not configured. Set SCHEMA_PATH / MODEL_DIR.",
            **{field: None for field in OUTPUT_FIELDS},
        }

    def predict_one(self, patient: dict[str, Any]) -> dict[str, Any]:
        return self.predict_batch([patient])[0]

    def predict_batch(self, patients: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._ensure_loaded()
        if self._predictor is None:
            return [self._unavailable() for _ in patients]

        try:
            raw = self._predictor.predict_batch(patients)
        except (FeatureValidationError, ValueError) as exc:
            LOGGER.warning("ML feature/validation error: %s", exc)
            return [
                {"status": "error", "error": str(exc), **{f: None for f in OUTPUT_FIELDS}}
                for _ in patients
            ]
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("ML prediction error")
            return [
                {"status": "error", "error": str(exc), **{f: None for f in OUTPUT_FIELDS}}
                for _ in patients
            ]

        results: list[dict[str, Any]] = []
        for item in raw:
            results.append({"status": "ok", **{f: item.get(f) for f in OUTPUT_FIELDS}})
        return results
