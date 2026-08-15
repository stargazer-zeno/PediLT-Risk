from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


class FeatureValidationError(ValueError):
    """Raised when online features do not match the training schema."""


def build_expected_feature_names(schema: dict) -> list[str]:
    names = list(schema["static_feature_names"])
    temporal_names = schema["temporal_feature_names"]
    max_seq_len = int(schema["max_seq_len"])

    for step in range(max_seq_len):
        names.extend(f"T{step:03d}_{name}" for name in temporal_names)
    names.extend(f"Mask_T{step:03d}" for step in range(max_seq_len))
    names.append("sequence_length")
    return names


class FeatureValidator:
    def __init__(self, expected_feature_names: Sequence[str]):
        self.expected_feature_names = list(expected_feature_names)
        self.expected_count = len(self.expected_feature_names)

    @classmethod
    def from_schema(cls, schema: dict) -> "FeatureValidator":
        return cls(build_expected_feature_names(schema))

    @classmethod
    def from_schema_path(cls, schema_path: str | Path) -> "FeatureValidator":
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        return cls.from_schema(schema)

    def validate_feature_names(self, feature_names: Iterable[str]) -> None:
        actual = list(feature_names)
        if len(actual) != self.expected_count:
            message = (
                f"Feature count mismatch: expected {self.expected_count}, "
                f"got {len(actual)}."
            )
            LOGGER.error(message)
            raise FeatureValidationError(message)

        if actual != self.expected_feature_names:
            mismatch_idx = next(
                (
                    idx
                    for idx, (expected, got) in enumerate(
                        zip(self.expected_feature_names, actual)
                    )
                    if expected != got
                ),
                None,
            )
            if mismatch_idx is None:
                mismatch_idx = min(len(actual), self.expected_count)
            expected_name = (
                self.expected_feature_names[mismatch_idx]
                if mismatch_idx < self.expected_count
                else "<missing>"
            )
            actual_name = actual[mismatch_idx] if mismatch_idx < len(actual) else "<missing>"
            message = (
                "Feature order/name mismatch at position "
                f"{mismatch_idx}: expected {expected_name!r}, got {actual_name!r}."
            )
            LOGGER.error(message)
            raise FeatureValidationError(message)

    def validate_array(
        self,
        features,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        array = np.asarray(features)
        if array.ndim != 2:
            message = f"Feature matrix must be 2D, got shape {array.shape}."
            LOGGER.error(message)
            raise FeatureValidationError(message)

        if array.shape[1] != self.expected_count:
            message = (
                f"Feature count mismatch: expected {self.expected_count}, "
                f"got {array.shape[1]}."
            )
            LOGGER.error(message)
            raise FeatureValidationError(message)

        if feature_names is not None:
            self.validate_feature_names(feature_names)

        if array.dtype.kind not in {"f", "i", "u", "b"}:
            message = (
                "Feature dtype mismatch: expected numeric float32-compatible "
                f"matrix, got dtype {array.dtype}."
            )
            LOGGER.error(message)
            raise FeatureValidationError(message)

        converted = array.astype(np.float32, copy=False)
        inf_locations = np.argwhere(np.isinf(converted))
        if len(inf_locations):
            row, col = inf_locations[0]
            name = self.expected_feature_names[int(col)]
            message = f"Invalid infinite value at row {int(row)}, feature {name!r}."
            LOGGER.error(message)
            raise FeatureValidationError(message)
        return converted

    def validate_frame(self, frame) -> np.ndarray:
        self.validate_feature_names(frame.columns)
        for col in frame.columns:
            if not np.issubdtype(frame[col].dtype, np.number):
                message = (
                    f"Feature dtype mismatch for {col!r}: expected numeric, "
                    f"got {frame[col].dtype}."
                )
                LOGGER.error(message)
                raise FeatureValidationError(message)
        return self.validate_array(frame.to_numpy(), list(frame.columns))
