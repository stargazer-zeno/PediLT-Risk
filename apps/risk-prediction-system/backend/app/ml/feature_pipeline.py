from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from .feature_validator import FeatureValidator, build_expected_feature_names
except ImportError:  # pragma: no cover - supports direct script execution
    from feature_validator import FeatureValidator, build_expected_feature_names

LOGGER = logging.getLogger(__name__)

EMPTY_STRINGS = {"", "NaN", "None", "nan"}


def default_train_v2_root() -> Path:
    env_value = os.environ.get("TRAIN_V2_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "train_v2"


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() in EMPTY_STRINGS


def clean_num(value: Any) -> float:
    value = str(value).strip()
    if value in EMPTY_STRINGS:
        return np.nan
    if "阴" in value or "(-)" in value or "（-）" in value or "（—）" in value:
        return 0.0
    if "阳" in value or "(+)" in value or "（+）" in value:
        return 1.0
    if "/" in value:
        parts = [part for part in value.split("/") if part.strip()]
        if parts:
            value = parts[0]
    value = value.replace("E*7", "e7").replace("E*", "e")
    match = re.search(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", value)
    if not match:
        return np.nan
    try:
        number = float(match.group())
    except Exception:
        return np.nan
    if number > 1e10 or number < -1000:
        return np.nan
    return number


def split_series_field(field: Any) -> tuple[str | None, list[str]]:
    text = str(field).replace("。", "")
    parts = text.split(": ", 1)
    if len(parts) != 2:
        parts = text.split(":", 1)
    if len(parts) != 2:
        return None, []
    name = parts[0].strip()
    values = [value.strip() for value in parts[1].split(", ")]
    return name, values


@dataclass(frozen=True)
class FeatureBundle:
    static: np.ndarray
    temporal: np.ndarray
    time_mask: np.ndarray
    sequence_lengths: np.ndarray
    truncated: np.ndarray


class FeaturePipeline:
    def __init__(
        self,
        schema_path: str | Path | None = None,
        train_v2_root: str | Path | None = None,
    ):
        self.train_v2_root = Path(train_v2_root).resolve() if train_v2_root else default_train_v2_root()
        self.schema_path = (
            Path(schema_path).resolve()
            if schema_path
            else self.train_v2_root / "datasets" / "schema.json"
        )
        self.schema = load_json(self.schema_path)
        self.max_seq_len = int(self.schema["max_seq_len"])
        self.static_feature_names = list(self.schema["static_feature_names"])
        self.temporal_feature_names = list(self.schema["temporal_feature_names"])
        self.temporal_specs = list(self.schema["temporal_feature_specs"])
        self.expected_feature_names = build_expected_feature_names(self.schema)
        self.validator = FeatureValidator(self.expected_feature_names)
        self._feature_lookup = {
            (spec["type"], spec["name"]): idx for idx, spec in enumerate(self.temporal_specs)
        }

    def preprocess(self, raw_patient_data: Any) -> list[Mapping[str, Any]]:
        if isinstance(raw_patient_data, (str, Path)):
            path = Path(raw_patient_data)
            if path.exists():
                raw_patient_data = load_json(path)
            else:
                raw_patient_data = json.loads(str(raw_patient_data))

        if isinstance(raw_patient_data, Mapping):
            items = [raw_patient_data]
        elif isinstance(raw_patient_data, list):
            items = raw_patient_data
        else:
            raise ValueError(
                "Input must be one patient JSON object, a list of patient objects, "
                "a JSON string, or a JSON file path."
            )

        if not items:
            raise ValueError("Input patient list is empty.")
        for idx, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"Patient item at index {idx} must be a JSON object.")
        return items

    def build_features(self, raw_patient_data: Any) -> FeatureBundle:
        items = self.preprocess(raw_patient_data)
        n_rows = len(items)
        static = np.full((n_rows, len(self.static_feature_names)), np.nan, dtype=np.float32)
        temporal = np.full(
            (n_rows, self.max_seq_len, len(self.temporal_feature_names)),
            np.nan,
            dtype=np.float32,
        )
        time_mask = np.zeros((n_rows, self.max_seq_len), dtype=bool)
        sequence_lengths = np.zeros(n_rows, dtype=np.int32)
        truncated = np.zeros(n_rows, dtype=bool)

        for idx, item in enumerate(items):
            static[idx] = np.asarray(self._encode_static(item), dtype=np.float32)
            temporal[idx], time_mask[idx], sequence_lengths[idx], truncated[idx] = (
                self._parse_temporal(item)
            )

        return FeatureBundle(
            static=static,
            temporal=temporal,
            time_mask=time_mask,
            sequence_lengths=sequence_lengths,
            truncated=truncated,
        )

    def transform(
        self,
        raw_patient_data: Any,
        output_csv_path: str | Path | None = None,
        as_dataframe: bool = False,
        validate: bool = True,
    ):
        bundle = self.build_features(raw_patient_data)
        features = self.flatten_features(bundle)
        if validate:
            self.validator.validate_array(features, self.expected_feature_names)

        if output_csv_path is not None or as_dataframe:
            frame = self.to_dataframe(features)
            if output_csv_path is not None:
                output_path = Path(output_csv_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(output_path, index=False)
                LOGGER.info("Wrote XGBoost features to %s", output_path)
            if as_dataframe:
                return frame
        return features

    def flatten_features(self, bundle: FeatureBundle) -> np.ndarray:
        temporal_flat = bundle.temporal.reshape(bundle.temporal.shape[0], -1)
        seq_len = bundle.sequence_lengths.astype(np.float32).reshape(-1, 1)
        return np.concatenate(
            [
                bundle.static.astype(np.float32),
                temporal_flat.astype(np.float32),
                bundle.time_mask.astype(np.float32),
                seq_len,
            ],
            axis=1,
        ).astype(np.float32, copy=False)

    def to_dataframe(self, features: np.ndarray) -> pd.DataFrame:
        self.validator.validate_array(features, self.expected_feature_names)
        return pd.DataFrame(features, columns=self.expected_feature_names)

    def _encode_static(self, item: Mapping[str, Any]) -> list[float]:
        base = item.get("基础信息", {}) or {}
        row: list[float] = []

        for col in self.schema["continuous_base_cols"]:
            try:
                row.append(float(base.get(col)))
            except Exception:
                row.append(np.nan)

        gender = str(base.get("患儿性别")).strip()
        row.append(1.0 if gender == "男" else (0.0 if gender == "女" else np.nan))

        donor_gender = str(base.get("供体性别")).strip()
        row.append(1.0 if donor_gender == "男" else (0.0 if donor_gender == "女" else np.nan))

        for col, values in self.schema["categorical_base_values"].items():
            value = str(base.get(col)).strip()
            valid = value not in EMPTY_STRINGS
            for candidate in values:
                row.append(1.0 if value == candidate else (0.0 if valid else np.nan))
        return row

    def _raw_sequence_length(self, item: Mapping[str, Any]) -> int:
        lengths: list[int] = []
        for section in [
            "时序随访基础信息",
            "时序检验指标 (纯数值序列)",
            "时序用药记录 (纯数值序列)",
        ]:
            for field in item.get(section, []) or []:
                _, values = split_series_field(field)
                if values:
                    lengths.append(len(values))
        return max(lengths) if lengths else 1

    def _parse_temporal(
        self,
        item: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, int, bool]:
        raw_len = self._raw_sequence_length(item)
        matrix = np.full((raw_len, len(self.temporal_specs)), np.nan, dtype=np.float32)

        def fill_section(section_name: str, spec_type: str) -> None:
            for field in item.get(section_name, []) or []:
                name, values = split_series_field(field)
                feature_idx = self._feature_lookup.get((spec_type, name))
                if feature_idx is None:
                    continue
                for pos, value in enumerate(values[:raw_len]):
                    if spec_type == "med":
                        matrix[pos, feature_idx] = 0.0 if is_empty_value(value) else 1.0
                    elif not is_empty_value(value):
                        matrix[pos, feature_idx] = clean_num(value)

        fill_section("时序随访基础信息", "temporal_base")
        fill_section("时序检验指标 (纯数值序列)", "lab")
        fill_section("时序用药记录 (纯数值序列)", "med")

        truncated = raw_len > self.max_seq_len
        if truncated:
            matrix = matrix[-self.max_seq_len :, :]
            effective_len = self.max_seq_len
        else:
            effective_len = raw_len

        padded = np.full(
            (self.max_seq_len, len(self.temporal_specs)),
            np.nan,
            dtype=np.float32,
        )
        mask = np.zeros(self.max_seq_len, dtype=bool)
        padded[-effective_len:, :] = matrix
        mask[-effective_len:] = True
        return padded, mask, raw_len, truncated


def main() -> None:
    parser = argparse.ArgumentParser(description="Build XGBoost online features from raw patient JSON.")
    parser.add_argument("--input", required=True, help="Path to raw_patient_data.json.")
    parser.add_argument("--output", required=True, help="Path to xgb_features.csv.")
    parser.add_argument("--schema-path", default=None, help="Optional path to train_v2 schema.json.")
    parser.add_argument("--train-v2-root", default=None, help="Optional train_v2 root directory.")
    parser.add_argument("--no-validate", action="store_true", help="Skip feature validator checks.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pipeline = FeaturePipeline(schema_path=args.schema_path, train_v2_root=args.train_v2_root)
    frame = pipeline.transform(
        args.input,
        output_csv_path=args.output,
        as_dataframe=True,
        validate=not args.no_validate,
    )
    print(f"Wrote {frame.shape[0]} rows x {frame.shape[1]} features to {args.output}")


if __name__ == "__main__":
    main()
