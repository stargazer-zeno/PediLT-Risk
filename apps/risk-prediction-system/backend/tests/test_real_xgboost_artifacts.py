from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.ml.feature_pipeline import FeaturePipeline
from app.ml.predictor import (
    EXPECTED_BOOSTED_ROUNDS,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_OBJECTIVE,
    XGBoostPredictor,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
SCHEMA_PATH = ARTIFACTS / "datasets" / "schema.json"
MODEL_DIR = ARTIFACTS / "xgboost"
MODEL_MANIFEST = ARTIFACTS / "model_manifest.json"
LONG_FOLLOWUP_SAMPLE = ROOT / "frontend" / "public" / "samples" / "demo_long_followup.json"


def _require_real_delivery() -> None:
    required = [
        SCHEMA_PATH,
        MODEL_MANIFEST,
        MODEL_DIR / "xgb_sequence_Label_1m.json",
        MODEL_DIR / "xgb_sequence_Label_1y.json",
        MODEL_DIR / "xgb_sequence_Label_5y.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip(f"real XGBoost delivery is not installed: {', '.join(missing)}")


def _load_schema() -> dict:
    _require_real_delivery()
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _sample_patient() -> dict:
    sample_path = LONG_FOLLOWUP_SAMPLE
    return json.loads(sample_path.read_text(encoding="utf-8"))


def _series_patient(length: int) -> dict:
    values = ", ".join(str(index + 1) for index in range(length))
    reference = _sample_patient()
    base_key = next(
        key
        for key, value in reference.items()
        if isinstance(value, dict) and "基础信息" in key
    )
    series_keys = [key for key, value in reference.items() if isinstance(value, list)]
    schema = _load_schema()
    names_by_type = {
        spec_type: next(
            spec["name"] for spec in schema["temporal_feature_specs"] if spec["type"] == spec_type
        )
        for spec_type in ("temporal_base", "lab", "med")
    }
    return {
        "patient_id": f"contract-{length}",
        base_key: {
            **{
                column: "1"
                for column in schema["continuous_base_cols"]
            },
            **{
                column: values[0] if values else ""
                for column, values in schema["categorical_base_values"].items()
            },
        },
        series_keys[0]: [f"{names_by_type['temporal_base']}: {values}"],
        series_keys[1]: [f"{names_by_type['lab']}: {values}"],
        series_keys[2]: [f"{names_by_type['med']}: {values}"],
    }


def test_real_schema_has_expected_contract() -> None:
    schema = _load_schema()

    assert not schema.get("_placeholder", False)
    assert schema["max_seq_len"] == 256
    assert len(schema["static_feature_names"]) == 142
    assert len(schema["temporal_feature_names"]) == 88
    assert (
        len(schema["static_feature_names"])
        + schema["max_seq_len"] * len(schema["temporal_feature_names"])
        + schema["max_seq_len"]
        + 1
        == EXPECTED_FEATURE_COUNT
        == 22_927
    )


def test_delivery_hashes_match_manifest() -> None:
    _require_real_delivery()
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))

    schema_sha = hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()
    assert schema_sha == manifest["schema"]["sha256"]

    for model_info in manifest["models"]:
        model_path = MODEL_DIR / model_info["file_name"]
        model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert model_sha == model_info["sha256"]


def test_real_models_load_with_expected_runtime_contract() -> None:
    _require_real_delivery()
    predictor = XGBoostPredictor(
        schema_path=SCHEMA_PATH,
        model_dir=MODEL_DIR,
        train_v2_root=ARTIFACTS,
    )

    assert set(predictor.models) == {"Label_1m", "Label_1y", "Label_5y"}
    for model in predictor.models.values():
        booster = model.get_booster()
        assert booster.num_features() == EXPECTED_FEATURE_COUNT
        assert booster.num_boosted_rounds() == EXPECTED_BOOSTED_ROUNDS
        assert model.get_xgb_params()["objective"] == EXPECTED_OBJECTIVE


def test_feature_pipeline_matches_real_shape_and_padding_contract() -> None:
    _require_real_delivery()
    pipeline = FeaturePipeline(schema_path=SCHEMA_PATH, train_v2_root=ARTIFACTS)

    short = pipeline.build_features(_series_patient(2))
    assert short.static.shape == (1, 142)
    assert short.temporal.shape == (1, 256, 88)
    assert short.time_mask.shape == (1, 256)
    assert short.sequence_lengths.shape == (1,)
    assert short.static.dtype == np.float32
    assert short.temporal.dtype == np.float32
    assert short.time_mask.dtype == np.bool_
    assert short.sequence_lengths.dtype == np.int32
    assert short.sequence_lengths.tolist() == [2]
    assert short.time_mask[0, :254].tolist() == [False] * 254
    assert short.time_mask[0, 254:].tolist() == [True, True]
    assert np.isnan(short.temporal[0, 0, :]).all()
    assert pipeline.flatten_features(short).shape == (1, EXPECTED_FEATURE_COUNT)
    assert pipeline.flatten_features(short).dtype == np.float32

    exact = pipeline.build_features(_series_patient(256))
    assert exact.sequence_lengths.tolist() == [256]
    assert exact.time_mask.all()
    assert exact.truncated.tolist() == [False]

    long = pipeline.build_features(_series_patient(260))
    assert long.sequence_lengths.tolist() == [260]
    assert long.truncated.tolist() == [True]
    assert long.time_mask.all()
    temporal_name = next(
        spec["name"]
        for spec in pipeline.temporal_specs
        if spec["type"] == "temporal_base"
    )
    temporal_index = pipeline._feature_lookup[("temporal_base", temporal_name)]
    assert long.temporal[0, 0, temporal_index] == pytest.approx(5.0)
    assert long.temporal[0, -1, temporal_index] == pytest.approx(260.0)


def test_feature_pipeline_handles_missing_unknown_and_medication_values() -> None:
    _require_real_delivery()
    pipeline = FeaturePipeline(schema_path=SCHEMA_PATH, train_v2_root=ARTIFACTS)
    schema = _load_schema()
    patient = _series_patient(2)
    base_key = next(key for key, value in patient.items() if isinstance(value, dict))
    base = patient[base_key]

    missing_numeric = schema["continuous_base_cols"][0]
    base[missing_numeric] = ""
    unknown_category = next(iter(schema["categorical_base_values"]))
    base[unknown_category] = "__unknown_category__"

    bundle = pipeline.build_features(patient)
    assert np.isnan(bundle.static[0, 0])
    category_offset = 7
    for column, values in schema["categorical_base_values"].items():
        if column == unknown_category:
            break
        category_offset += len(values)
    category_width = len(schema["categorical_base_values"][unknown_category])
    assert bundle.static[0, category_offset : category_offset + category_width].tolist() == [
        0.0
    ] * category_width

    med_name = next(
        spec["name"] for spec in pipeline.temporal_specs if spec["type"] == "med"
    )
    med_index = pipeline._feature_lookup[("med", med_name)]
    assert bundle.temporal[0, -2:, med_index].tolist() == [1.0, 1.0]


def test_real_models_predict_from_local_patient_json() -> None:
    _require_real_delivery()
    predictor = XGBoostPredictor(
        schema_path=SCHEMA_PATH,
        model_dir=MODEL_DIR,
        train_v2_root=ARTIFACTS,
    )

    predictions = predictor.predict_one(_sample_patient())

    assert set(predictions) == {
        "death_probability_1m",
        "death_probability_1y",
        "death_probability_5y",
    }
    for probability in predictions.values():
        assert np.isfinite(probability)
        assert 0.0 <= probability <= 1.0


def test_long_followup_demo_matches_ml_contract_and_ignores_clinical_events() -> None:
    _require_real_delivery()
    patient = json.loads(LONG_FOLLOWUP_SAMPLE.read_text(encoding="utf-8"))
    pipeline = FeaturePipeline(schema_path=SCHEMA_PATH, train_v2_root=ARTIFACTS)

    bundle = pipeline.build_features(patient)
    features_with_events = pipeline.flatten_features(bundle)
    patient_without_events = copy.deepcopy(patient)
    patient_without_events.pop("临床事件")
    features_without_events = pipeline.transform(patient_without_events)

    assert bundle.sequence_lengths.tolist() == [15]
    assert features_with_events.shape == (1, EXPECTED_FEATURE_COUNT)
    assert features_with_events.dtype == np.float32
    np.testing.assert_allclose(features_with_events, features_without_events, equal_nan=True)

    predictor = XGBoostPredictor(
        schema_path=SCHEMA_PATH,
        model_dir=MODEL_DIR,
        train_v2_root=ARTIFACTS,
    )
    predictions = predictor.predict_one(patient)
    assert len(predictions) == 3
    assert all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in predictions.values())
