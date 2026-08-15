import json
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import aggregator, app  # noqa: E402

client = TestClient(app)

SAMPLE = {
    "id": "node_1",
    "基础信息": {"患儿性别": "男", "手术年龄": "0.79", "术式": "活体"},
    "时序随访基础信息": ["术后天数: 1, 30。"],
    "时序检验指标 (纯数值序列)": ["ALB: 21.0, 22.0。", "ALT: 212, 100。"],
    "时序用药记录 (纯数值序列)": ["环孢素: 20.0。"],
}


def test_api_health_reports_branches():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "ml" in body and "llm" in body
    assert "available" in body["ml"]
    assert body["ml"]["available"] is True
    assert body["ml"]["expected_feature_count"] == 22_927
    assert "enabled" in body["llm"]


def test_api_predict_returns_both_branches():
    resp = client.post("/api/predict", json=SAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "node_1"
    assert "status" in body["ml"]
    assert "status" in body["llm"]
    assert body["ml"]["status"] == "ok"
    ml_probabilities = {
        key: body["ml"][key]
        for key in ("death_probability_1m", "death_probability_1y", "death_probability_5y")
    }
    for probability in ml_probabilities.values():
        assert 0.0 <= probability <= 1.0
    # LLM is not configured in tests -> disabled
    assert body["llm"]["status"] == "disabled"
    for hidden_field in (
        "native_probabilities",
        "probability_sources",
        "fallback_fields",
        "native_parse_status",
        "answer_text",
        "raw_response",
    ):
        assert hidden_field not in body["llm"]


def test_api_batch_predict():
    resp = client.post("/api/batch_predict", json=[SAMPLE, SAMPLE])
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2


def test_api_preserves_explicit_llm_null(monkeypatch):
    def predict_batch(patients):
        return [
            {
                "status": "ok",
                "death_probability_1m": 0.1,
                "death_probability_1y": None,
                "death_probability_5y": None,
                "parse_status": "ok",
                "parse_warnings": [],
                "patterns": [],
                "error": None,
            }
            for _ in patients
        ]

    monkeypatch.setattr(aggregator.llm, "predict_batch", predict_batch)

    single = client.post("/api/predict", json=SAMPLE)
    batch = client.post("/api/batch_predict", json=[SAMPLE, SAMPLE])

    assert single.status_code == 200
    assert single.json()["llm"]["status"] == "ok"
    assert single.json()["llm"]["death_probability_1y"] is None
    assert '"death_probability_1y":null' in single.text
    assert batch.status_code == 200
    assert all(
        item["llm"]["death_probability_5y"] is None
        for item in batch.json()["results"]
    )


def test_job_lifecycle_json_upload():
    payload = json.dumps([SAMPLE, SAMPLE]).encode("utf-8")
    resp = client.post(
        "/api/jobs",
        files={"file": ("batch.json", payload, "application/json")},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # poll until completed
    status = None
    for _ in range(50):
        meta = client.get(f"/api/jobs/{job_id}").json()
        status = meta["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert status == "completed", meta

    results = client.get(f"/api/jobs/{job_id}/results").json()
    assert results["total"] == 2
    assert len(results["results"]) == 2

    csv_resp = client.get(f"/api/jobs/{job_id}/download?format=csv")
    assert csv_resp.status_code == 200
    assert "ml_status" in csv_resp.text
    assert "llm_status" in csv_resp.text
    assert "llm_rationale_source" in csv_resp.text
    assert "llm_native_probability_1m" not in csv_resp.text
    assert "llm_probability_source_1y" not in csv_resp.text
    assert "llm_fallback_fields" not in csv_resp.text
    assert "llm_answer_text" not in csv_resp.text
    assert "llm_native_parse_status" not in csv_resp.text

    json_resp = client.get(f"/api/jobs/{job_id}/download?format=json")
    assert json_resp.status_code == 200


def test_job_not_found():
    assert client.get("/api/jobs/does-not-exist").status_code == 404
