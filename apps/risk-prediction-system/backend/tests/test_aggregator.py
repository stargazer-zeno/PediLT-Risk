from app.aggregator import PredictionAggregator, flatten_for_csv


class FakeMLService:
    def __init__(self, probability_1y=0.03):
        self.probability_1y = probability_1y

    def predict_batch(self, patients):
        return [
            {
                "status": "ok",
                "death_probability_1m": 0.02,
                "death_probability_1y": self.probability_1y,
                "death_probability_5y": 0.04,
                "error": None,
            }
            for _ in patients
        ]


class FakeLLMService:
    def __init__(self):
        self.calls = []

    def predict_batch(self, patients):
        self.calls.append(patients)
        return [
            {
                "status": "ok",
                "death_probability_1m": 0.0129,
                "death_probability_1y": None,
                "death_probability_5y": None,
                "rationale": "总体风险原因",
                "rationale_source": "vllm_second_pass",
                "patterns": [],
                "parse_status": "ok",
                "parse_warnings": [],
                "error": None,
            }
            for _ in patients
        ]


def test_aggregator_keeps_ml_and_llm_probabilities_independent():
    llm = FakeLLMService()
    aggregator = PredictionAggregator(FakeMLService(), llm)
    result = aggregator.predict_one({"id": "patient_1"})
    changed_ml_result = PredictionAggregator(FakeMLService(0.99), llm).predict_one(
        {"id": "patient_1"}
    )

    assert len(llm.calls) == 2
    assert result["ml"]["death_probability_1y"] == 0.03
    assert changed_ml_result["ml"]["death_probability_1y"] == 0.99
    assert result["llm"]["death_probability_1y"] is None
    assert changed_ml_result["llm"]["death_probability_1y"] is None
    row = flatten_for_csv(result)
    assert row["llm_death_probability_1y"] == "null"
    assert row["llm_death_probability_5y"] == "null"
    assert row["llm_rationale"] == "总体风险原因"
    assert row["llm_rationale_source"] == "vllm_second_pass"
    assert row["llm_parse_status"] == "ok"
    for hidden_column in (
        "llm_native_probability_1m",
        "llm_probability_source_1y",
        "llm_fallback_fields",
        "llm_answer_text",
        "llm_native_parse_status",
    ):
        assert hidden_column not in row


def test_csv_leaves_non_model_null_probabilities_empty():
    result = {
        "id": "patient_1",
        "ml": {"status": "ok"},
        "llm": {
            "status": "error",
            "death_probability_1m": None,
            "death_probability_1y": None,
            "death_probability_5y": None,
        },
    }

    row = flatten_for_csv(result)

    assert row["llm_death_probability_1m"] is None
    assert row["llm_death_probability_1y"] is None
    assert row["llm_death_probability_5y"] is None
