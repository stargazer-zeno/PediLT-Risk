import json

import pytest

from app.llm.llm_service import (
    LLMService,
    parse_llm_json,
    parse_sft_pattern_response,
)
from app.llm.llm_client import LLMClientError
from app.llm.prompt_builder import (
    LLMPromptBuilder,
    SFTPatternPromptBuilder,
    build_prompt_builder,
    sanitize_patient,
)


def test_sanitize_removes_leakage():
    patient = {
        "id": "1",
        "真实标签": {"是否死亡": 1, "1m": 0},
        "基础信息": {"患儿性别": "男", "是否死亡": "True", "死亡时间": "2017", "死亡原因": "x"},
        "临床事件": [
            "术后第30天：复查肝功能较前改善。",
            "术后第90天：患儿死亡。",
            {"术后天数": 180, "事件": "调整免疫抑制剂剂量"},
            {"术后天数": 365, "死亡原因": "感染"},
        ],
        "时序检验指标 (纯数值序列)": ["ALB: 21.0。"],
    }
    cleaned = sanitize_patient(patient)
    assert "真实标签" not in cleaned
    assert cleaned["临床事件"] == [
        "术后第30天：复查肝功能较前改善。",
        {"术后天数": 180, "事件": "调整免疫抑制剂剂量"},
    ]
    assert "是否死亡" not in cleaned["基础信息"]
    assert "死亡时间" not in cleaned["基础信息"]
    assert "死亡原因" not in cleaned["基础信息"]
    assert cleaned["基础信息"]["患儿性别"] == "男"
    assert cleaned["时序检验指标 (纯数值序列)"] == ["ALB: 21.0。"]
    assert patient["临床事件"][1] == "术后第90天：患儿死亡。"
    assert patient["基础信息"]["是否死亡"] == "True"


@pytest.mark.parametrize(
    "unsafe_event",
    [
        "患儿去世",
        "记录生存结局",
        "终点标签：阳性",
        "patient died after follow-up",
        {"event": "deceased"},
    ],
)
def test_sanitize_filters_outcome_leakage_from_clinical_events(unsafe_event):
    cleaned = sanitize_patient(
        {
            "id": "1",
            "临床事件": ["术后第7天：超声提示血流通畅。", unsafe_event],
        }
    )
    assert cleaned["临床事件"] == ["术后第7天：超声提示血流通畅。"]


def test_sanitize_drops_invalid_clinical_event_section():
    cleaned = sanitize_patient({"id": "1", "临床事件": "患儿死亡"})
    assert "临床事件" not in cleaned


def test_prompt_messages_structure():
    msgs = LLMPromptBuilder().build_messages({"id": "1", "基础信息": {}})
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "JSON" in msgs[0]["content"]


def test_sft_prompt_messages_structure_and_sanitization():
    patient = {
        "id": "1",
        "真实标签": {"是否死亡": 1},
        "基础信息": {"患儿性别": "男", "是否死亡": "True"},
        "临床事件": ["术后第7天：超声提示血流通畅。", "术后第30天：患儿病故。"],
    }
    msgs = SFTPatternPromptBuilder().build_messages(patient)
    assert msgs[0]["content"] == "You are a helpful assistant."
    assert "<Pattern>" in msgs[1]["content"]
    assert "<Answer>" in msgs[1]["content"]
    assert "真实标签" not in msgs[1]["content"]
    assert "是否死亡" not in msgs[1]["content"]
    assert "超声提示血流通畅" in msgs[1]["content"]
    assert "患儿病故" not in msgs[1]["content"]
    assert "临床事件1项" in msgs[1]["content"]


def test_generic_prompt_keeps_only_safe_clinical_events():
    msgs = LLMPromptBuilder().build_messages(
        {
            "id": "1",
            "临床事件": ["术后第30天：调整用药。", "死亡时间：术后第90天"],
        }
    )
    assert "术后第30天：调整用药。" in msgs[1]["content"]
    assert "死亡时间" not in msgs[1]["content"]


def test_parse_llm_json_variants():
    assert parse_llm_json('{"death_probability_1m": 0.1}')["death_probability_1m"] == 0.1
    fenced = '```json\n{"death_probability_1m": 0.2}\n```'
    assert parse_llm_json(fenced)["death_probability_1m"] == 0.2
    noisy = 'Here you go: {"death_probability_1y": 0.3} thanks'
    assert parse_llm_json(noisy)["death_probability_1y"] == 0.3


def test_parse_sft_pattern_response():
    raw = (
        "<think>ignore</think>\n"
        "<Pattern>ALB@异常 + ALT@改善</Pattern>"
        "<Analysis>白蛋白偏低但转氨酶改善。</Analysis>"
        '<Answer> {"1m": 0.12, "1y": 0.34, "5y": 0.56}</Answer>'
    )
    parsed = parse_sft_pattern_response(raw)
    assert parsed["parse_status"] == "ok"
    assert parsed["pred_probs"] == {"1m": 0.12, "1y": 0.34, "5y": 0.56}
    assert parsed["pair_count"] == 1
    assert parsed["patterns"][0]["pattern"] == "ALB@异常 + ALT@改善"


def test_parse_sft_pattern_response_preserves_null_probabilities():
    raw = (
        "<Pattern>ALB@异常</Pattern>"
        "<Analysis>白蛋白偏低提示营养风险。</Analysis>"
        '<Answer>{"1m": 0.0129, "1y": null, "5y": null}</Answer>'
    )
    parsed = parse_sft_pattern_response(raw)
    assert parsed["pred_probs"] == {"1m": 0.0129, "1y": None, "5y": None}
    assert parsed["probability_fields_valid"] == {"1m": True, "1y": True, "5y": True}
    assert parsed["parse_status"] == "ok"


def test_llm_sft_mode_returns_pattern_result():
    from app.config import LLMSettings

    svc = LLMService(
        LLMSettings(
            base_url="http://example.test/v1",
            model_name="served-model",
            prompt_mode="sft_pattern_prob",
            rationale_second_pass=False,
        )
    )
    svc.client.chat = lambda _messages: (
        "<Pattern>ALB@异常</Pattern>"
        "<Analysis>低白蛋白提示营养及肝功能风险。</Analysis>"
        '<Answer>{"1m": 0.1, "1y": 0.2, "5y": 0.3}</Answer>'
    )
    out = svc.predict_one({"id": "1", "基础信息": {"患儿性别": "男"}})
    assert out["status"] == "ok"
    assert out["death_probability_1m"] == 0.1
    assert out["patterns"][0]["analysis"] == "低白蛋白提示营养及肝功能风险。"
    assert out["parse_status"] == "ok"
    assert "native_probabilities" not in out
    assert "probability_sources" not in out
    assert "fallback_fields" not in out
    assert "answer_text" not in out
    assert "raw_response" not in out
    assert out["basic_info_pattern_filter"]["enabled"] is True
    assert out["basic_info_pattern_filter"]["removed_pattern_count"] == 0


def test_llm_sft_mode_filters_basic_info_before_parse():
    from app.config import LLMSettings

    svc = LLMService(
        LLMSettings(
            base_url="http://example.test/v1",
            model_name="served-model",
            prompt_mode="sft_pattern_prob",
            rationale_second_pass=False,
        )
    )
    svc.client.chat = lambda _messages: (
        "<Pattern>GRWR@正常 + ALT@改善</Pattern>"
        "<Analysis>GRWR处于正常范围。ALT逐步下降，提示肝细胞损伤改善。</Analysis>"
        '<Answer>{"1m": 0.1, "1y": 0.2, "5y": 0.3}</Answer>'
    )
    out = svc.predict_one({"id": "1", "基础信息": {"GRWR": "2.5"}})
    assert out["status"] == "ok"
    assert out["parse_status"] == "format_warning"
    assert out["death_probability_1m"] == 0.1
    assert out["patterns"] == [
        {
            "pattern": "ALT@改善",
            "analysis": "ALT逐步下降，提示肝细胞损伤改善。",
        }
    ]
    assert "raw_response" not in out
    assert "basic_info_pattern_filter removed 1 pattern item(s)" in out["parse_warnings"]
    assert out["basic_info_pattern_filter"]["enabled"] is True
    assert out["basic_info_pattern_filter"]["removed_pattern_count"] == 1
    assert out["basic_info_pattern_filter"]["removed_patterns"][0]["entity"] == "GRWR"


def test_llm_sft_mode_can_disable_basic_info_filter():
    from app.config import LLMSettings

    svc = LLMService(
        LLMSettings(
            base_url="http://example.test/v1",
            model_name="served-model",
            prompt_mode="sft_pattern_prob",
            pattern_guard_enabled=False,
            rationale_second_pass=False,
        )
    )
    svc.client.chat = lambda _messages: (
        "<Pattern>GRWR@正常 + ALT@改善</Pattern>"
        "<Analysis>GRWR处于正常范围。ALT逐步下降。</Analysis>"
        '<Answer>{"1m": 0.1, "1y": 0.2, "5y": 0.3}</Answer>'
    )
    out = svc.predict_one({"id": "1", "基础信息": {"GRWR": "2.5"}})
    assert out["status"] == "ok"
    assert out["parse_status"] == "ok"
    assert out["patterns"][0]["pattern"] == "GRWR@正常 + ALT@改善"
    assert out["basic_info_pattern_filter"]["enabled"] is False
    assert out["basic_info_pattern_filter"]["removed_pattern_count"] == 0


def test_llm_sft_null_probabilities_are_preserved_without_ml_completion():
    from app.config import LLMSettings

    svc = LLMService(
        LLMSettings(
            base_url="http://example.test/v1",
            model_name="served-model",
            prompt_mode="sft_pattern_prob",
            pattern_guard_enabled=False,
            rationale_second_pass=True,
            rationale_max_tokens=768,
        )
    )
    first_pass_calls = []
    rationale_calls = []

    def fake_chat(messages, **kwargs):
        if kwargs.get("response_format"):
            rationale_calls.append((messages, kwargs))
            return '{"rationale":"低白蛋白与胆红素恶化共同提示总体风险。"}'
        first_pass_calls.append((messages, kwargs))
        return (
            "<Pattern>ALB@异常 + TB@恶化</Pattern>"
            "<Analysis>低白蛋白与胆红素升高提示肝功能风险。</Analysis>"
            '<Answer>{"1m":0.0129,"1y":null,"5y":null}</Answer>'
        )

    svc.client.chat = fake_chat
    patient = {"id": "1", "基础信息": {"患儿性别": "男"}}
    out = svc.predict_one(patient)
    repeated = svc.predict_one(patient)
    batched = svc.predict_batch([patient])[0]

    assert out["status"] == "ok"
    assert out["death_probability_1m"] == 0.0129
    assert out["death_probability_1y"] is None
    assert out["death_probability_5y"] is None
    assert out["parse_status"] == "ok"
    assert out["parse_warnings"] == []
    assert repeated["death_probability_1y"] is None
    assert repeated["death_probability_5y"] is None
    assert batched["death_probability_1y"] is None
    assert batched["death_probability_5y"] is None
    for hidden_field in (
        "native_probabilities",
        "probability_sources",
        "fallback_fields",
        "native_parse_status",
        "answer_text",
        "raw_response",
    ):
        assert hidden_field not in out
    assert out["rationale"] == "低白蛋白与胆红素恶化共同提示总体风险。"
    assert out["rationale_source"] == "vllm_second_pass"
    assert len(first_pass_calls) == 3
    assert len(rationale_calls) == 3
    assert rationale_calls[0][1]["max_tokens"] == 768
    assert rationale_calls[0][1]["response_format"]["type"] == "json_schema"
    rationale_context = json.loads(rationale_calls[0][0][1]["content"])
    assert rationale_context["final_probabilities"] == {
        "1m": 0.0129,
        "1y": None,
        "5y": None,
    }


def test_llm_sft_missing_or_invalid_probabilities_are_partial():
    from app.config import LLMSettings

    svc = LLMService(
        LLMSettings(
            base_url="http://example.test/v1",
            model_name="served-model",
            prompt_mode="sft_pattern_prob",
            pattern_guard_enabled=False,
            rationale_second_pass=False,
        )
    )
    svc.client.chat = lambda _messages: (
        "<Pattern>ALB@异常</Pattern>"
        "<Analysis>低白蛋白提示营养风险。</Analysis>"
        '<Answer>{"1m":0.1,"5y":"invalid"}</Answer>'
    )
    out = svc.predict_one({"id": "1"})
    assert out["status"] == "partial"
    assert out["parse_status"] == "partial_probabilities"
    assert out["death_probability_1m"] == 0.1
    assert out["death_probability_1y"] is None
    assert out["death_probability_5y"] is None
    assert out["parse_warnings"] == []
    assert out["error"] is None


def test_llm_rationale_failure_uses_pattern_analysis_fallback():
    from app.config import LLMSettings

    svc = LLMService(
        LLMSettings(
            base_url="http://example.test/v1",
            model_name="served-model",
            prompt_mode="sft_pattern_prob",
            pattern_guard_enabled=False,
            rationale_second_pass=True,
        )
    )
    calls = 0

    def fake_chat(_messages, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return (
                "<Pattern>INR@异常</Pattern>"
                "<Analysis>INR 异常提示凝血功能风险。</Analysis>"
                '<Answer>{"1m":0.1,"1y":0.2,"5y":0.3}</Answer>'
            )
        raise LLMClientError("second pass unavailable")

    svc.client.chat = fake_chat
    out = svc.predict_one({"id": "1"})
    assert out["status"] == "ok"
    assert out["parse_status"] == "ok"
    assert out["rationale"] == "INR 异常提示凝血功能风险。"
    assert out["rationale_source"] == "pattern_analysis_fallback"
    assert "rationale second pass failed" in out["parse_warnings"][-1]
    assert out["error"] is None


def test_llm_generic_json_mode_works_with_vllm():
    from app.config import LLMSettings

    svc = LLMService(
        LLMSettings(
            base_url="http://example.test/v1",
            model_name="Qwen3-4B-sft",
            prompt_mode="generic_json",
        )
    )
    svc.client.chat = lambda _messages: (
        '{"death_probability_1m": 0.11, "death_probability_1y": 0.22, '
        '"death_probability_5y": 0.33, "rationale": "稳定"}'
    )
    out = svc.predict_one({"id": "1", "基础信息": {"患儿性别": "男"}})
    assert out["status"] == "ok"
    assert out["death_probability_1m"] == 0.11
    assert out["rationale"] == "稳定"
    assert out["patterns"] == []


def test_llm_generic_json_mode_preserves_explicit_null():
    from app.config import LLMSettings

    svc = LLMService(
        LLMSettings(
            base_url="http://example.test/v1",
            model_name="served-model",
            prompt_mode="generic_json",
        )
    )
    svc.client.chat = lambda _messages: (
        '{"death_probability_1m": 0.11, "death_probability_1y": null, '
        '"death_probability_5y": null}'
    )

    out = svc.predict_one({"id": "1"})

    assert out["status"] == "ok"
    assert out["parse_status"] == "ok"
    assert out["death_probability_1m"] == 0.11
    assert out["death_probability_1y"] is None
    assert out["death_probability_5y"] is None


def test_llm_generic_json_mode_rejects_missing_or_invalid_probability():
    from app.config import LLMSettings

    svc = LLMService(
        LLMSettings(
            base_url="http://example.test/v1",
            model_name="served-model",
            prompt_mode="generic_json",
        )
    )
    svc.client.chat = lambda _messages: (
        '{"death_probability_1m": 0.11, "death_probability_5y": "invalid"}'
    )

    out = svc.predict_one({"id": "1"})

    assert out["status"] == "partial"
    assert out["parse_status"] == "partial_probabilities"
    assert out["death_probability_1y"] is None
    assert out["death_probability_5y"] is None


@pytest.mark.parametrize("response", ["not json", "[]"])
def test_llm_generic_json_mode_rejects_unparseable_or_non_object_output(response):
    from app.config import LLMSettings

    svc = LLMService(
        LLMSettings(
            base_url="http://example.test/v1",
            model_name="served-model",
            prompt_mode="generic_json",
        )
    )
    svc.client.chat = lambda _messages: response

    out = svc.predict_one({"id": "1"})

    assert out["status"] == "error"
    assert out["parse_status"] == "json_parse_failed"
    assert out["death_probability_1m"] is None


def test_unsupported_prompt_mode_is_rejected():
    with pytest.raises(ValueError, match="sft_pattern_prob or generic_json"):
        build_prompt_builder("unsupported_mode")


def test_llm_disabled_returns_status():
    from app.config import LLMSettings

    svc = LLMService(LLMSettings(base_url=None, model_name=None))
    out = svc.predict_one({"id": "1", "基础信息": {}})
    assert out["status"] == "disabled"
    assert out["death_probability_1m"] is None
