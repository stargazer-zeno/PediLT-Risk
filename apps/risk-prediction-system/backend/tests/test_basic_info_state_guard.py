import re

from app.llm.basic_info_state_guard import PLACEHOLDER_PAIR, correct_model_output


PAIR_RE = re.compile(r"<Pattern>(.*?)</Pattern>\s*<Analysis>(.*?)</Analysis>", re.S)


def test_basic_info_is_removed_from_mixed_pattern():
    text = (
        "<Pattern>GRWR@正常 + ALT@改善</Pattern>\n"
        "<Analysis>GRWR处于正常范围。ALT逐步下降，提示肝细胞损伤改善。</Analysis>"
    )
    result = correct_model_output(text)
    assert "<Pattern>ALT@改善</Pattern>" in result.text
    assert "GRWR" not in result.text
    assert result.removed_pattern_count == 1
    assert not result.placeholder_inserted


def test_pure_basic_info_pair_becomes_placeholder():
    text = "<Pattern>诊断1@胆道闭锁</Pattern>\n<Analysis>诊断1为胆道闭锁。</Analysis>"
    result = correct_model_output(text)
    assert result.text == PLACEHOLDER_PAIR
    assert result.removed_pattern_count == 1
    assert result.placeholder_inserted


def test_baseline_weight_without_dynamic_evidence_is_removed():
    text = "<Pattern>患儿术时体重@正常</Pattern>\n<Analysis>患儿术时体重为8.1kg。</Analysis>"
    result = correct_model_output(text)
    assert result.text == PLACEHOLDER_PAIR
    assert result.removed_patterns[0].reason == "baseline_body_measurement_removed_from_pattern"


def test_dynamic_weight_with_time_series_is_kept():
    text = (
        "<Pattern>体重@波动</Pattern>\n"
        "<Analysis>术后随访体重从8.0kg升至9.2kg后降至8.5kg，呈现波动。</Analysis>"
    )
    result = correct_model_output(text)
    assert result.text == text
    assert result.removed_pattern_count == 0


def test_lab_trend_states_are_unchanged():
    text = (
        "<Pattern>CR@突发异常 + TB@恶化 + ALB@改善</Pattern>\n"
        "<Analysis>CR突发异常，TB恶化，ALB改善。</Analysis>"
    )
    result = correct_model_output(text)
    assert result.text == text
    assert result.removed_pattern_count == 0


def test_medication_patterns_are_unchanged():
    text = (
        "<Pattern>免疫抑制剂浓度@波动 + 他克莫司@调整</Pattern>\n"
        "<Analysis>免疫抑制剂浓度多次上下变化，他克莫司剂量随之调整。</Analysis>"
    )
    result = correct_model_output(text)
    assert result.text == text
    assert result.removed_pattern_count == 0


def test_answer_survives_filtering():
    text = (
        "<Pattern>GRWR@正常 + ALT@改善</Pattern>\n"
        "<Analysis>GRWR处于正常范围。ALT逐步下降。</Analysis>"
        '<Answer>{"1m": 0.1, "1y": 0.2, "5y": 0.3}</Answer>'
    )
    result = correct_model_output(text)
    assert "<Pattern>ALT@改善</Pattern>" in result.text
    assert '<Answer>{"1m": 0.1, "1y": 0.2, "5y": 0.3}</Answer>' in result.text


def test_pair_count_can_decrease_when_basic_pair_removed():
    text = (
        "<Pattern>GRWR@正常</Pattern>\n"
        "<Analysis>GRWR处于正常范围。</Analysis>\n\n"
        "<Pattern>ALT@改善</Pattern>\n"
        "<Analysis>ALT改善。</Analysis>"
    )
    result = correct_model_output(text)
    assert len(PAIR_RE.findall(text)) == 2
    assert len(PAIR_RE.findall(result.text)) == 1
    assert "<Pattern>ALT@改善</Pattern>" in result.text
