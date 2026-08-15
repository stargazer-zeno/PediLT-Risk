from __future__ import annotations

import copy
import json
from typing import Any

# Fields that directly leak the mortality outcome must never reach the LLM.
LEAKAGE_TOP_LEVEL = {"真实标签"}
LEAKAGE_BASE_FIELDS = {"是否死亡", "死亡时间", "死亡原因"}
CLINICAL_EVENTS_FIELD = "临床事件"
EVENT_LEAKAGE_TERMS = (
    "死亡",
    "去世",
    "病故",
    "生存结局",
    "存活结局",
    "结局标签",
    "终点标签",
    "mortality",
    "deceased",
    "died",
    "death",
)

SYSTEM_PROMPT = (
    "你是一名资深的小儿肝移植随访临床专家。你的任务是根据患儿的术前基础信息、"
    "术后随访的时序检验指标、时序用药记录与历史临床事件，评估该患儿在当前随访时间点之后"
    "未来 1 个月、1 年、5 年内的死亡风险概率。\n"
    "请只依据所提供的客观临床数据进行推断，不要臆造未给出的信息。\n"
    "输出必须是严格的 JSON 对象，不要包含任何额外文字、解释或 Markdown 代码块标记，"
    "格式如下：\n"
    '{"death_probability_1m": <0-1之间的小数>, '
    '"death_probability_1y": <0-1之间的小数>, '
    '"death_probability_5y": <0-1之间的小数>, '
    '"rationale": "<简要中文临床推理，不超过100字>"}'
)

SFT_SYSTEM_PROMPT = "You are a helpful assistant."

SFT_USER_PREFIX_TEMPLATE = """你是一位有丰富经验的小儿肝移植医生。

### 【任务说明】
请基于以下患者截至最后一次随访时点的全部历史资料，先总结该患者截至最后一次随访时点的关键临床模式，再对其后续1个月、1年及5年内发生死亡事件的风险进行前瞻性预测，给出对应的死亡风险预测概率值。
--------------------------------------------------
{SUMMARY_TEXT}
【患者病历数据】"""

SFT_OUTPUT_INSTRUCTIONS = """### 【输出指令】
请只输出以下标签拼接格式，不要输出任何格式之外的说明文字。

输出要求如下：
1. 先按临床模式逐组输出，每组包含：
   - `<Pattern>特征a@趋势x + 特征b@趋势y</Pattern>`
   - `<Analysis>对该模式的医学分析</Analysis>`
2. `pattern` 中允许使用的趋势仅限：`恶化`、`异常`、`波动`、`突发异常`、`改善`、`正常`
3. 所有临床模式输出完成后，追加死亡风险预测概率，格式为：
   `<Answer> {"1m": <未来 1 个月内发生死亡的概率>, "1y": <未来 1 年内发生死亡的概率>, "5y": <未来 5 年内发生死亡的概率>}</Answer>`
4. 概率取值必须是 0.0-1.0 之间的浮点数；数值 > 0.5 表示预测该时间窗口内死亡风险较高，数值 < 0.5 表示预测该时间窗口内存活概率更高

输出格式示例：
<Pattern>特征a@趋势x + 特征b@趋势y</Pattern> <Analysis>对该模式的医学分析</Analysis><Pattern>特征c@趋势z</Pattern> <Analysis>对该模式的医学分析</Analysis> <Answer> {"1m": 0.1234, "1y": 0.2345, "5y": 0.3456}</Answer>"""

def _event_contains_leakage(event: Any) -> bool:
    if isinstance(event, str):
        text = event
    else:
        try:
            text = json.dumps(event, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            text = str(event)
    normalized = "".join(text.lower().split())
    return any(term.lower() in normalized for term in EVENT_LEAKAGE_TERMS)


def _sanitize_clinical_events(events: Any) -> list[Any] | None:
    """Keep useful historical events while conservatively dropping outcome leaks."""
    if not isinstance(events, list):
        return None

    safe_events: list[Any] = []
    for event in events:
        if isinstance(event, str):
            if not event.strip() or _event_contains_leakage(event):
                continue
            safe_events.append(event)
        elif isinstance(event, dict):
            if not event or _event_contains_leakage(event):
                continue
            safe_events.append(event)
    return safe_events


def sanitize_patient(patient: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy with mortality-leaking fields and events removed."""
    cleaned = copy.deepcopy(patient)
    for key in LEAKAGE_TOP_LEVEL:
        cleaned.pop(key, None)
    base = cleaned.get("基础信息")
    if isinstance(base, dict):
        for field in LEAKAGE_BASE_FIELDS:
            base.pop(field, None)

    if CLINICAL_EVENTS_FIELD in cleaned:
        safe_events = _sanitize_clinical_events(cleaned[CLINICAL_EVENTS_FIELD])
        if safe_events is None:
            cleaned.pop(CLINICAL_EVENTS_FIELD, None)
        else:
            cleaned[CLINICAL_EVENTS_FIELD] = safe_events
    return cleaned


def _section_count(patient: dict[str, Any], section: str) -> int:
    value = patient.get(section)
    return len(value) if isinstance(value, list) else 0


def build_summary_text(patient: dict[str, Any]) -> str:
    """Create a deterministic brief summary for the SFT evaluation prompt."""
    cleaned = sanitize_patient(patient)
    base = cleaned.get("基础信息") or {}
    if not isinstance(base, dict):
        base = {}

    preferred_fields = [
        "患儿性别",
        "手术年龄",
        "患儿术时体重",
        "术式",
        "ABO血型相容性",
        "主要诊断分类",
        "诊断1",
        "诊断2",
        "诊断3",
        "供体性别",
        "供体年龄",
        "GRWR",
    ]
    base_parts = [
        f"{field}: {base[field]}"
        for field in preferred_fields
        if field in base and str(base[field]).strip()
    ]
    if not base_parts:
        base_parts = [
            f"{key}: {value}"
            for key, value in list(base.items())[:12]
            if str(value).strip()
        ]

    counts = (
        f"随访基础信息{_section_count(cleaned, '时序随访基础信息')}项，"
        f"检验指标{_section_count(cleaned, '时序检验指标 (纯数值序列)')}项，"
        f"用药记录{_section_count(cleaned, '时序用药记录 (纯数值序列)')}项，"
        f"临床事件{_section_count(cleaned, CLINICAL_EVENTS_FIELD)}项。"
    )
    return (
        f"患者ID: {cleaned.get('id', 'unknown')}\n"
        f"基础信息摘要: {'；'.join(base_parts) if base_parts else '未提供'}\n"
        f"时序资料摘要: {counts}"
    )


class LLMPromptBuilder:
    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_user_content(self, patient: dict[str, Any]) -> str:
        cleaned = sanitize_patient(patient)
        body = json.dumps(cleaned, ensure_ascii=False, indent=2)
        return (
            "以下是一名患儿截至当前随访时间点的临床数据（已移除任何结局标签）：\n\n"
            f"{body}\n\n"
            "请评估该患儿在当前随访点之后未来 1 个月、1 年、5 年内的死亡风险概率，"
            "并严格按系统指令的 JSON 格式输出。"
        )

    def build_messages(self, patient: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_user_content(patient)},
        ]


class SFTPatternPromptBuilder:
    """Build prompts for the SFT model used by basic_test_sft_pattern_prob.py."""

    def __init__(self, system_prompt: str = SFT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_user_content(self, patient: dict[str, Any]) -> str:
        cleaned = sanitize_patient(patient)
        summary_text = build_summary_text(cleaned)
        clean_ehr = json.dumps(cleaned, ensure_ascii=False, indent=2)
        prefix = SFT_USER_PREFIX_TEMPLATE.replace("{SUMMARY_TEXT}", summary_text)
        return f"{prefix}\n{clean_ehr}\n\n{SFT_OUTPUT_INSTRUCTIONS}"

    def build_messages(self, patient: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_user_content(patient)},
        ]


def build_prompt_builder(prompt_mode: str):
    if prompt_mode == "sft_pattern_prob":
        return SFTPatternPromptBuilder()
    if prompt_mode in {"", "generic_json"}:
        return LLMPromptBuilder()
    raise ValueError(
        f"Unsupported LLM_PROMPT_MODE={prompt_mode!r}; use sft_pattern_prob or generic_json."
    )
