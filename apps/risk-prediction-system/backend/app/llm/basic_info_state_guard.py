#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


PAIR_RE = re.compile(r"<Pattern>(.*?)</Pattern>\s*<Analysis>(.*?)</Analysis>", re.S)
PATTERN_RE = re.compile(r"<Pattern>(.*?)</Pattern>", re.S)
FOLLOWUP_TIME_RE = re.compile(r"术后[0-9零一二三四五六七八九十百]+[年月日]")
NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")
PLACEHOLDER_PAIR = (
    "<Pattern>无可保留动态指标</Pattern>\n"
    "<Analysis>过滤基础信息后未识别到符合要求的检验指标或用药动态模式。</Analysis>"
)

BASIC_INFO_KEYWORDS: Tuple[str, ...] = tuple(
    sorted(
        {
            "患儿术时体重",
            "术时体重",
            "患儿出生日期",
            "出生日期",
            "手术日期",
            "手术年龄",
            "供肝重量",
            "供体年龄",
            "供体性别",
            "供体血型",
            "患儿性别",
            "患儿血型",
            "患儿年龄",
            "主要诊断分类",
            "ABO血型相容性",
            "ABO血型",
            "ABO不相容",
            "ABO",
            "GRWR",
            "CYP",
            "供体CYP",
            "患儿CYP",
            "受体CYP",
            "受体年龄",
            "诊断",
            "诊断1",
            "诊断2",
            "诊断3",
            "胆道闭锁",
            "葛西",
            "肝移植术后",
            "活体肝移植",
            "移植方式",
            "移植类型",
            "术式",
            "全肝",
            "劈离",
            "供肝活检",
            "供肝病理",
            "血型",
            "相容",
            "相同",
            "性别",
            "年龄",
            "身高",
            "体重",
        },
        key=len,
        reverse=True,
    )
)

DIAGNOSIS_ENTITY_KEYWORDS = ("诊断", "胆道闭锁", "葛西", "肝移植术后", "主要诊断分类")
NORMAL_STATE_KEYWORDS = ("正常", "阴性", "相同", "相容", "匹配")
EVENT_STATE_KEYWORDS = ("事件", "病理", "并发症", "存在", "阳性")
BAD_BASIC_STATE_TERMS = (
    "剧烈变化",
    "剧烈波动",
    "显著变化",
    "显著波动",
    "明显变化",
    "明显波动",
    "风险趋势异常",
    "突发异常",
    "严重异常",
    "不稳定",
    "震荡",
    "波动",
    "恶化",
    "改善",
    "好转",
    "恢复",
    "加重",
    "异常",
    "升高",
    "降低",
    "下降",
    "上升",
    "增长",
    "递增",
    "递减",
    "偏高",
    "偏低",
)

ANALYSIS_BAD_TERMS = (
    "剧烈变化",
    "剧烈波动",
    "显著变化",
    "显著波动",
    "明显变化",
    "明显波动",
    "突发异常",
    "风险趋势异常",
    "恶化",
    "改善",
    "好转",
    "波动",
)

BASIC_INFO_EXCLUSION_KEYWORDS = ("肝穿", "肝穿刺", "超声", "弹性成像", "影像", "随访")
BASIC_INFO_EXCLUSION_ALLOWLIST = ("供肝活检", "供肝病理")
BODY_MEASUREMENT_KEYWORDS = ("体重", "身高")
BODY_DYNAMIC_STATE_TERMS = (
    "剧烈变化",
    "剧烈波动",
    "显著变化",
    "显著波动",
    "明显变化",
    "明显波动",
    "突发异常",
    "波动",
    "变化",
    "异常",
    "升高",
    "降低",
    "下降",
    "上升",
    "增长",
    "递增",
    "递减",
    "改善",
    "好转",
    "恶化",
    "加重",
    "偏高",
    "偏低",
)
BODY_DYNAMIC_ANALYSIS_TERMS = (
    "术后",
    "随访",
    "时间点",
    "从",
    "由",
    "至",
    "到",
    "→",
    "->",
    "升至",
    "降至",
    "逐渐",
    "持续",
    "多次",
    "连续",
    "早期",
    "中期",
    "后期",
    "趋势",
    "波动",
    "变化",
)
ANALYSIS_BASIC_INFO_TERMS = ("基础信息", "基础资料", "体格信息", "基线信息")
DYNAMIC_ENTITY_ALLOWLIST_KEYWORDS = (
    "ALT",
    "AST",
    "TB",
    "DB",
    "γ-GT",
    "GGT",
    "ALP",
    "ALB",
    "TP",
    "CHE",
    "PT",
    "INR",
    "APTT",
    "FIB",
    "CR",
    "WBC",
    "PLT",
    "HB",
    "CRP",
    "PCT",
    "N(%)",
    "胆汁酸",
    "胆汁淤积",
    "肝功能",
    "凝血",
    "肌酐",
    "尿素",
    "尿酸",
    "血糖",
    "电解质",
    "白细胞",
    "血小板",
    "血红蛋白",
    "中性粒",
    "CMV",
    "EBV",
    "HBV",
    "DNA",
    "HBsAg",
    "HBsAb",
    "HBeAg",
    "HBeAb",
    "HBcAb",
    "免疫抑制",
    "他克莫司",
    "环孢素",
    "吗替麦考酚",
    "泼尼松",
    "甲泼尼龙",
    "激素",
    "浓度",
    "剂量",
    "停药",
    "加药",
    "调整",
    "抗感染",
    "抗病毒",
)


@dataclass(frozen=True)
class PatternCorrection:
    entity: str
    original_state: str
    corrected_state: str
    original_item: str
    corrected_item: str
    reason: str


@dataclass(frozen=True)
class PatternRemoval:
    entity: str
    original_state: str
    original_item: str
    reason: str


@dataclass(frozen=True)
class CorrectionResult:
    text: str
    correction_count: int
    corrections: Tuple[PatternCorrection, ...]
    analysis_rewrite_count: int = 0
    removed_pattern_count: int = 0
    removed_patterns: Tuple[PatternRemoval, ...] = tuple()
    placeholder_inserted: bool = False

    def to_dict(self) -> dict:
        return {
            "correction_count": self.correction_count,
            "analysis_rewrite_count": self.analysis_rewrite_count,
            "corrections": [asdict(item) for item in self.corrections],
            "removed_pattern_count": self.removed_pattern_count,
            "removed_patterns": [asdict(item) for item in self.removed_patterns],
            "placeholder_inserted": self.placeholder_inserted,
        }


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def is_basic_info_entity(entity: str) -> bool:
    if contains_any(entity, DYNAMIC_ENTITY_ALLOWLIST_KEYWORDS):
        return False
    if FOLLOWUP_TIME_RE.search(entity):
        return False
    if contains_any(entity, BASIC_INFO_EXCLUSION_KEYWORDS) and not contains_any(
        entity, BASIC_INFO_EXCLUSION_ALLOWLIST
    ):
        return False
    return contains_any(entity, BASIC_INFO_KEYWORDS)


def is_bad_basic_state(state: str) -> bool:
    return contains_any(state, BAD_BASIC_STATE_TERMS)


def corrected_basic_state(entity: str, state: str) -> str:
    if contains_any(entity, DIAGNOSIS_ENTITY_KEYWORDS) or "诊断" in state:
        return "诊断"
    if contains_any(state, NORMAL_STATE_KEYWORDS):
        return "正常/阴性"
    if contains_any(state, EVENT_STATE_KEYWORDS):
        return "事件存在"
    return "基础状态"


def is_body_measurement_entity(entity: str) -> bool:
    return contains_any(entity, BODY_MEASUREMENT_KEYWORDS)


def has_body_dynamic_evidence(state: str, analysis_text: str) -> bool:
    if not contains_any(state, BODY_DYNAMIC_STATE_TERMS):
        return False
    if not analysis_text:
        return False
    if not contains_any(analysis_text, BODY_DYNAMIC_ANALYSIS_TERMS):
        return False

    numeric_count = len(NUMERIC_RE.findall(analysis_text))
    if numeric_count >= 2:
        return True
    return contains_any(analysis_text, ("波动", "变化", "上升", "下降", "升高", "降低")) and contains_any(
        analysis_text, ("术后", "随访", "时间点", "多次", "连续")
    )


def parse_pattern_item(item: str) -> Tuple[str, str] | None:
    text = item.strip()
    if not text or text.count("@") != 1:
        return None
    if "<" in text or ">" in text:
        return None
    entity, state = text.split("@", 1)
    entity = entity.strip()
    state = state.strip()
    if not entity or not state:
        return None
    return entity, state


def removable_unparsed_basic_item(item: str, analysis_text: str) -> PatternRemoval | None:
    text = item.strip()
    if not text or "<" in text or ">" in text:
        return None
    if not is_basic_info_entity(text):
        return None
    if is_body_measurement_entity(text) and has_body_dynamic_evidence("波动", f"{text}。{analysis_text}"):
        return None
    return PatternRemoval(
        entity=text,
        original_state="",
        original_item=text,
        reason="basic_info_removed_from_pattern",
    )


def removal_for_pattern_item(entity: str, state: str, original_item: str, analysis_text: str) -> PatternRemoval | None:
    if not is_basic_info_entity(entity):
        return None
    if is_body_measurement_entity(entity) and has_body_dynamic_evidence(state, analysis_text):
        return None

    reason = "baseline_body_measurement_removed_from_pattern" if is_body_measurement_entity(entity) else (
        "basic_info_removed_from_pattern"
    )
    return PatternRemoval(
        entity=entity,
        original_state=state,
        original_item=original_item,
        reason=reason,
    )


def filter_pattern_text(pattern_text: str, analysis_text: str = "") -> Tuple[str, Tuple[PatternRemoval, ...]]:
    parts = [part.strip() for part in pattern_text.split("+")]
    kept_parts: List[str] = []
    removals: List[PatternRemoval] = []

    for raw_part in parts:
        parsed = parse_pattern_item(raw_part)
        if parsed is None:
            removal = removable_unparsed_basic_item(raw_part, analysis_text)
            if removal is None:
                kept_parts.append(raw_part)
            else:
                removals.append(removal)
            continue

        entity, state = parsed
        removal = removal_for_pattern_item(entity, state, raw_part, analysis_text)
        if removal is None:
            kept_parts.append(raw_part)
        else:
            removals.append(removal)

    return " + ".join(part for part in kept_parts if part), tuple(removals)


def correct_pattern_text(pattern_text: str) -> Tuple[str, Tuple[PatternCorrection, ...]]:
    filtered_pattern, removals = filter_pattern_text(pattern_text)
    corrections = [
        PatternCorrection(
            entity=item.entity,
            original_state=item.original_state,
            corrected_state="",
            original_item=item.original_item,
            corrected_item="",
            reason=item.reason,
        )
        for item in removals
    ]
    return filtered_pattern, tuple(corrections)


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?])", text)
    return [part for part in parts if part]


def correction_entity_label(entities: Sequence[str]) -> str:
    unique = []
    for entity in entities:
        if entity not in unique:
            unique.append(entity)
    if not unique:
        return "基础信息"
    return "、".join(unique[:4])


def clean_analysis_text(
    analysis_text: str,
    removals: Sequence[PatternRemoval],
    kept_pattern_text: str,
) -> Tuple[str, int]:
    if not removals:
        return analysis_text, 0

    removed_entities = [item.entity for item in removals if item.entity]
    kept_entities = []
    for part in kept_pattern_text.split("+"):
        parsed = parse_pattern_item(part)
        if parsed:
            kept_entities.append(parsed[0])

    rewrite_count = 0
    out: List[str] = []
    for sentence in split_sentences(analysis_text):
        mentions_removed = any(entity and entity in sentence for entity in removed_entities)
        mentions_basic_info = contains_any(sentence, ANALYSIS_BASIC_INFO_TERMS)
        if mentions_removed or (mentions_basic_info and removed_entities):
            rewrite_count += 1
            continue
        out.append(sentence)

    cleaned = "".join(out).strip()
    if cleaned:
        return cleaned, rewrite_count

    kept_label = correction_entity_label(kept_entities)
    fallback = f"保留的检验/用药动态指标为{kept_label}；基础信息已从Pattern与分析中移除。"
    return fallback, rewrite_count


def correct_analysis_text(analysis_text: str, corrections: Sequence[PatternCorrection]) -> Tuple[str, int]:
    if not corrections:
        return analysis_text, 0

    entities = [item.entity for item in corrections]
    label = correction_entity_label(entities)
    replacement = (
        f"{label}属于基础/体格或移植相关信息，仅作为基线状态记录，"
        "不按检验指标式剧烈变化解释。"
    )
    rewrite_count = 0
    out: List[str] = []

    for sentence in split_sentences(analysis_text):
        mentions_entity = any(entity and entity in sentence for entity in entities) or contains_any(
            sentence, ("基础信息", "基础资料", "体格信息")
        )
        if mentions_entity and contains_any(sentence, ANALYSIS_BAD_TERMS):
            out.append(replacement)
            rewrite_count += 1
        else:
            out.append(sentence)

    if not out:
        return analysis_text, 0
    return "".join(out), rewrite_count


def _correct_pairs(text: str) -> Tuple[str, Tuple[PatternCorrection, ...], int]:
    all_removals: List[PatternRemoval] = []
    analysis_rewrite_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal analysis_rewrite_count
        pattern_text = match.group(1)
        analysis_text = match.group(2)
        filtered_pattern, removals = filter_pattern_text(pattern_text, analysis_text)
        if not removals:
            return match.group(0)

        all_removals.extend(removals)
        if not filtered_pattern:
            analysis_rewrite_count += 1
            return ""

        analysis_text, rewrites = clean_analysis_text(analysis_text, removals, filtered_pattern)
        analysis_rewrite_count += rewrites
        return f"<Pattern>{filtered_pattern}</Pattern>\n<Analysis>{analysis_text}</Analysis>"

    return PAIR_RE.sub(replace, text), tuple(
        PatternCorrection(
            entity=item.entity,
            original_state=item.original_state,
            corrected_state="",
            original_item=item.original_item,
            corrected_item="",
            reason=item.reason,
        )
        for item in all_removals
    ), analysis_rewrite_count


def _filter_pairs(text: str) -> Tuple[str, Tuple[PatternRemoval, ...], int]:
    all_removals: List[PatternRemoval] = []
    analysis_rewrite_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal analysis_rewrite_count
        pattern_text = match.group(1)
        analysis_text = match.group(2)
        filtered_pattern, removals = filter_pattern_text(pattern_text, analysis_text)
        if not removals:
            return match.group(0)

        all_removals.extend(removals)
        if not filtered_pattern:
            analysis_rewrite_count += 1
            return ""

        analysis_text, rewrites = clean_analysis_text(analysis_text, removals, filtered_pattern)
        analysis_rewrite_count += rewrites
        return f"<Pattern>{filtered_pattern}</Pattern>\n<Analysis>{analysis_text}</Analysis>"

    return PAIR_RE.sub(replace, text), tuple(all_removals), analysis_rewrite_count


def _correct_remaining_patterns(text: str) -> Tuple[str, Tuple[PatternCorrection, ...]]:
    filtered, removals = _filter_remaining_patterns(text)
    corrections = [
        PatternCorrection(
            entity=item.entity,
            original_state=item.original_state,
            corrected_state="",
            original_item=item.original_item,
            corrected_item="",
            reason=item.reason,
        )
        for item in removals
    ]
    return filtered, tuple(corrections)


def _filter_remaining_patterns(text: str) -> Tuple[str, Tuple[PatternRemoval, ...]]:
    all_removals: List[PatternRemoval] = []

    def replace(match: re.Match[str]) -> str:
        if re.match(r"\s*<Analysis>", text[match.end() :]):
            return match.group(0)
        pattern_text = match.group(1)
        filtered_pattern, removals = filter_pattern_text(pattern_text)
        if not removals:
            return match.group(0)

        all_removals.extend(removals)
        if not filtered_pattern:
            return ""
        return f"<Pattern>{filtered_pattern}</Pattern>"

    return PATTERN_RE.sub(replace, text), tuple(all_removals)


def normalize_output_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def add_placeholder_if_needed(text: str, removed_count: int) -> Tuple[str, bool]:
    normalized = normalize_output_text(text)
    if removed_count <= 0 or PAIR_RE.findall(normalized):
        return normalized if removed_count > 0 else text, False
    if PLACEHOLDER_PAIR in normalized:
        return normalized, False
    if normalized:
        return f"{normalized}\n\n{PLACEHOLDER_PAIR}", True
    return PLACEHOLDER_PAIR, True


def correct_model_output(text: str) -> CorrectionResult:
    if not text:
        return CorrectionResult(text=text or "", correction_count=0, corrections=tuple())

    corrected = text
    all_removals: List[PatternRemoval] = []
    analysis_rewrite_count = 0
    for _ in range(5):
        corrected, pair_removals, rewrites = _filter_pairs(corrected)
        corrected, remaining_removals = _filter_remaining_patterns(corrected)
        removals_this_pass = pair_removals + remaining_removals
        if not removals_this_pass:
            break
        all_removals.extend(removals_this_pass)
        analysis_rewrite_count += rewrites

    removals = tuple(all_removals)
    corrected, placeholder_inserted = add_placeholder_if_needed(corrected, len(removals))
    corrections = tuple(
        PatternCorrection(
            entity=item.entity,
            original_state=item.original_state,
            corrected_state="",
            original_item=item.original_item,
            corrected_item="",
            reason=item.reason,
        )
        for item in removals
    )
    return CorrectionResult(
        text=corrected,
        correction_count=len(removals),
        corrections=corrections,
        analysis_rewrite_count=analysis_rewrite_count,
        removed_pattern_count=len(removals),
        removed_patterns=removals,
        placeholder_inserted=placeholder_inserted,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Correct inappropriate basic-info trend states in model output text.")
    parser.add_argument("text", nargs="?", default="", help="Text to correct. If omitted, stdin is used.")
    parser.add_argument("--json", action="store_true", help="Print correction metadata as JSON.")
    args = parser.parse_args()

    text = args.text or sys.stdin.read()
    result = correct_model_output(text)
    if args.json:
        print(json.dumps({"text": result.text, **result.to_dict()}, ensure_ascii=False, indent=2))
    else:
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
