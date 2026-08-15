from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from ..config import LLMSettings, get_settings
from .basic_info_state_guard import correct_model_output
from .llm_client import LLMClient, LLMClientError
from .prompt_builder import build_prompt_builder

LOGGER = logging.getLogger(__name__)

OUTPUT_FIELDS = (
    "death_probability_1m",
    "death_probability_1y",
    "death_probability_5y",
)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
PAIR_PATTERN = re.compile(
    r"<Pattern>(.*?)</Pattern>\s*<Analysis>(.*?)</Analysis>",
    flags=re.DOTALL | re.IGNORECASE,
)
ANSWER_PATTERN = re.compile(
    r"<Answer>(.*?)</Answer>",
    flags=re.DOTALL | re.IGNORECASE,
)
LABEL_KEYS = ("1m", "1y", "5y")
VALID_TRENDS = ("恶化", "异常", "波动", "突发异常", "改善", "正常")


def _clamp_prob(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return max(0.0, min(1.0, num))


def parse_llm_json(content: str) -> dict[str, Any]:
    """Extract the JSON object from a (possibly noisy) LLM response."""
    text = content.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise
        return json.loads(match.group())


def strip_thinking_and_code_fences(raw_text: str) -> str:
    text = raw_text.split("</think>")[-1].strip() if "</think>" in raw_text else raw_text
    text = re.sub(r"```[a-zA-Z]*", "", text)
    return text.replace("```", "").strip()


def _normalize_probability_value(value: Any, *, present: bool) -> tuple[float | None, bool]:
    """Return a normalized probability and whether the source value is valid.

    An explicit JSON null is valid and must remain None. A missing field or an
    unparseable/non-finite value is invalid, even though both also normalize to
    None at the Python type level.
    """
    if not present:
        return None, False
    if value is None:
        return None, True
    probability = _clamp_prob(value)
    return probability, probability is not None


def normalize_json_text(text: str) -> str:
    return (
        text.replace("，", ",")
        .replace("：", ":")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def parse_answer_probs(
    text: str,
) -> tuple[dict[str, float | None], dict[str, bool], str, bool]:
    """Parse SFT <Answer> probabilities with regex fallback."""
    probs: dict[str, float | None] = {key: None for key in LABEL_KEYS}
    valid_fields: dict[str, bool] = {key: False for key in LABEL_KEYS}
    answer_match = ANSWER_PATTERN.search(text)
    answer_text = answer_match.group(1).strip() if answer_match else ""
    search_text = answer_text or text

    start = search_text.find("{")
    end = search_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = normalize_json_text(search_text[start : end + 1])
        try:
            payload = json.loads(candidate, strict=False)
            if isinstance(payload, dict):
                for key in LABEL_KEYS:
                    probs[key], valid_fields[key] = _normalize_probability_value(
                        payload.get(key),
                        present=key in payload,
                    )
        except json.JSONDecodeError:
            pass

    for key in LABEL_KEYS:
        if valid_fields[key]:
            continue
        match = re.search(
            rf'["\']?{re.escape(key)}["\']?\s*[:：]\s*(null|[01](?:\.\d+)?)(?=\s*[,，}}\n\r]|$)',
            search_text,
            flags=re.IGNORECASE,
        )
        if match:
            token = match.group(1)
            if token.lower() == "null":
                probs[key] = None
                valid_fields[key] = True
            else:
                probs[key], valid_fields[key] = _normalize_probability_value(
                    token,
                    present=True,
                )

    return probs, valid_fields, answer_text, answer_match is not None


def parse_pattern_pairs(text: str) -> tuple[list[dict[str, str]], str, bool]:
    pairs = [
        {"pattern": pattern.strip(), "analysis": analysis.strip()}
        for pattern, analysis in PAIR_PATTERN.findall(text)
        if pattern.strip() and analysis.strip()
    ]

    normalized = "\n\n".join(
        f"<Pattern>{pair['pattern']}</Pattern>\n<Analysis>{pair['analysis']}</Analysis>"
        for pair in pairs
    )

    tags_balanced = (
        text.count("<Pattern>") == text.count("</Pattern>")
        and text.count("<Analysis>") == text.count("</Analysis>")
        and text.count("<Pattern>") == text.count("<Analysis>") == len(pairs)
    )
    return pairs, normalized, tags_balanced


def validate_pattern_trends(pattern_pairs: list[dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    for idx, pair in enumerate(pattern_pairs, start=1):
        trend_tokens = re.findall(r"@([^+\s<]+)", pair["pattern"])
        invalid_trends = [trend for trend in trend_tokens if trend not in VALID_TRENDS]
        if invalid_trends:
            warnings.append(
                f"pattern_{idx} contains invalid trends: {', '.join(invalid_trends)}"
            )
    return warnings


def normalize_json_patterns(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    patterns: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            pattern = item.get("pattern") or item.get("Pattern") or ""
            analysis = item.get("analysis") or item.get("Analysis") or ""
        elif isinstance(item, str):
            pattern = item
            analysis = ""
        else:
            continue
        pattern = str(pattern).strip()
        analysis = str(analysis).strip()
        if pattern or analysis:
            patterns.append({"pattern": pattern, "analysis": analysis})
    return patterns


def parse_sft_pattern_response(raw_text: str | None) -> dict[str, Any]:
    if not isinstance(raw_text, str):
        return {
            "pred_probs": {key: None for key in LABEL_KEYS},
            "probability_fields_valid": {key: False for key in LABEL_KEYS},
            "answer_text": "",
            "has_answer_tag": False,
            "patterns": [],
            "pattern_output": "",
            "pair_count": 0,
            "parse_status": "failed",
            "parse_warnings": ["empty or non-string response"],
        }

    text = strip_thinking_and_code_fences(raw_text)
    probs, valid_fields, answer_text, has_answer_tag = parse_answer_probs(text)
    patterns, pattern_output, tags_balanced = parse_pattern_pairs(text)
    trend_warnings = validate_pattern_trends(patterns)

    all_probs_valid = all(valid_fields.values())
    has_pattern = len(patterns) > 0

    parse_warnings: list[str] = []
    if not has_answer_tag:
        parse_warnings.append("missing <Answer> tag")
    if not tags_balanced:
        parse_warnings.append("unbalanced or partially matched Pattern/Analysis tags")
    parse_warnings.extend(trend_warnings)

    if all_probs_valid and has_pattern and has_answer_tag and tags_balanced and not trend_warnings:
        parse_status = "ok"
    elif all_probs_valid and has_pattern:
        parse_status = "format_warning"
    elif not all_probs_valid and has_pattern:
        parse_status = "answer_parse_failed"
    elif all_probs_valid and not has_pattern:
        parse_status = "pattern_parse_failed"
    else:
        parse_status = "failed"

    return {
        "pred_probs": probs,
        "probability_fields_valid": valid_fields,
        "answer_text": answer_text,
        "has_answer_tag": has_answer_tag,
        "patterns": patterns,
        "pattern_output": pattern_output,
        "pair_count": len(patterns),
        "parse_status": parse_status,
        "parse_warnings": parse_warnings,
    }


class LLMService:
    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or get_settings().llm
        self.builder = build_prompt_builder(self.settings.prompt_mode)
        self.client = LLMClient(self.settings)

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "enabled": self.enabled,
            "base_url": self.settings.base_url,
            "model_name": self.settings.model_name,
            "prompt_mode": self.settings.prompt_mode,
            "pattern_guard_enabled": self.settings.effective_pattern_guard_enabled,
            "rationale_second_pass": self.settings.rationale_second_pass,
        }
        if self.enabled:
            info.update(self.client.ping())
        return info

    def _disabled(self) -> dict[str, Any]:
        return {
            "status": "disabled",
            "error": "LLM not configured (LLM_BASE_URL).",
            **{field: None for field in OUTPUT_FIELDS},
            "rationale": None,
            "rationale_source": None,
            "patterns": [],
            "pattern_output": "",
            "pair_count": 0,
            "parse_status": None,
            "parse_warnings": [],
        }

    @staticmethod
    def _pattern_analysis_fallback(patterns: list[dict[str, str]]) -> str | None:
        analyses = [
            str(item.get("analysis") or "").strip()
            for item in patterns
            if str(item.get("analysis") or "").strip()
        ]
        return "；".join(analyses) or None

    def _generate_rationale(
        self,
        patterns: list[dict[str, str]],
        probabilities: dict[str, float | None],
    ) -> tuple[str | None, str | None, str | None]:
        fallback = self._pattern_analysis_fallback(patterns)
        if not patterns:
            return fallback, "pattern_analysis" if fallback else None, None
        if not self.settings.rationale_second_pass:
            return fallback, "pattern_analysis" if fallback else None, None

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "clinical_rationale",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"rationale": {"type": "string"}},
                    "required": ["rationale"],
                    "additionalProperties": False,
                },
            },
        }
        context = {
            "patterns": patterns,
            "final_probabilities": probabilities,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是临床风险总结助手。仅根据提供的 Pattern、Analysis 和最终概率，"
                    "用中文总结总体风险原因，不新增患者事实。严格返回 JSON 对象："
                    '{"rationale":"..."}。'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False),
            },
        ]
        try:
            content = self.client.chat(
                messages,
                max_tokens=self.settings.rationale_max_tokens,
                response_format=response_format,
            )
            payload = parse_llm_json(strip_thinking_and_code_fences(content))
            rationale = payload.get("rationale") if isinstance(payload, dict) else None
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("rationale is missing or empty")
            return rationale.strip(), "vllm_second_pass", None
        except (LLMClientError, json.JSONDecodeError, TypeError, ValueError) as exc:
            LOGGER.warning("LLM rationale second pass failed: %s", exc)
            return (
                fallback,
                "pattern_analysis_fallback" if fallback else None,
                f"rationale second pass failed; Pattern Analysis fallback used: {exc}",
            )

    def predict_one(
        self,
        patient: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enabled:
            return self._disabled()

        messages = self.builder.build_messages(patient)
        try:
            content = self.client.chat(messages)
        except LLMClientError as exc:
            LOGGER.warning("LLM call failed: %s", exc)
            return {
                "status": "error",
                "error": str(exc),
                **{field: None for field in OUTPUT_FIELDS},
                "rationale": None,
                "rationale_source": None,
                "patterns": [],
                "pattern_output": "",
                "parse_status": None,
                "parse_warnings": [],
            }

        if self.settings.prompt_mode == "sft_pattern_prob":
            guarded_content = content
            filter_metadata: dict[str, Any] = {
                "enabled": self.settings.effective_pattern_guard_enabled,
                "correction_count": 0,
                "analysis_rewrite_count": 0,
                "removed_pattern_count": 0,
                "removed_patterns": [],
                "placeholder_inserted": False,
            }
            if self.settings.effective_pattern_guard_enabled:
                filter_result = correct_model_output(content)
                guarded_content = filter_result.text
                filter_metadata.update(filter_result.to_dict())

            parsed = parse_sft_pattern_response(guarded_content)
            parse_warnings = list(parsed["parse_warnings"])
            removed_count = int(filter_metadata.get("removed_pattern_count") or 0)
            if removed_count > 0:
                parse_warnings.append(
                    f"basic_info_pattern_filter removed {removed_count} pattern item(s)"
                )
                if parsed["parse_status"] == "ok":
                    parsed["parse_status"] = "format_warning"
            probabilities = dict(parsed["pred_probs"])
            invalid_fields = [
                key
                for key in LABEL_KEYS
                if not parsed["probability_fields_valid"].get(key, False)
            ]
            has_patterns = bool(parsed["patterns"])

            if not has_patterns:
                status = "error"
                parse_status = parsed["parse_status"]
            elif invalid_fields:
                status = "partial"
                parse_status = "partial_probabilities"
            else:
                status = "ok"
                parse_status = "format_warning" if parse_warnings else "ok"

            rationale, rationale_source, rationale_warning = self._generate_rationale(
                parsed["patterns"], probabilities
            )
            if rationale_warning:
                parse_warnings.append(rationale_warning)
            error = None if status in {"ok", "partial"} else (
                "; ".join(parse_warnings) or parse_status
            )
            return {
                "status": status,
                "death_probability_1m": probabilities["1m"],
                "death_probability_1y": probabilities["1y"],
                "death_probability_5y": probabilities["5y"],
                "rationale": rationale,
                "rationale_source": rationale_source,
                "patterns": parsed["patterns"],
                "pattern_output": parsed["pattern_output"],
                "pair_count": parsed["pair_count"],
                "parse_status": parse_status,
                "parse_warnings": parse_warnings,
                "basic_info_pattern_filter": filter_metadata,
                "error": error,
            }

        try:
            parsed = parse_llm_json(content)
        except json.JSONDecodeError as exc:
            return {
                "status": "error",
                "error": f"Could not parse LLM JSON output: {exc}",
                **{field: None for field in OUTPUT_FIELDS},
                "rationale": None,
                "rationale_source": None,
                "patterns": [],
                "pattern_output": "",
                "parse_status": "json_parse_failed",
                "parse_warnings": [str(exc)],
            }

        if not isinstance(parsed, dict):
            message = "LLM JSON output must be an object."
            return {
                "status": "error",
                "error": message,
                **{field: None for field in OUTPUT_FIELDS},
                "rationale": None,
                "rationale_source": None,
                "patterns": [],
                "pattern_output": "",
                "parse_status": "json_parse_failed",
                "parse_warnings": [message],
            }

        probabilities: dict[str, float | None] = {}
        valid_fields: dict[str, bool] = {}
        for key, field_name in zip(LABEL_KEYS, OUTPUT_FIELDS):
            probabilities[key], valid_fields[key] = _normalize_probability_value(
                parsed.get(field_name),
                present=field_name in parsed,
            )
        patterns = normalize_json_patterns(parsed.get("patterns"))
        warnings: list[str] = []
        invalid_fields = [key for key in LABEL_KEYS if not valid_fields[key]]
        parse_status = "partial_probabilities" if invalid_fields else "ok"
        status = "partial" if invalid_fields else "ok"

        return {
            "status": status,
            "death_probability_1m": probabilities["1m"],
            "death_probability_1y": probabilities["1y"],
            "death_probability_5y": probabilities["5y"],
            "rationale": parsed.get("rationale"),
            "rationale_source": "vllm_first_pass" if parsed.get("rationale") else None,
            "patterns": patterns,
            "pattern_output": parsed.get("pattern_output") or "",
            "pair_count": len(patterns),
            "parse_status": parse_status,
            "parse_warnings": warnings,
            "error": None,
        }

    def predict_batch(
        self,
        patients: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [self.predict_one(patient) for patient in patients]
