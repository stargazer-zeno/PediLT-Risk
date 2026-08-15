"""Reward functions for probability-only GRPO."""

from __future__ import annotations

from typing import Any

from llm.common.response_parser import parse_probability_json


FORMAT_WEIGHT = 1.0
BRIER_COMPLEMENT_WEIGHT = 2.0
INVALID_WEIGHT = 1.0


def completion_to_text(completion: Any) -> str:
    """Normalize the completion shapes emitted by supported TRL versions."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list) and completion:
        return completion_to_text(completion[-1])
    return ""


def valid_binary_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    return None


def score_completion(text: str, true_labels: dict[str, Any] | None) -> float:
    """Score one completion using the documented weighted reward."""

    prediction = parse_probability_json(text)
    if prediction is None:
        return -INVALID_WEIGHT

    squared_errors = []
    labels = true_labels or {}
    for key, probability in prediction.items():
        label = valid_binary_label(labels.get(key))
        if probability is not None and label is not None:
            squared_errors.append((probability - label) ** 2)
    brier_complement = 1.0 - sum(squared_errors) / len(squared_errors) if squared_errors else 0.0
    return FORMAT_WEIGHT + BRIER_COMPLEMENT_WEIGHT * brier_complement


def probability_grpo_reward(completions: list[Any], true_labels: list[dict[str, Any]], **_: Any) -> list[float]:
    """TRL-compatible batch reward function.

    ``true_labels`` must contain the observed binary labels per prompt; null
    endpoints are ignored in the Brier term. Invalid JSON receives -1 and no
    Brier-complement reward.
    """

    if len(completions) != len(true_labels):
        raise ValueError("completions and true_labels must have equal length")
    return [score_completion(completion_to_text(item), labels) for item, labels in zip(completions, true_labels)]
