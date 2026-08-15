"""Strict parser for the probability-only response contract."""

from __future__ import annotations

import json
from typing import Any


REQUIRED_KEYS = {"1m", "1y", "5y"}


def parse_probability_json(text: object) -> dict[str, float | None] | None:
    """Return a validated probability object, or ``None`` for invalid output.

    The parser intentionally rejects Markdown fences, explanations, unknown
    keys, out-of-range probabilities, and all-null responses.
    """

    if not isinstance(text, str):
        return None
    try:
        value: Any = json.loads(text.strip())
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or set(value) != REQUIRED_KEYS:
        return None
    result: dict[str, float | None] = {}
    for key in ("1m", "1y", "5y"):
        item = value[key]
        if item is None:
            result[key] = None
        elif isinstance(item, bool):
            return None
        else:
            try:
                probability = float(item)
            except (TypeError, ValueError):
                return None
            if not 0.0 <= probability <= 1.0:
                return None
            result[key] = probability
    return result if any(item is not None for item in result.values()) else None
