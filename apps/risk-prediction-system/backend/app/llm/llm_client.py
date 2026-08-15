from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from ..config import LLMSettings

LOGGER = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    """Thin OpenAI-compatible chat client for a remote LLM / vLLM service."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self._model_name_cache: str | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def _endpoint(self) -> str:
        base = (self.settings.base_url or "").rstrip("/")
        return f"{base}/chat/completions"

    def _models_endpoint(self) -> str:
        base = (self.settings.base_url or "").rstrip("/")
        return f"{base}/models"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

    def _resolve_model_name(self) -> str:
        if self.settings.model_name:
            return self.settings.model_name
        if self._model_name_cache:
            return self._model_name_cache
        try:
            with httpx.Client(timeout=min(self.settings.timeout_seconds, 10)) as client:
                resp = client.get(self._models_endpoint(), headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMClientError(
                "LLM_MODEL_NAME is not set and model auto-discovery failed: "
                f"{exc}"
            ) from exc

        models = data.get("data") if isinstance(data, dict) else None
        if not isinstance(models, list) or not models:
            raise LLMClientError(
                "LLM_MODEL_NAME is not set and /models returned no model ids."
            )
        first = models[0]
        model_id = first.get("id") if isinstance(first, dict) else None
        if not model_id:
            raise LLMClientError(
                "LLM_MODEL_NAME is not set and /models response has no data[0].id."
            )
        self._model_name_cache = str(model_id)
        return self._model_name_cache

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        if not self.enabled:
            raise LLMClientError("LLM is not configured (LLM_BASE_URL).")

        model_name = self._resolve_model_name()

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "stream": False,
        }
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.settings.max_tokens
        )
        if effective_max_tokens:
            payload["max_tokens"] = effective_max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if self.settings.enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.settings.enable_thinking
            }

        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                with httpx.Client(timeout=self.settings.timeout_seconds) as client:
                    resp = client.post(
                        self._endpoint(), json=payload, headers=self._headers()
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                LOGGER.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt,
                    self.settings.max_retries,
                    exc,
                )
                if attempt < self.settings.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        raise LLMClientError(f"LLM request failed after retries: {last_error}")

    def ping(self) -> dict[str, Any]:
        """Lightweight reachability check against the models endpoint."""
        if not self.enabled:
            return {"reachable": False, "reason": "not configured"}
        try:
            with httpx.Client(timeout=min(self.settings.timeout_seconds, 10)) as client:
                resp = client.get(
                    self._models_endpoint(),
                    headers=self._headers(),
                )
                info: dict[str, Any] = {
                    "reachable": resp.status_code < 500,
                    "status_code": resp.status_code,
                }
                if resp.is_success:
                    try:
                        model_name = self.settings.model_name or self._resolve_model_name()
                    except LLMClientError:
                        model_name = None
                    if model_name:
                        info["model_name"] = model_name
                return info
        except httpx.HTTPError as exc:
            return {"reachable": False, "reason": str(exc)}
