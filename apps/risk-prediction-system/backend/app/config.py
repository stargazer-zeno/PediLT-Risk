from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _optional_bool_env(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _project_root() -> Path:
    # app/config.py -> app -> backend -> project root
    return Path(__file__).resolve().parents[2]


@dataclass
class MLSettings:
    """Where to discover the XGBoost artifacts.

    The validated schema.json and xgb_sequence_Label_*.json are bundled with
    this research release. Environment variables may override their location
    for validation or development.
    """

    train_v2_root: str | None = field(
        default_factory=lambda: os.environ.get("TRAIN_V2_ROOT")
    )
    schema_path: str | None = field(
        default_factory=lambda: os.environ.get("SCHEMA_PATH")
    )
    model_dir: str | None = field(default_factory=lambda: os.environ.get("MODEL_DIR"))

    def resolved(self) -> "MLSettings":
        """Fall back to the bundled ./artifacts directory when nothing is set."""
        if not any([self.train_v2_root, self.schema_path, self.model_dir]):
            artifacts = _project_root() / "artifacts"
            schema = artifacts / "datasets" / "schema.json"
            models = artifacts / "xgboost"
            if schema.exists() and models.exists():
                return MLSettings(
                    train_v2_root=str(artifacts),
                    schema_path=str(schema),
                    model_dir=str(models),
                )
        return self


@dataclass
class LLMSettings:
    base_url: str | None = field(default_factory=lambda: os.environ.get("LLM_BASE_URL"))
    api_key: str = field(
        default_factory=lambda: os.environ.get("LLM_API_KEY", "EMPTY")
    )
    model_name: str | None = field(
        default_factory=lambda: os.environ.get("LLM_MODEL_NAME")
    )
    prompt_mode: str = field(
        default_factory=lambda: os.environ.get("LLM_PROMPT_MODE", "generic_json")
        .strip()
        .lower()
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.environ.get("LLM_TEMPERATURE", "0.0"))
    )
    top_p: float = field(
        default_factory=lambda: float(os.environ.get("LLM_TOP_P", "1.0"))
    )
    max_tokens: int | None = field(
        default_factory=lambda: (
            int(os.environ["LLM_MAX_TOKENS"])
            if os.environ.get("LLM_MAX_TOKENS")
            else None
        )
    )
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("LLM_MAX_RETRIES", "3"))
    )
    enable_thinking: bool | None = field(
        default_factory=lambda: _optional_bool_env("LLM_ENABLE_THINKING")
    )
    pattern_guard_enabled: bool | None = field(
        default_factory=lambda: _optional_bool_env("LLM_PATTERN_GUARD_ENABLED")
    )
    rationale_second_pass: bool = field(
        default_factory=lambda: _bool_env("LLM_RATIONALE_SECOND_PASS", True)
    )
    rationale_max_tokens: int = field(
        default_factory=lambda: int(
            os.environ.get("LLM_RATIONALE_MAX_TOKENS", "768")
        )
    )

    @property
    def enabled(self) -> bool:
        # model_name can be auto-discovered from OpenAI-compatible /models.
        return bool(self.base_url)

    @property
    def effective_pattern_guard_enabled(self) -> bool:
        if self.pattern_guard_enabled is not None:
            return self.pattern_guard_enabled
        return self.prompt_mode == "sft_pattern_prob"


@dataclass
class Settings:
    ml: MLSettings = field(default_factory=lambda: MLSettings().resolved())
    llm: LLMSettings = field(default_factory=LLMSettings)

    jobs_dir: str = field(
        default_factory=lambda: os.environ.get(
            "JOBS_DIR", str(_project_root() / "data" / "jobs")
        )
    )
    max_upload_mb: int = field(
        default_factory=lambda: int(os.environ.get("MAX_UPLOAD_MB", "50"))
    )
    job_workers: int = field(
        default_factory=lambda: int(os.environ.get("JOB_WORKERS", "2"))
    )
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.environ.get("CORS_ORIGINS", "*").split(",")
            if o.strip()
        ]
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:  # pragma: no cover - test helper
    global _settings
    _settings = None
