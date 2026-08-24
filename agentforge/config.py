"""Central, validated runtime configuration without serializing secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


ROOT = Path(__file__).resolve().parents[1]


def _boolean(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"invalid boolean value: {value}")
    return normalized in {"true", "1", "yes"}


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["deterministic", "hybrid", "llm"] = "deterministic"
    provider: Literal["deterministic", "openai", "openai-compatible"] = "openai"
    base_url: str | None = None
    model: str | None = None
    api_mode: Literal["local", "responses", "chat_completions"] = "responses"
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=5)
    allow_fallback: bool = True
    enable_code_generation: bool = False
    enable_code_repair: bool = False
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def validate_mode_configuration(self) -> "LLMConfig":
        if self.mode == "deterministic":
            self.provider = "deterministic"
            self.api_mode = "local"
            self.model = None
            self.api_key = None
            self.enable_code_generation = False
            self.enable_code_repair = False
            return self
        if self.provider == "deterministic":
            raise ValueError("hybrid/llm mode requires an external LLM provider")
        if self.provider == "openai-compatible" and self.api_mode != "chat_completions":
            raise ValueError("openai-compatible provider requires chat_completions api_mode")
        return self

    @field_validator("base_url", "model")
    @classmethod
    def empty_to_none(cls, value: str | None) -> str | None:
        return value.strip() or None if isinstance(value, str) else value

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "LLMConfig":
        overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
        requested_mode = overrides.get("mode") or os.getenv("AGENTFORGE_MODE", "deterministic")
        if requested_mode != "deterministic":
            load_dotenv(ROOT / ".env", override=False)
        values: dict[str, Any] = {
            "mode": requested_mode,
            "provider": overrides.get("provider") or os.getenv("LLM_PROVIDER", "openai"),
            "base_url": overrides.get("base_url") or os.getenv("OPENAI_BASE_URL"),
            "model": overrides.get("model") or os.getenv("OPENAI_MODEL"),
            "api_mode": overrides.get("api_mode") or os.getenv("OPENAI_API_MODE", "responses"),
            "timeout_seconds": overrides.get("timeout_seconds") or os.getenv("LLM_TIMEOUT_SECONDS", "60"),
            "max_retries": overrides.get("max_retries") if "max_retries" in overrides
                           else os.getenv("LLM_MAX_RETRIES", "2"),
            "allow_fallback": _boolean(
                overrides.get("allow_fallback") if "allow_fallback" in overrides
                else os.getenv("LLM_ALLOW_FALLBACK"), True,
            ),
            "enable_code_generation": _boolean(
                overrides.get("enable_code_generation") if "enable_code_generation" in overrides
                else os.getenv("LLM_ENABLE_CODE_GENERATION"), False,
            ),
            "enable_code_repair": _boolean(
                overrides.get("enable_code_repair") if "enable_code_repair" in overrides
                else os.getenv("LLM_ENABLE_CODE_REPAIR"), False,
            ),
        }
        if requested_mode != "deterministic":
            values["api_key"] = os.getenv("OPENAI_API_KEY") or None
        return cls.model_validate(values)

    @property
    def safe_summary(self) -> dict[str, Any]:
        return self.model_dump(exclude={"api_key"})
