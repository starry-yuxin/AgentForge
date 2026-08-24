"""Auditable, secret-free LLM call records."""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class LLMCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    call_id: str
    purpose: str
    provider: str
    model: str | None = None
    api_mode: str
    status: Literal["success", "failed", "fallback"]
    response_text: str = ""
    parsed_output: Any = None
    prompt_version: str
    attempts: int = Field(default=1, ge=0)
    duration_seconds: float = Field(default=0, ge=0)
    fallback_used: bool = False
    error_type: str | None = None
    error_message: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
