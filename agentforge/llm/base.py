"""Provider-neutral synchronous LLM interface."""

from __future__ import annotations
import time
from abc import ABC, abstractmethod
from uuid import uuid4
from agentforge.llm.models import LLMCallResult


class LLMError(RuntimeError): pass
class LLMConfigurationError(LLMError): pass


class BaseLLMClient(ABC):
    provider = "unknown"
    api_mode = "unknown"
    model: str | None = None

    def _sanitize(self, value: str) -> str:
        for secret in getattr(self, "_secret_values", ()):
            if secret: value = value.replace(secret, "[REDACTED]")
        return value

    @abstractmethod
    def _request(self, system: str, user: str) -> tuple[str, dict[str, int]]: ...

    def call(self, *, purpose: str, prompt_version: str, system: str, user: str) -> LLMCallResult:
        started = time.perf_counter()
        try:
            text, usage = self._request(system, user)
            if not text.strip():
                raise LLMError("provider returned an empty response")
            return LLMCallResult(call_id=f"llm-{uuid4().hex[:12]}", purpose=purpose,
                provider=self.provider, model=self.model, api_mode=self.api_mode,
                status="success", response_text=text, prompt_version=prompt_version,
                duration_seconds=time.perf_counter() - started, usage=usage)
        except Exception as exc:
            message = self._sanitize(str(exc))[:500]
            return LLMCallResult(call_id=f"llm-{uuid4().hex[:12]}", purpose=purpose,
                provider=self.provider, model=self.model, api_mode=self.api_mode,
                status="failed", prompt_version=prompt_version,
                duration_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__, error_message=message)
