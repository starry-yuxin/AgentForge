from agentforge.llm.base import BaseLLMClient, LLMConfigurationError


class DeterministicClient(BaseLLMClient):
    provider, api_mode, model = "deterministic", "local", None
    def _request(self, system: str, user: str):
        raise LLMConfigurationError("deterministic mode does not call an LLM")
