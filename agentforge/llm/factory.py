from agentforge.config import LLMConfig
from agentforge.llm.base import LLMConfigurationError
from agentforge.llm.deterministic import DeterministicClient
from agentforge.llm.openai_compatible import OpenAICompatibleClient
from agentforge.llm.openai_responses import OpenAIResponsesClient


def create_llm_client(config: LLMConfig, *, client=None):
    if config.mode == "deterministic": return DeterministicClient()
    if not config.api_key or not config.model:
        raise LLMConfigurationError("LLM mode requires OPENAI_API_KEY and OPENAI_MODEL")
    if config.provider == "openai" and config.api_mode == "responses":
        return OpenAIResponsesClient(config, client)
    if config.api_mode == "chat_completions":
        return OpenAICompatibleClient(config, client)
    raise LLMConfigurationError("unsupported provider/API mode combination")
