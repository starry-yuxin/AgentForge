"""Official OpenAI Responses API adapter."""

from openai import OpenAI
from agentforge.llm.base import BaseLLMClient


class OpenAIResponsesClient(BaseLLMClient):
    provider, api_mode = "openai", "responses"
    def __init__(self, config, client=None):
        self.model = config.model
        self._secret_values = (config.api_key.get_secret_value(),)
        self.client = client or OpenAI(api_key=config.api_key.get_secret_value(),
            base_url=config.base_url, timeout=config.timeout_seconds, max_retries=config.max_retries)
    def _request(self, system, user):
        response = self.client.responses.create(model=self.model, instructions=system, input=user)
        usage_obj = getattr(response, "usage", None)
        usage = {name: int(getattr(usage_obj, name, 0) or 0) for name in
                 ("input_tokens", "output_tokens", "total_tokens")} if usage_obj else {}
        return response.output_text, usage
