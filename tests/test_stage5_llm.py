import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from agentforge.config import LLMConfig
from agentforge.llm import BaseLLMClient, create_llm_client
from agentforge.llm.parsing import extract_json, parse_model
from agentforge.llm.openai_responses import OpenAIResponsesClient
from agentforge.llm.openai_compatible import OpenAICompatibleClient
from agentforge.workflow import WorkflowOrchestrator


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class FakeClient(BaseLLMClient):
    provider, api_mode, model = "fake", "offline", "fake-model"
    def __init__(self, responses): self.responses = iter(responses)
    def _request(self, system, user): return next(self.responses), {"total_tokens": 7}


def requirement_json():
    return json.dumps({"task_type": "binary_classification", "industry": "customer_churn",
        "dataset_path": "data/churn_sample.csv", "target_column": "churn",
        "primary_metric": "f1", "minimum_score": .6,
        "candidate_algorithms": ["logistic_regression"],
        "data_characteristics": ["missing_values", "categorical_features"],
        "constraints": ["train_validation_test_isolation"],
        "required_interfaces": ["train", "predict", "evaluate"], "max_runtime_seconds": 120})


def test_deterministic_config_erases_external_values():
    cfg = LLMConfig(mode="deterministic", api_key="secret", model="x")
    assert cfg.provider == "deterministic" and cfg.api_key is None and cfg.api_mode == "local"
    assert "api_key" not in cfg.safe_summary


def test_compatible_requires_chat_completions():
    with pytest.raises(ValueError):
        LLMConfig(mode="hybrid", provider="openai-compatible", api_mode="responses")


def test_factory_requires_key_and_model():
    with pytest.raises(Exception, match="OPENAI_API_KEY"):
        create_llm_client(LLMConfig(mode="llm", allow_fallback=False))


@pytest.mark.parametrize("text", ['{"value": 4}', '```json\n{"value": 4}\n```'])
def test_strict_json_parser(text):
    assert parse_model(text, Schema).value == 4


@pytest.mark.parametrize("text", ["nothing", '{"value": 4} trailing', '{"value": 4, "x": 1}'])
def test_strict_json_parser_rejects_bad_output(text):
    with pytest.raises(ValueError): parse_model(text, Schema)


def test_base_client_records_failure_without_throwing():
    call = FakeClient([]).call(purpose="x", prompt_version="v1", system="s", user="u")
    assert call.status == "failed" and call.provider == "fake" and call.error_type == "StopIteration"


def test_responses_adapter_uses_output_text():
    api = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw:
        SimpleNamespace(output_text='{"value": 1}', usage=SimpleNamespace(input_tokens=2, output_tokens=3, total_tokens=5))))
    cfg = LLMConfig(mode="llm", model="m", api_key="key")
    result = OpenAIResponsesClient(cfg, api).call(purpose="x", prompt_version="v", system="s", user="u")
    assert result.status == "success" and result.usage["total_tokens"] == 5


def test_provider_error_redacts_api_key():
    key = "unit-test-secret-value"
    def fail(**kw): raise RuntimeError(f"provider rejected {key}")
    api = SimpleNamespace(responses=SimpleNamespace(create=fail))
    cfg = LLMConfig(mode="llm", model="m", api_key=key)
    result = OpenAIResponsesClient(cfg, api).call(purpose="x", prompt_version="v", system="s", user="u")
    assert key not in result.error_message and "[REDACTED]" in result.error_message


def test_chat_adapter_uses_message_content():
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"value": 1}'))], usage=None)
    api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: completion)))
    cfg = LLMConfig(mode="llm", provider="openai-compatible", api_mode="chat_completions",
                    base_url="https://example.invalid/v1", model="m", api_key="key")
    assert OpenAICompatibleClient(cfg, api).call(purpose="x", prompt_version="v", system="s", user="u").status == "success"


def test_hybrid_fake_client_end_to_end(tmp_path):
    client = FakeClient([requirement_json(), '{"algorithms":["logistic_regression"]}'])
    cfg = LLMConfig(mode="hybrid", model="fake-model", api_key="unused")
    state = WorkflowOrchestrator(output_root=tmp_path, llm_config=cfg, llm_client=client).run(
        "customer churn f1 logistic regression", persist=False)
    assert state.status == "completed"
    assert state.llm_provider == "fake" and state.llm_call_count == 2
    assert state.generation_modes == {"logistic_regression": "deterministic_template"}


def test_hybrid_bad_output_falls_back(tmp_path):
    cfg = LLMConfig(mode="hybrid", model="fake-model", api_key="unused", allow_fallback=True)
    state = WorkflowOrchestrator(output_root=tmp_path, llm_config=cfg,
        llm_client=FakeClient(["bad", "bad"])).run("customer churn f1", persist=False)
    assert state.status == "completed" and state.llm_fallback_count >= 1


def test_llm_missing_key_without_fallback_fails_and_reports(tmp_path):
    cfg = LLMConfig(mode="llm", allow_fallback=False)
    state = WorkflowOrchestrator(output_root=tmp_path, llm_config=cfg).run("customer churn", persist=False)
    assert state.status == "failed" and state.best_candidate is None
    assert Path(state.final_report_paths["json"]).is_file()
