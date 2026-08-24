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
from agentforge.agents.requirement import RequirementAgent, normalize_llm_requirement


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


def test_requirement_llm_null_fields_use_trusted_defaults():
    payload = json.loads(requirement_json())
    payload["dataset_path"] = None
    requirement, call = RequirementAgent().parse_with_llm(
        "customer churn", FakeClient([json.dumps(payload)]))
    assert call.status == "success"
    assert requirement is not None
    assert requirement.dataset_path.endswith("data/churn_sample.csv")
    assert requirement.field_sources["dataset_path"] == "default_config"


@pytest.mark.parametrize("display, canonical", [
    ("F1", "f1"), (" f1 ", "f1"), ("F1 Score", "f1"), ("f1_score", "f1"),
    ("ROC-AUC", "roc_auc"), ("ROC AUC", "roc_auc"), ("roc_auc", "roc_auc"),
    ("AUC", "roc_auc"), ("Accuracy", "accuracy"), ("Precision", "precision"),
    ("Recall", "recall"),
])
def test_llm_metric_aliases(display, canonical):
    assert normalize_llm_requirement({"primary_metric": display})["primary_metric"] == canonical


@pytest.mark.parametrize("display, canonical", [
    ("Logistic Regression", "logistic_regression"),
    ("logistic-regression", "logistic_regression"),
    ("logistic_regression", "logistic_regression"),
    ("Random Forest", "random_forest"),
    ("random-forest", "random_forest"),
    ("random_forest", "random_forest"),
])
def test_llm_algorithm_aliases(display, canonical):
    result = normalize_llm_requirement({"candidate_algorithms": [display]})
    assert result["candidate_algorithms"] == [canonical]


@pytest.mark.parametrize("display", [
    "Binary Classification", "binary-classification", "binary_classification",
    "Tabular Binary Classification",
])
def test_llm_task_aliases(display):
    assert normalize_llm_requirement({"task_type": display})["task_type"] == "binary_classification"


def test_llm_algorithm_normalization_deduplicates_in_first_seen_order():
    result = normalize_llm_requirement({"candidate_algorithms": [
        "Random Forest", "logistic-regression", "random_forest", "Logistic Regression",
    ]})
    assert result["candidate_algorithms"] == ["random_forest", "logistic_regression"]


@pytest.mark.parametrize("payload", [
    {"primary_metric": "balanced accuracy"},
    {"candidate_algorithms": ["gradient boosting"]},
    {"task_type": "multiclass classification"},
])
def test_llm_unknown_aliases_are_rejected(payload):
    with pytest.raises(ValueError, match="unsupported"):
        normalize_llm_requirement(payload)


def test_llm_display_names_normalize_and_preserve_sources():
    payload = json.loads(requirement_json())
    payload.update({
        "task_type": " Binary-Classification ",
        "primary_metric": "F1 Score",
        "candidate_algorithms": ["Logistic Regression", "Random-Forest"],
        "dataset_path": None,
    })
    requirement, call = RequirementAgent().parse_with_llm(
        "customer churn", FakeClient([json.dumps(payload)]))
    assert call.status == "success"
    assert requirement.task_type == "binary_classification"
    assert requirement.primary_metric == "f1"
    assert requirement.candidate_algorithms == ["logistic_regression", "random_forest"]
    assert requirement.field_sources["task_type"] == "llm"
    assert requirement.field_sources["primary_metric"] == "llm"
    assert requirement.field_sources["candidate_algorithms"] == "llm"
    assert requirement.field_sources["dataset_path"] == "default_config"


def test_responses_adapter_uses_output_text():
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text='{"value": 1}', usage=SimpleNamespace(
            input_tokens=2, output_tokens=3, total_tokens=5))
    api = SimpleNamespace(responses=SimpleNamespace(create=create))
    cfg = LLMConfig(mode="llm", model="m", api_key="key")
    result = OpenAIResponsesClient(cfg, api).call(purpose="x", prompt_version="v", system="s", user="u")
    assert result.status == "success" and result.usage["total_tokens"] == 5
    assert captured == {"model": "m", "instructions": "s", "input": "u"}


def test_provider_error_redacts_api_key():
    key = "unit-test-secret-value"
    def fail(**kw): raise RuntimeError(f"provider rejected {key}")
    api = SimpleNamespace(responses=SimpleNamespace(create=fail))
    cfg = LLMConfig(mode="llm", model="m", api_key=key)
    result = OpenAIResponsesClient(cfg, api).call(purpose="x", prompt_version="v", system="s", user="u")
    assert key not in result.error_message and "[REDACTED]" in result.error_message


def test_chat_adapter_uses_message_content():
    captured = {}
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"value": 1}'))], usage=None)
    def create(**kwargs):
        captured.update(kwargs)
        return completion
    api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    cfg = LLMConfig(mode="llm", provider="openai-compatible", api_mode="chat_completions",
                    base_url="https://example.invalid/v1", model="m", api_key="key")
    client = OpenAICompatibleClient(cfg, api)
    assert client.call(purpose="x", prompt_version="v", system="s", user="u").status == "success"
    assert captured["model"] == "m"
    assert captured["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert all(message["role"] != "developer" for message in captured["messages"])
    assert client.base_url == "https://example.invalid/v1"


def test_compatible_message_conversion_preserves_order_and_content():
    messages = [
        {"role": "developer", "content": "first"},
        {"role": "user", "content": "question"},
        {"role": "developer", "content": "second", "name": "policy"},
        {"role": "assistant", "content": "answer"},
    ]
    assert OpenAICompatibleClient.compatible_messages(messages) == [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "question"},
        {"role": "system", "content": "second", "name": "policy"},
        {"role": "assistant", "content": "answer"},
    ]
    assert messages[0]["role"] == "developer"


def test_dotenv_loads_mode_and_process_environment_wins(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "AGENTFORGE_MODE=hybrid\nLLM_PROVIDER=openai\nOPENAI_MODEL=dotenv-model\n"
        "OPENAI_API_MODE=chat_completions\nOPENAI_API_KEY=dotenv-secret\n",
        encoding="utf-8",
    )
    for name in ("AGENTFORGE_MODE", "LLM_PROVIDER", "OPENAI_MODEL", "OPENAI_API_MODE",
                 "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = LLMConfig.from_env(env_file=dotenv)
    assert config.mode == "hybrid" and config.model == "dotenv-model"
    assert config.api_key.get_secret_value() == "dotenv-secret"
    monkeypatch.setenv("OPENAI_MODEL", "process-model")
    config = LLMConfig.from_env({"model": "override-model"}, env_file=dotenv)
    assert config.model == "override-model"
    config = LLMConfig.from_env(env_file=dotenv)
    assert config.model == "process-model"


def test_explicit_deterministic_skips_dotenv(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("AGENTFORGE_MODE=hybrid\nOPENAI_API_KEY=must-not-load\n", encoding="utf-8")
    monkeypatch.delenv("AGENTFORGE_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = LLMConfig.from_env({"mode": "deterministic"}, env_file=dotenv)
    assert config.mode == "deterministic" and config.api_key is None
    assert "OPENAI_API_KEY" not in __import__("os").environ


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
