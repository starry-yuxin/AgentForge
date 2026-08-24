from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import networkx as nx
import pytest

from agentforge.agents import CodeAgent, KnowledgeAgent, PlannerAgent, RequirementAgent
from agentforge.knowledge import KnowledgeGraphStore, KnowledgeRetriever
from agentforge.knowledge.models import RetrievalResult


ROOT = Path(__file__).resolve().parents[1]


def _requirement(text: str | None = None, overrides=None):
    return RequirementAgent().parse(text or (
        "请构建客户流失预测模型，比较Logistic Regression和Random Forest，"
        "以F1作为主要指标，最低要求为0.60，包含缺失值、类别特征和类别不平衡"
    ), overrides)


def _knowledge_agent():
    store = KnowledgeGraphStore.load_graphml(ROOT / "knowledge" / "knowledge_graph.graphml")
    return KnowledgeAgent(KnowledgeRetriever(store))


def test_requirement_agent_parses_chinese_and_records_sources() -> None:
    requirement = _requirement()
    assert requirement.task_type == "binary_classification"
    assert requirement.candidate_algorithms == ["logistic_regression", "random_forest"]
    assert requirement.minimum_score == 0.60
    assert requirement.field_sources["minimum_score"] == "user_input"
    assert requirement.field_sources["dataset_path"] == "default_config"


def test_requirement_agent_parses_english_and_override_wins() -> None:
    requirement = _requirement(
        "Build a customer churn prediction with logistic regression and random forest, "
        "minimum F1 of 0.55 and missing values.",
        {"minimum_score": 0.7, "primary_metric": "roc_auc"},
    )
    assert requirement.minimum_score == 0.7
    assert requirement.primary_metric == "roc_auc"
    assert requirement.field_sources["minimum_score"] == "explicit_override"
    assert requirement.field_sources["primary_metric"] == "explicit_override"


def test_requirement_agent_parses_paths_target_interfaces_and_runtime() -> None:
    requirement = _requirement(
        "Customer churn prediction; dataset path sample/input.csv; target column churn_flag; "
        "provide train, predict, evaluate; maximum runtime seconds: 45; primary metric F1"
    )
    assert requirement.dataset_path == "sample/input.csv"
    assert requirement.target_column == "churn_flag"
    assert requirement.required_interfaces == ["train", "predict", "evaluate"]
    assert requirement.max_runtime_seconds == 45
    assert requirement.field_sources["dataset_path"] == "user_input"


def test_requirement_agent_rejects_explicit_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unsupported primary_metric"):
        _requirement("Customer churn prediction with primary metric profit")


def test_requirement_agent_rejects_unsupported_task_and_bad_override() -> None:
    with pytest.raises(ValueError, match="unsupported task_type"):
        _requirement("Build an image classification model")
    with pytest.raises(ValueError, match="minimum_score"):
        _requirement(overrides={"minimum_score": 1.5})


def test_knowledge_agent_calls_real_retriever_and_returns_traceable_results() -> None:
    store = KnowledgeGraphStore.load_graphml(ROOT / "knowledge" / "knowledge_graph.graphml")
    retriever = KnowledgeRetriever(store)
    retriever.query = Mock(wraps=retriever.query)
    knowledge = KnowledgeAgent(retriever).retrieve(_requirement())
    retriever.query.assert_called_once()
    assert {item.capability_id for item in knowledge.algorithms} == {
        "logistic_regression", "random_forest"
    }
    assert {"numerical_imputation", "categorical_imputation", "one_hot_encoding",
            "standard_scaling", "class_weight_balancing"} <= {
        item.capability_id for item in knowledge.preprocessors
    }
    assert all(item.source_document and item.match_reasons for item in knowledge.algorithms)


def test_knowledge_agent_fails_explicitly_when_retrieval_is_empty() -> None:
    store = KnowledgeGraphStore(nx.MultiDiGraph())
    retriever = Mock()
    retriever.store = store
    retriever.query.return_value = RetrievalResult(
        query={}, algorithms=[], preprocessors=[], metrics=[], failure_experiences=[]
    )
    with pytest.raises(LookupError, match="no supported algorithms"):
        KnowledgeAgent(retriever).retrieve(_requirement())


def test_planner_uses_knowledge_registry_and_preserves_preprocessing_difference() -> None:
    requirement = _requirement()
    knowledge = _knowledge_agent().retrieve(requirement)
    plans = PlannerAgent().plan(requirement, knowledge)
    by_algorithm = {plan.algorithm: plan for plan in plans}
    assert set(by_algorithm) == {"logistic_regression", "random_forest"}
    assert "standard_scaling" in by_algorithm["logistic_regression"].preprocessing_steps
    assert "standard_scaling" not in by_algorithm["random_forest"].preprocessing_steps
    assert "logistic_regression" in by_algorithm["logistic_regression"].supporting_capability_ids


def test_code_agent_generates_independent_importable_interfaces(tmp_path: Path) -> None:
    requirement = _requirement()
    plans = PlannerAgent().plan(requirement, _knowledge_agent().retrieve(requirement))
    artifacts = [CodeAgent().generate(plan, "run-test", tmp_path) for plan in plans]
    assert len({artifact.code_path for artifact in artifacts}) == 2
    for artifact in artifacts:
        source = Path(artifact.code_path).read_text(encoding="utf-8")
        assert artifact.syntax_valid and artifact.generator_mode == "deterministic_template"
        assert all(f"def {name}(" in source for name in ("train", "predict", "evaluate"))
        spec = importlib.util.spec_from_file_location(artifact.algorithm, artifact.code_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        assert all(callable(getattr(module, name)) for name in artifact.interface_spec)
