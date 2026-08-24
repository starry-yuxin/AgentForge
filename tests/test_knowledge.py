from __future__ import annotations

import json
from pathlib import Path

from agentforge.knowledge import KnowledgeGraphStore, KnowledgeRetriever, load_capabilities
from agentforge.knowledge.graph_store import RELATION_TYPES
from agentforge.knowledge.importer import validate_sources


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge"


def _build_store() -> KnowledgeGraphStore:
    capabilities = load_capabilities(KNOWLEDGE_DIR / "capabilities.json")
    validate_sources(capabilities, KNOWLEDGE_DIR / "documents")
    store = KnowledgeGraphStore()
    store.build_from_capabilities(capabilities)
    return store


def _benchmark_fixture() -> dict:
    return {
        "status": "success",
        "rows": 1200,
        "positive_rate": 0.1766666667,
        "train_size": 720,
        "validation_size": 240,
        "test_size": 240,
        "best_algorithm": "logistic_regression",
        "random_state": 42,
        "candidates": [
            {
                "algorithm": "logistic_regression",
                "selected_threshold": 0.675,
                "metrics": {"f1": 0.6739, "roc_auc": 0.9145},
            },
            {
                "algorithm": "random_forest",
                "selected_threshold": 0.375,
                "metrics": {"f1": 0.6327, "roc_auc": 0.8858},
            },
        ],
    }


def test_capabilities_are_valid_and_traceable() -> None:
    capabilities = load_capabilities(KNOWLEDGE_DIR / "capabilities.json")
    validate_sources(capabilities, KNOWLEDGE_DIR / "documents")
    assert len(capabilities) >= 19
    required = {
        "BinaryClassification", "LogisticRegression", "RandomForest",
        "NumericalImputation", "CategoricalImputation", "OneHotEncoding",
        "StandardScaling", "ClassWeightBalancing", "ValidationThresholdOptimization",
        "Accuracy", "Precision", "Recall", "F1", "ROCAUC", "MissingValueError",
        "CategoricalEncodingError", "LowMinorityRecall", "LowF1AtDefaultThreshold",
        "DataLeakageRisk",
    }
    assert required <= {item.name for item in capabilities}
    assert all((KNOWLEDGE_DIR / "documents" / item.source_document).is_file() for item in capabilities)
    assert all(item.source_section for item in capabilities)


def test_required_node_and_relation_types_exist() -> None:
    store = _build_store()
    store.add_validation_run(_benchmark_fixture())
    node_types = {attributes["node_type"] for _, attributes in store.graph.nodes(data=True)}
    required_node_types = {
        "Task", "Algorithm", "Preprocessor", "Metric", "Constraint", "Dataset",
        "ValidationRun", "FailureExperience", "Dependency",
    }
    assert required_node_types <= node_types
    relation_types = {attributes["relation"] for _, _, attributes in store.graph.edges(data=True)}
    assert RELATION_TYPES <= relation_types


def test_graphml_round_trip_restores_complex_attributes(tmp_path: Path) -> None:
    store = _build_store()
    store.add_validation_run(_benchmark_fixture())
    path = tmp_path / "graph.graphml"
    store.export_graphml(path)
    loaded = KnowledgeGraphStore.load_graphml(path)
    assert loaded.graph.number_of_nodes() == store.graph.number_of_nodes()
    assert loaded.graph.number_of_edges() == store.graph.number_of_edges()
    assert loaded.graph.nodes["logistic_regression"]["inputs"] == ["encoded_numeric_matrix"]
    assert loaded.graph.nodes["stage1_validation"]["split_sizes"] == {
        "train": 720, "validation": 240, "test": 240
    }


def test_retrieval_returns_explainable_recommendations() -> None:
    result = KnowledgeRetriever(_build_store()).query(
        task_type="binary_classification",
        data_characteristics=["missing_values", "imbalanced_data"],
        required_metric="f1",
    )
    assert {item.id for item in result.algorithms} == {"logistic_regression", "random_forest"}
    assert {"numerical_imputation", "categorical_imputation", "class_weight_balancing"} <= {
        item.id for item in result.preprocessors
    }
    assert [item.id for item in result.metrics] == ["f1"]
    assert {"missing_value_error", "low_minority_recall", "low_f1_default_threshold"} <= {
        item.id for item in result.failure_experiences
    }
    assert all(item.reasons and item.source_document for group in (
        result.algorithms, result.preprocessors, result.metrics, result.failure_experiences
    ) for item in group)


def test_validation_and_failure_writeback_can_be_exported(tmp_path: Path) -> None:
    store = _build_store()
    run_id = store.add_validation_run(_benchmark_fixture())
    failure_id = store.add_failure_experience(
        "runtime_low_recall", name="RuntimeLowRecall",
        description="Observed low recall in a validation run.",
        improvements=["class_weight_balancing", "validation_threshold_optimization"],
    )
    assert store.graph.nodes[run_id]["best_algorithm"] == "logistic_regression"
    assert store.graph.nodes[failure_id]["node_type"] == "FailureExperience"
    achieved = [
        data for source, _, data in store.graph.edges(data=True)
        if source == run_id and data["relation"] == "ACHIEVED_METRIC"
    ]
    assert len(achieved) == 4
    output = tmp_path / "graph.json"
    store.export_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["directed"] is True

