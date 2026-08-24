"""NetworkX graph construction, persistence, and validation-result write-back."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from agentforge.knowledge.models import Capability


JSON_PREFIX = "__json__:"
RELATION_TYPES = {
    "SUITABLE_FOR", "REQUIRES", "EVALUATED_BY", "HANDLES", "SATISFIES",
    "USED_ALGORITHM", "PERFORMED_ON", "FAILED_BECAUSE", "IMPROVED_BY",
    "ACHIEVED_METRIC", "DEPENDS_ON", "DERIVED_FROM",
}


def _slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _graphml_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)) or value is None:
        return JSON_PREFIX + json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _decoded_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(JSON_PREFIX):
        return json.loads(value[len(JSON_PREFIX):])
    return value


class KnowledgeGraphStore:
    def __init__(self, graph: nx.MultiDiGraph | None = None) -> None:
        self.graph = graph or nx.MultiDiGraph(name="AgentForge Knowledge Graph")

    def add_edge(self, source: str, target: str, relation: str, **attributes: Any) -> None:
        if relation not in RELATION_TYPES:
            raise ValueError(f"unsupported relation type: {relation}")
        self.graph.add_edge(source, target, relation=relation, **attributes)

    def build_from_capabilities(self, capabilities: list[Capability]) -> None:
        self.graph.clear()
        by_name = {_slug(item.name): item.id for item in capabilities}
        by_name.update({item.id: item.id for item in capabilities})
        types_by_id = {item.id: item.type for item in capabilities}
        for item in capabilities:
            attributes = item.model_dump()
            attributes.pop("id")
            attributes["node_type"] = attributes.pop("type")
            self.graph.add_node(item.id, **attributes)

            document_id = f"document:{_slug(item.source_document)}"
            self.graph.add_node(
                document_id, node_type="SourceDocument", name=item.source_document,
                description="Project built-in example source material",
            )
            self.add_edge(item.id, document_id, "DERIVED_FROM", section=item.source_section)

            for dependency in item.dependencies:
                target = by_name.get(_slug(dependency))
                if target:
                    relation = "REQUIRES" if types_by_id[target] == "Preprocessor" else "DEPENDS_ON"
                else:
                    target = f"dependency:{_slug(dependency)}"
                    self.graph.add_node(target, node_type="Dependency", name=dependency)
                    relation = "DEPENDS_ON"
                self.add_edge(item.id, target, relation)

            for constraint in item.constraints:
                constraint_id = f"constraint:{_slug(constraint)}"
                self.graph.add_node(constraint_id, node_type="Constraint", name=constraint)
                self.add_edge(item.id, constraint_id, "SATISFIES")

        task_id = "binary_classification"
        for item in capabilities:
            if item.type == "Algorithm" and task_id in item.applicable_tasks:
                self.add_edge(item.id, task_id, "SUITABLE_FOR")
            if item.type in {"Algorithm", "Task"}:
                for metric in item.metrics:
                    metric_id = by_name.get(_slug(metric))
                    if metric_id:
                        self.add_edge(item.id, metric_id, "EVALUATED_BY")
            for condition in item.applicable_conditions:
                characteristic_id = f"constraint:{_slug(condition)}"
                self.graph.add_node(characteristic_id, node_type="Constraint", name=condition)
                self.add_edge(item.id, characteristic_id, "HANDLES")
            if item.type == "FailureExperience":
                for dependency in item.dependencies:
                    target = by_name.get(_slug(dependency))
                    if target and types_by_id[target] == "Preprocessor":
                        self.add_edge(item.id, target, "IMPROVED_BY")

        # Seed one traceable historical failure so FAILED_BECAUSE is represented.
        self.graph.add_node(
            "historical_missing_value_failure", node_type="ValidationRun",
            name="Historical missing-value failure example", status="failed",
            disclaimer="Project built-in example failure, not a production run.",
        )
        self.add_edge("historical_missing_value_failure", "missing_value_error", "FAILED_BECAUSE")

    def add_validation_run(
        self,
        benchmark: dict[str, Any],
        *,
        run_id: str = "stage1_validation",
        dataset_id: str = "customer_churn_sample",
    ) -> str:
        self.graph.add_node(
            dataset_id,
            node_type="Dataset",
            name=dataset_id,
            rows=benchmark["rows"],
            positive_rate=benchmark["positive_rate"],
            characteristics=["tabular_data", "missing_values", "imbalanced_data"],
        )
        self.graph.add_node(
            run_id,
            node_type="ValidationRun",
            name="Stage 1 deterministic churn benchmark",
            status=benchmark["status"],
            best_algorithm=benchmark["best_algorithm"],
            random_state=benchmark["random_state"],
            created_at=datetime.now(timezone.utc).isoformat(),
            disclaimer=("Fixed-random-seed synthetic-data demonstration; results do not "
                        "represent real-business generalization performance."),
            split_sizes={
                "train": benchmark["train_size"],
                "validation": benchmark["validation_size"],
                "test": benchmark["test_size"],
            },
        )
        self.add_edge(run_id, dataset_id, "PERFORMED_ON")
        algorithm_ids = {
            "logistic_regression": "logistic_regression",
            "random_forest": "random_forest",
        }
        metric_ids = {"f1": "f1", "roc_auc": "roc_auc"}
        for candidate in benchmark["candidates"]:
            algorithm_id = algorithm_ids[candidate["algorithm"]]
            self.add_edge(
                run_id, algorithm_id, "USED_ALGORITHM",
                selected_threshold=candidate["selected_threshold"],
                selected_as_best=candidate["algorithm"] == benchmark["best_algorithm"],
            )
            for metric_name, metric_id in metric_ids.items():
                self.add_edge(
                    run_id, metric_id, "ACHIEVED_METRIC",
                    algorithm=algorithm_id,
                    value=candidate["metrics"][metric_name],
                    split="test",
                )
        return run_id

    def add_failure_experience(
        self,
        failure_id: str,
        *,
        name: str,
        description: str,
        improvements: list[str],
        source_document: str = "runtime_writeback",
    ) -> str:
        self.graph.add_node(
            failure_id, node_type="FailureExperience", name=name,
            description=description, source_document=source_document,
            source_section="runtime write-back", version="1.0.0",
        )
        for improvement in improvements:
            if improvement not in self.graph:
                raise ValueError(f"unknown improvement node: {improvement}")
            self.add_edge(failure_id, improvement, "IMPROVED_BY")
        return failure_id

    def add_workflow_validation_run(
        self,
        run_id: str,
        *,
        results: list[dict[str, Any]],
        best_algorithm: str,
        dataset_path: str,
        minimum_score: float,
    ) -> str:
        """Persist a stage-three run without replacing any historical validation node."""
        if run_id in self.graph:
            raise ValueError(f"validation run already exists: {run_id}")
        dataset_id = "customer_churn_sample"
        self.graph.add_node(
            dataset_id, node_type="Dataset", name=dataset_id, source_path=dataset_path,
            characteristics=["tabular_data", "missing_values", "imbalanced_data"],
        )
        self.graph.add_node(
            run_id, node_type="ValidationRun", name="Stage 3 deterministic agent workflow",
            status="success", best_algorithm=best_algorithm, minimum_score=minimum_score,
            created_at=datetime.now(timezone.utc).isoformat(),
            disclaimer=("Fixed-random-seed synthetic-data demonstration; results do not "
                        "represent real-business generalization performance."),
        )
        self.add_edge(run_id, dataset_id, "PERFORMED_ON")
        for result in results:
            algorithm = result["algorithm"]
            self.add_edge(
                run_id, algorithm, "USED_ALGORITHM",
                selected_threshold=result["selected_threshold"],
                selected_as_best=algorithm == best_algorithm,
                minimum_score_met=result["minimum_score_met"],
            )
            for split_name, metrics in (
                ("validation", result["validation_metrics"]),
                ("test", result["test_metrics"]),
            ):
                for metric_name in ("f1", "roc_auc"):
                    if metric_name in metrics:
                        self.add_edge(
                            run_id, metric_name, "ACHIEVED_METRIC", algorithm=algorithm,
                            value=metrics[metric_name], split=split_name,
                        )
        return run_id

    def export_graphml(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        safe = nx.MultiDiGraph(name=self.graph.graph.get("name", "AgentForge Knowledge Graph"))
        for node_id, attributes in self.graph.nodes(data=True):
            safe.add_node(node_id, **{key: _graphml_value(value) for key, value in attributes.items()})
        for source, target, key, attributes in self.graph.edges(keys=True, data=True):
            safe.add_edge(
                source, target, key=key,
                **{name: _graphml_value(value) for name, value in attributes.items()},
            )
        nx.write_graphml(safe, destination)

    def export_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = nx.node_link_data(self.graph, edges="edges")
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_graphml(cls, path: str | Path) -> "KnowledgeGraphStore":
        loaded = nx.read_graphml(Path(path), force_multigraph=True)
        graph = nx.MultiDiGraph()
        for node_id, attributes in loaded.nodes(data=True):
            graph.add_node(node_id, **{key: _decoded_value(value) for key, value in attributes.items()})
        for source, target, key, attributes in loaded.edges(keys=True, data=True):
            graph.add_edge(
                source, target, key=key,
                **{name: _decoded_value(value) for name, value in attributes.items()},
            )
        return cls(graph)
