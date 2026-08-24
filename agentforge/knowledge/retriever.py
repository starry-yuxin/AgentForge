"""Explainable deterministic retrieval over the local knowledge graph."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agentforge.knowledge.graph_store import KnowledgeGraphStore, _slug
from agentforge.knowledge.models import Recommendation, RetrievalResult


class KnowledgeRetriever:
    def __init__(self, store: KnowledgeGraphStore) -> None:
        self.store = store

    def query(
        self,
        *,
        task_type: str,
        data_characteristics: Iterable[str] = (),
        required_metric: str | None = None,
        constraints: Iterable[str] = (),
        failure_type: str | None = None,
    ) -> RetrievalResult:
        task = _slug(task_type)
        characteristics = {_slug(value) for value in data_characteristics}
        requested_constraints = {_slug(value) for value in constraints}
        metric = _slug(required_metric) if required_metric else None

        buckets: dict[str, list[Recommendation]] = {
            "Algorithm": [], "Preprocessor": [], "Metric": [], "FailureExperience": []
        }
        for node_id, attributes in self.store.graph.nodes(data=True):
            node_type = attributes.get("node_type")
            if node_type not in buckets:
                continue
            reasons: list[str] = []
            tasks = {_slug(value) for value in attributes.get("applicable_tasks", [])}
            conditions = {_slug(value) for value in attributes.get("applicable_conditions", [])}
            node_metrics = {_slug(value) for value in attributes.get("metrics", [])}
            node_constraints = {_slug(value) for value in attributes.get("constraints", [])}

            if task in tasks or (task == "binary_classification" and "binary_classification" in tasks):
                reasons.append(f"suitable for task: {task_type}")
            matched_characteristics = sorted(characteristics & conditions)
            if matched_characteristics:
                reasons.append("matches data characteristics: " + ", ".join(matched_characteristics))
            if metric and (metric in node_metrics or _slug(attributes.get("name", "")) == metric):
                reasons.append(f"supports required metric: {required_metric}")
            matched_constraints = sorted(requested_constraints & node_constraints)
            if matched_constraints:
                reasons.append("satisfies constraints: " + ", ".join(matched_constraints))
            if failure_type and node_type == "FailureExperience" and (
                _slug(attributes.get("name", "")) == _slug(failure_type) or node_id == _slug(failure_type)
            ):
                reasons.append(f"matches failure type: {failure_type}")

            include = bool(reasons)
            if node_type == "Algorithm":
                include = task in tasks and (not metric or metric in node_metrics)
            elif node_type == "Metric":
                include = bool(metric and (_slug(attributes.get("name", "")) == metric or metric in node_metrics))
            elif node_type in {"Preprocessor", "FailureExperience"}:
                include = bool(characteristics & conditions) or any("failure type" in reason for reason in reasons)
            if not include:
                continue

            buckets[node_type].append(
                Recommendation(
                    id=node_id,
                    name=attributes.get("name", node_id),
                    node_type=node_type,
                    score=len(reasons),
                    reasons=reasons,
                    source_document=attributes.get("source_document", "graph-derived"),
                    source_section=attributes.get("source_section", "graph relationship"),
                )
            )

        for recommendations in buckets.values():
            recommendations.sort(key=lambda item: (-item.score, item.name))
        return RetrievalResult(
            query={
                "task_type": task_type,
                "data_characteristics": sorted(characteristics),
                "required_metric": required_metric,
                "constraints": sorted(requested_constraints),
                "failure_type": failure_type,
            },
            algorithms=buckets["Algorithm"],
            preprocessors=buckets["Preprocessor"],
            metrics=buckets["Metric"],
            failure_experiences=buckets["FailureExperience"],
        )

