"""Agent adapter over the real stage-two knowledge retriever."""

from __future__ import annotations

from agentforge.knowledge import KnowledgeRetriever
from agentforge.models import AlgorithmRequirement, RetrievedCapability, RetrievedKnowledge


class KnowledgeAgent:
    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self.retriever = retriever

    def retrieve(self, requirement: AlgorithmRequirement) -> RetrievedKnowledge:
        result = self.retriever.query(
            task_type=requirement.task_type,
            data_characteristics=requirement.data_characteristics,
            required_metric=requirement.primary_metric,
            constraints=requirement.constraints,
        )
        if not result.algorithms:
            raise LookupError("knowledge retrieval returned no supported algorithms")

        def convert(item, extra_reason: str | None = None) -> RetrievedCapability:
            attributes = dict(self.retriever.store.graph.nodes.get(item.id, {}))
            reasons = list(item.reasons)
            if extra_reason and extra_reason not in reasons:
                reasons.append(extra_reason)
            return RetrievedCapability(
                capability_id=item.id,
                name=item.name,
                capability_type=item.node_type,
                description=attributes.get("description", ""),
                match_reasons=reasons,
                source_document=item.source_document,
                source_section=item.source_section,
                metadata=attributes,
            )

        algorithms = [convert(item) for item in result.algorithms]
        preprocessors = {item.id: convert(item) for item in result.preprocessors}
        # Follow real REQUIRES graph edges so algorithm-specific preprocessing is complete.
        for algorithm in algorithms:
            for _, target, attributes in self.retriever.store.graph.out_edges(
                algorithm.capability_id, data=True
            ):
                if attributes.get("relation") != "REQUIRES":
                    continue
                node = self.retriever.store.graph.nodes[target]
                if node.get("node_type") != "Preprocessor":
                    continue
                preprocessors[target] = RetrievedCapability(
                    capability_id=target, name=node.get("name", target),
                    capability_type="Preprocessor", description=node.get("description", ""),
                    match_reasons=[f"required by algorithm: {algorithm.capability_id}"],
                    source_document=node.get("source_document", "graph-derived"),
                    source_section=node.get("source_section", "REQUIRES relationship"),
                    metadata=dict(node),
                )
        metrics = [convert(item) for item in result.metrics]
        failures = [convert(item) for item in result.failure_experiences]
        dependencies = sorted({
            dependency
            for capability in [*algorithms, *preprocessors.values()]
            for dependency in capability.metadata.get("dependencies", [])
            if dependency.lower() in {"numpy", "pandas", "scipy", "scikit-learn", "networkx", "pydantic"}
        })
        total = len(algorithms) + len(preprocessors) + len(metrics) + len(failures)
        return RetrievedKnowledge(
            algorithms=algorithms, preprocessors=list(preprocessors.values()), metrics=metrics,
            failure_experiences=failures, dependencies=dependencies,
            retrieval_summary=(f"Retrieved {total} traceable matches for "
                               f"{requirement.task_type}/{requirement.primary_metric}."),
            total_matches=total,
        )
