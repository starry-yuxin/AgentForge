"""Persistence of successful workflow validation runs into the knowledge graph."""

from __future__ import annotations

from pathlib import Path

from agentforge.knowledge import KnowledgeGraphStore
from agentforge.models import WorkflowState


class PersistenceAgent:
    def persist(
        self,
        state: WorkflowState,
        *,
        graphml_path: str | Path,
        json_path: str | Path,
    ) -> bool:
        successful = [
            result.model_dump() for result in state.candidate_results
            if result.status == "completed"
        ]
        if not successful or state.best_candidate is None or state.request is None:
            raise ValueError("cannot persist a workflow without a successful best candidate")
        store = KnowledgeGraphStore.load_graphml(graphml_path)
        if "stage1_validation" not in store.graph:
            raise ValueError("source graph is missing stage1_validation")
        store.add_workflow_validation_run(
            state.run_id, results=successful,
            best_algorithm=state.best_candidate.algorithm,
            dataset_path=state.request.dataset_path,
            minimum_score=state.request.minimum_score,
        )
        store.export_graphml(graphml_path)
        store.export_json(json_path)
        return True
