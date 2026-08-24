"""Deterministic multi-agent workflow orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentforge.agents import (
    CodeAgent, KnowledgeAgent, PersistenceAgent, PlannerAgent, ReportAgent,
    RequirementAgent, ValidationAgent,
)
from agentforge.knowledge import KnowledgeGraphStore, KnowledgeRetriever
from agentforge.models import WorkflowState
from agentforge.workflow.events import invoke_traced, record_skipped


ROOT = Path(__file__).resolve().parents[2]


class WorkflowOrchestrator:
    def __init__(
        self,
        *,
        output_root: str | Path | None = None,
        graphml_path: str | Path | None = None,
        graph_json_path: str | Path | None = None,
        requirement_agent: RequirementAgent | None = None,
        validation_agent: ValidationAgent | None = None,
    ) -> None:
        self.output_root = Path(output_root or ROOT / "outputs" / "runs")
        self.graphml_path = Path(graphml_path or ROOT / "knowledge" / "knowledge_graph.graphml")
        self.graph_json_path = Path(
            graph_json_path or self.graphml_path.with_suffix(".json")
        )
        self.requirement_agent = requirement_agent or RequirementAgent()
        self.validation_agent = validation_agent or ValidationAgent()
        self.planner_agent = PlannerAgent()
        self.code_agent = CodeAgent()
        self.report_agent = ReportAgent()
        self.persistence_agent = PersistenceAgent()

    def _new_run(self) -> tuple[str, Path]:
        while True:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_id = f"run-{stamp}-{uuid4().hex[:8]}"
            run_dir = self.output_root / run_id
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return run_id, run_dir

    def run(
        self,
        request_text: str,
        *,
        overrides: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> WorkflowState:
        run_id, run_dir = self._new_run()
        state = WorkflowState(run_id=run_id, status="running")
        try:
            state.request = invoke_traced(
                state, "RequirementAgent", "natural-language request and explicit overrides",
                self.requirement_agent.parse, request_text, overrides,
            )
            store = KnowledgeGraphStore.load_graphml(self.graphml_path)
            knowledge_agent = KnowledgeAgent(KnowledgeRetriever(store))
            state.retrieved_knowledge = invoke_traced(
                state, "KnowledgeAgent",
                f"task={state.request.task_type}, metric={state.request.primary_metric}",
                knowledge_agent.retrieve, state.request,
            )
            state.candidate_plans = invoke_traced(
                state, "PlannerAgent",
                f"requested_candidates={len(state.request.candidate_algorithms)}",
                self.planner_agent.plan, state.request, state.retrieved_knowledge,
            )
            for plan in state.candidate_plans:
                artifact = invoke_traced(
                    state, f"CodeAgent[{plan.algorithm}]", f"plan_id={plan.plan_id}",
                    self.code_agent.generate, plan, run_id, run_dir,
                )
                state.generated_artifacts.append(artifact)
                result = invoke_traced(
                    state, f"ValidationAgent[{plan.algorithm}]",
                    f"algorithm={plan.algorithm}, dataset={Path(state.request.dataset_path).name}",
                    self.validation_agent.validate, artifact, state.request, plan,
                )
                state.candidate_results.append(result)

            successful = [result for result in state.candidate_results if result.status == "completed"]
            if not successful:
                state.status = "failed"
            else:
                state.best_candidate = max(
                    successful,
                    key=lambda result: float(result.selection_metric_value or float("-inf")),
                )
                state.status = (
                    "completed" if len(successful) == len(state.candidate_results)
                    else "partially_completed"
                )
            state.final_report_paths = invoke_traced(
                state, "ReportAgent", f"candidate_results={len(state.candidate_results)}",
                self.report_agent.generate, state, run_dir,
            )
            if persist and successful:
                try:
                    state.knowledge_persisted = invoke_traced(
                        state, "PersistenceAgent", f"run_id={state.run_id}",
                        self.persistence_agent.persist, state,
                        graphml_path=self.graphml_path, json_path=self.graph_json_path,
                    )
                except Exception as exc:
                    state.errors.append(f"PersistenceAgent: {type(exc).__name__}: {exc}")
                    if state.status == "completed":
                        state.status = "partially_completed"
            else:
                record_skipped(
                    state, "PersistenceAgent",
                    "Persistence disabled." if not persist else "No successful candidate to persist.",
                )
            # Refresh both reports with final persistence status and trace.
            state.final_report_paths = self.report_agent.generate(state, run_dir)
            return state
        except Exception as exc:
            state.errors.append(f"{type(exc).__name__}: {exc}")
            state.status = "failed"
            try:
                state.final_report_paths = self.report_agent.generate(state, run_dir)
            except Exception as report_exc:
                state.errors.append(f"ReportAgent: {type(report_exc).__name__}: {report_exc}")
            return state
