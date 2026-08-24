from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agentforge.agents import CodeAgent, KnowledgeAgent, PlannerAgent, RequirementAgent, ValidationAgent
from agentforge.knowledge import KnowledgeGraphStore, KnowledgeRetriever
from agentforge.models import CandidateResult
from agentforge.workflow import WorkflowOrchestrator


ROOT = Path(__file__).resolve().parents[1]
REQUEST = (
    "请构建客户流失预测模型，比较Logistic Regression和Random Forest，"
    "以F1作为主要指标，最低要求为0.60，包含缺失值、类别特征和类别不平衡"
)


def _agent_inputs(tmp_path: Path, algorithm: str = "logistic_regression"):
    requirement = RequirementAgent().parse(REQUEST)
    store = KnowledgeGraphStore.load_graphml(ROOT / "knowledge" / "knowledge_graph.graphml")
    knowledge = KnowledgeAgent(KnowledgeRetriever(store)).retrieve(requirement)
    plan = next(
        plan for plan in PlannerAgent().plan(requirement, knowledge)
        if plan.algorithm == algorithm
    )
    artifact = CodeAgent().generate(plan, "run-validation", tmp_path)
    return requirement, plan, artifact


def _temp_graph(tmp_path: Path) -> tuple[Path, Path]:
    graphml = tmp_path / "knowledge_graph.graphml"
    graph_json = tmp_path / "knowledge_graph.json"
    shutil.copy2(ROOT / "knowledge" / "knowledge_graph.graphml", graphml)
    shutil.copy2(ROOT / "knowledge" / "knowledge_graph.json", graph_json)
    return graphml, graph_json


def test_validation_agent_reexecutes_candidate_and_checks_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requirement, plan, artifact = _agent_inputs(tmp_path)
    called = []

    def fake_execution(data_path, algorithm):
        called.append((data_path, algorithm))
        return {
            "selected_threshold": 0.6,
            "validation_metrics": {"accuracy": 0.8, "precision": 0.7, "recall": 0.8,
                                   "f1": 0.65, "roc_auc": 0.9},
            "metrics": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9,
                        "f1": 0.99, "roc_auc": 0.99, "runtime_seconds": 0.01,
                        "confusion_matrix": [[1, 0], [0, 1]]},
        }

    monkeypatch.setattr("agentforge.agents.validation.evaluate_candidate", fake_execution)
    result = ValidationAgent().validate(artifact, requirement, plan)
    assert called == [(requirement.dataset_path, "logistic_regression")]
    assert result.minimum_score_met
    assert result.selection_metric_value == 0.65
    assert result.validation_metrics["f1"] != result.test_metrics["f1"]
    assert Path(artifact.result_output_path).is_file()


def test_validation_agent_returns_failed_result_without_raising(tmp_path: Path) -> None:
    requirement, plan, artifact = _agent_inputs(tmp_path)
    requirement = requirement.model_copy(update={"dataset_path": str(tmp_path / "missing.csv")})
    result = ValidationAgent().validate(artifact, requirement, plan)
    assert result.status == "failed"
    assert "FileNotFoundError" in result.error


class OneFailureValidationAgent(ValidationAgent):
    def validate(self, artifact, requirement, plan):
        if plan.algorithm == "logistic_regression":
            return CandidateResult(
                plan_id=plan.plan_id, algorithm=plan.algorithm, status="failed",
                generated_code_path=artifact.code_path,
                selection_metric_name=requirement.primary_metric,
                error="InjectedError: isolated candidate failure",
            )
        return super().validate(artifact, requirement, plan)


class AllFailureValidationAgent(ValidationAgent):
    def validate(self, artifact, requirement, plan):
        return CandidateResult(
            plan_id=plan.plan_id, algorithm=plan.algorithm, status="failed",
            generated_code_path=artifact.code_path,
            selection_metric_name=requirement.primary_metric,
            error="InjectedError: all candidates failed",
        )


def test_one_candidate_failure_allows_other_candidate_and_partial_status(tmp_path: Path) -> None:
    graphml, graph_json = _temp_graph(tmp_path)
    workflow = WorkflowOrchestrator(
        output_root=tmp_path / "runs", graphml_path=graphml,
        graph_json_path=graph_json, validation_agent=OneFailureValidationAgent(),
    )
    state = workflow.run(REQUEST, persist=False)
    assert state.status == "partially_completed"
    assert [result.status for result in state.candidate_results] == ["failed", "completed"]
    assert state.best_candidate.algorithm == "random_forest"


def test_all_candidate_failures_produce_failed_workflow(tmp_path: Path) -> None:
    graphml, graph_json = _temp_graph(tmp_path)
    state = WorkflowOrchestrator(
        output_root=tmp_path / "runs", graphml_path=graphml,
        graph_json_path=graph_json, validation_agent=AllFailureValidationAgent(),
    ).run(REQUEST, persist=False)
    assert state.status == "failed"
    assert state.best_candidate is None
    assert all(result.status == "failed" for result in state.candidate_results)


def test_complete_workflow_uses_validation_for_selection_and_writes_reports(tmp_path: Path) -> None:
    formal_graph_before = (ROOT / "knowledge" / "knowledge_graph.graphml").read_bytes()
    graphml, graph_json = _temp_graph(tmp_path)
    workflow = WorkflowOrchestrator(
        output_root=tmp_path / "runs", graphml_path=graphml, graph_json_path=graph_json,
    )
    state = workflow.run(REQUEST, persist=False)
    assert state.status == "completed"
    assert state.best_candidate.algorithm == max(
        state.candidate_results, key=lambda item: item.validation_metrics["f1"]
    ).algorithm
    assert state.best_candidate.selection_metric_name == "f1"
    assert all("validation" in " ".join(result.validation_messages).lower()
               for result in state.candidate_results)
    assert Path(state.final_report_paths["json"]).is_file()
    assert Path(state.final_report_paths["markdown"]).is_file()
    report = json.loads(Path(state.final_report_paths["json"]).read_text(encoding="utf-8"))
    assert report["run_id"] == state.run_id
    assert report["best_candidate"]["algorithm"] == state.best_candidate.algorithm
    assert (ROOT / "knowledge" / "knowledge_graph.graphml").read_bytes() == formal_graph_before

    statuses = [event.status for event in state.events]
    assert statuses[0:2] == ["started", "completed"]
    assert statuses[-1] == "skipped"
    assert all(event.duration_seconds >= 0 for event in state.events)


def test_run_ids_and_output_directories_are_unique(tmp_path: Path) -> None:
    graphml, graph_json = _temp_graph(tmp_path)
    workflow = WorkflowOrchestrator(
        output_root=tmp_path / "runs", graphml_path=graphml, graph_json_path=graph_json,
    )
    first = workflow.run(REQUEST, persist=False)
    second = workflow.run(REQUEST, persist=False)
    assert first.run_id != second.run_id
    assert Path(first.final_report_paths["json"]).parents[1] != Path(
        second.final_report_paths["json"]
    ).parents[1]


def test_persistence_writes_unique_validation_run_to_temporary_graph(tmp_path: Path) -> None:
    graphml, graph_json = _temp_graph(tmp_path)
    state = WorkflowOrchestrator(
        output_root=tmp_path / "runs", graphml_path=graphml, graph_json_path=graph_json,
    ).run(REQUEST, persist=True)
    assert state.knowledge_persisted and state.status == "completed"
    store = KnowledgeGraphStore.load_graphml(graphml)
    assert "stage1_validation" in store.graph
    assert state.run_id in store.graph
    edges = [data for source, _, data in store.graph.edges(data=True) if source == state.run_id]
    achieved = [data for data in edges if data["relation"] == "ACHIEVED_METRIC"]
    assert {data["split"] for data in achieved} == {"validation", "test"}
    assert len([data for data in edges if data["relation"] == "USED_ALGORITHM"]) == 2
