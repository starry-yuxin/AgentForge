from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agentforge.agents import CodeAgent, KnowledgeAgent, PlannerAgent, RepairAgent, RequirementAgent, ValidationAgent
from agentforge.knowledge import KnowledgeGraphStore, KnowledgeRetriever
from agentforge.models import ExecutionResult
from agentforge.repair import ErrorClassifier
from agentforge.validation import AstSecurityChecker, InterfaceChecker
from agentforge.workflow import WorkflowOrchestrator


ROOT = Path(__file__).resolve().parents[1]
REQUEST = (
    "请构建客户流失预测模型，比较Logistic Regression和Random Forest，"
    "以F1作为主要指标，最低要求为0.60"
)
METRICS = {"accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65,
           "roc_auc": 0.85, "runtime_seconds": 0.0, "confusion_matrix": [[1, 0], [0, 1]]}


def _inputs(tmp_path: Path):
    requirement = RequirementAgent().parse(REQUEST)
    store = KnowledgeGraphStore.load_graphml(ROOT / "knowledge" / "knowledge_graph.graphml")
    retriever = KnowledgeRetriever(store)
    knowledge = KnowledgeAgent(retriever).retrieve(requirement)
    plan = PlannerAgent().plan(requirement, knowledge)[0]
    artifact = CodeAgent().generate(plan, "run-test", tmp_path)
    return requirement, plan, artifact, retriever


def _execution(artifact, payload: dict | None, *, status="completed", stderr=""):
    Path(artifact.model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(artifact.result_output_path).parent.mkdir(parents=True, exist_ok=True)
    if status == "completed":
        Path(artifact.model_output_path).touch()
    if payload is not None:
        Path(artifact.result_output_path).write_text(json.dumps(payload), encoding="utf-8")
    return ExecutionResult(
        attempt=artifact.attempt, command=["<venv-python>"], cwd="attempt-0",
        return_code=0 if status == "completed" else 1,
        stderr=stderr, result_json_path=artifact.result_output_path,
        model_path=artifact.model_output_path, process_status=status,
    )


def _payload(**changes):
    payload = {
        "schema_version": "1.0", "prediction_count": 1200,
        "prediction_labels": [0, 1],
        "evaluation": {
            "algorithm": "logistic_regression", "selected_threshold": 0.6,
            "validation_metrics": dict(METRICS), "test_metrics": dict(METRICS),
            "split_sizes": {"train": 720, "validation": 240, "test": 240},
        },
    }
    payload.update(changes)
    return payload


def _assess(tmp_path: Path, payload):
    requirement, plan, artifact, _ = _inputs(tmp_path)
    security = AstSecurityChecker().check(artifact.code_path)
    interface = InterfaceChecker().check(artifact.code_path)
    execution = _execution(artifact, payload)
    return ValidationAgent().assess(
        artifact, requirement, plan, security, interface, execution
    )


def test_unified_validation_accepts_valid_schema_and_uses_validation_minimum(tmp_path: Path) -> None:
    result = _assess(tmp_path, _payload())
    assert result.status == "completed" and result.minimum_score_met
    assert result.selection_metric_value == result.validation_metrics["f1"]
    assert result.validation_metrics is not result.test_metrics
    assert all(check.category in {
        "syntax", "security", "interface", "execution", "functionality", "metrics", "resource"
    } for check in result.validation_checks)


@pytest.mark.parametrize(("mutation", "failed_check"), [
    (lambda payload: payload.update(prediction_count=1199), "prediction_length"),
    (lambda payload: payload.update(prediction_labels=[0, 2]), "prediction_labels"),
    (lambda payload: payload["evaluation"]["validation_metrics"].update(f1=float("nan")),
     "metric_schema"),
    (lambda payload: payload["evaluation"]["test_metrics"].pop("roc_auc"), "metric_schema"),
    (lambda payload: payload["evaluation"].pop("validation_metrics"), "split_separation"),
])
def test_unified_validation_rejects_bad_outputs(tmp_path: Path, mutation, failed_check: str) -> None:
    payload = _payload()
    mutation(payload)
    result = _assess(tmp_path, payload)
    assert result.status == "failed"
    assert not next(check for check in result.validation_checks if check.name == failed_check).passed


def test_missing_result_json_and_model_are_detected(tmp_path: Path) -> None:
    requirement, plan, artifact, _ = _inputs(tmp_path)
    security = AstSecurityChecker().check(artifact.code_path)
    interface = InterfaceChecker().check(artifact.code_path)
    execution = ExecutionResult(
        attempt=0, command=["<venv-python>"], cwd="attempt-0", return_code=0,
        result_json_path=artifact.result_output_path, model_path=artifact.model_output_path,
        process_status="failed",
    )
    result = ValidationAgent().assess(
        artifact, requirement, plan, security, interface, execution
    )
    assert not next(check for check in result.validation_checks if check.name == "result_json").passed
    assert not next(check for check in result.validation_checks if check.name == "model_artifact").passed


def test_classifier_uses_structured_security_interface_timeout_and_real_error(tmp_path: Path) -> None:
    requirement, plan, artifact, _ = _inputs(tmp_path)
    security = AstSecurityChecker().check(artifact.code_path)
    interface = InterfaceChecker().check(artifact.code_path)
    execution = _execution(
        artifact, None, status="failed",
        stderr="ValueError: Input X contains NaN. Use SimpleImputer.",
    )
    classifier = ErrorClassifier()
    assert classifier.classify(
        execution=execution, security=security, interface=interface, checks=[]
    ) == "MissingValueError"
    assert classifier.classify(
        execution=execution.model_copy(update={"timed_out": True, "process_status": "timed_out"}),
        security=security, interface=interface, checks=[],
    ) == "TimeoutError"


def test_repair_agent_preserves_source_writes_diff_and_uses_knowledge(tmp_path: Path) -> None:
    requirement, plan, artifact, retriever = _inputs(tmp_path)
    faulty = CodeAgent().generate(
        plan, "run-test", tmp_path, failure_injection="missing_imputer"
    )
    before = Path(faulty.code_path).read_text(encoding="utf-8")
    record, repaired = RepairAgent(retriever).repair(
        requirement, plan, faulty, "MissingValueError", [], attempt=1,
        run_id="run-test", run_dir=tmp_path, error_summary="Input X contains NaN",
    )
    assert Path(faulty.code_path).read_text(encoding="utf-8") == before
    assert Path(repaired.code_path).is_file() and repaired.code_path != faulty.code_path
    assert Path(record.diff_path).is_file()
    assert "train_candidate_missing_imputer" in Path(record.diff_path).read_text(encoding="utf-8")
    assert record.retrieved_experience_ids == ["missing_value_error"]
    assert "numerical_imputation" in record.repair_strategy
    with pytest.raises(ValueError, match="attempt"):
        RepairAgent(retriever).repair(
            requirement, plan, artifact, "MissingValueError", [], attempt=3,
            run_id="run-test", run_dir=tmp_path, error_summary="x",
        )
    with pytest.raises(ValueError, match="not deterministically repairable"):
        RepairAgent(retriever).repair(
            requirement, plan, artifact, "SecurityViolation", [], attempt=1,
            run_id="run-test", run_dir=tmp_path, error_summary="x",
        )


def _temp_graph(tmp_path: Path):
    graphml = tmp_path / "knowledge_graph.graphml"
    graph_json = tmp_path / "knowledge_graph.json"
    shutil.copy2(ROOT / "knowledge" / "knowledge_graph.graphml", graphml)
    shutil.copy2(ROOT / "knowledge" / "knowledge_graph.json", graph_json)
    return graphml, graph_json


def test_fault_injection_really_fails_repairs_and_reports_without_formal_pollution(tmp_path: Path) -> None:
    formal_before = (ROOT / "knowledge" / "knowledge_graph.graphml").read_bytes()
    graphml, graph_json = _temp_graph(tmp_path)
    state = WorkflowOrchestrator(
        output_root=tmp_path / "runs", graphml_path=graphml, graph_json_path=graph_json,
    ).run(REQUEST, persist=True, inject_failure="missing_imputer")
    logistic = next(result for result in state.candidate_results
                    if result.algorithm == "logistic_regression")
    assert state.status == "completed" and state.total_repair_attempts == 1
    assert [attempt.return_code for attempt in logistic.attempts] == [1, 0]
    assert "NaN" in logistic.attempts[0].stderr
    assert logistic.failure_type == "MissingValueError"
    assert logistic.repair_records[0].status == "validated"
    assert Path(logistic.repair_records[0].diff_path).is_file()
    assert "fixed_seed_stability" in {check.name for check in state.best_candidate.validation_checks}
    report = Path(state.final_report_paths["markdown"]).read_text(encoding="utf-8")
    assert "production-grade security sandbox" in report
    assert "MissingValueError" in report
    store = KnowledgeGraphStore.load_graphml(graphml)
    assert store.graph.nodes[state.run_id]["repaired"] is True
    assert any(data["relation"] == "FAILED_BECAUSE"
               for source, _, data in store.graph.edges(data=True) if source == state.run_id)
    assert (ROOT / "knowledge" / "knowledge_graph.graphml").read_bytes() == formal_before


class AlwaysMissingRunner:
    def run(self, artifact, data_path, *, timeout_seconds):
        return ExecutionResult(
            attempt=artifact.attempt, command=["<python>"], cwd="attempt",
            return_code=1, stderr="ValueError: Input X contains NaN. Use SimpleImputer.",
            result_json_path=artifact.result_output_path, model_path=artifact.model_output_path,
            process_status="failed",
        )


def test_maximum_two_repairs_is_enforced_and_other_candidate_continues(tmp_path: Path) -> None:
    graphml, graph_json = _temp_graph(tmp_path)
    workflow = WorkflowOrchestrator(
        output_root=tmp_path / "runs", graphml_path=graphml, graph_json_path=graph_json,
    )
    workflow.subprocess_runner = AlwaysMissingRunner()
    state = workflow.run(REQUEST, persist=False)
    assert state.total_repair_attempts == 4  # two bounded repairs per candidate
    assert all(len(result.attempts) == 3 for result in state.candidate_results)
    assert set(state.unrepaired_failures) == {"logistic_regression", "random_forest"}
    assert state.status == "failed"
