from __future__ import annotations

import subprocess
import json
import pickle
from pathlib import Path

import pytest

from agentforge.agents import CodeAgent, KnowledgeAgent, PlannerAgent, RequirementAgent
from agentforge.knowledge import KnowledgeGraphStore, KnowledgeRetriever
from agentforge.models import GeneratedArtifact
from agentforge.validation import SubprocessRunner


ROOT = Path(__file__).resolve().parents[1]
REQUEST = "Customer churn prediction using logistic regression with minimum F1 of 0.60"


def _artifact(
    tmp_path: Path, algorithm: str = "logistic_regression", hyperparameters: dict | None = None,
) -> tuple[GeneratedArtifact, str]:
    requirement = RequirementAgent().parse(
        REQUEST, {"candidate_algorithms": [algorithm]}
    )
    store = KnowledgeGraphStore.load_graphml(ROOT / "knowledge" / "knowledge_graph.graphml")
    knowledge = KnowledgeAgent(KnowledgeRetriever(store)).retrieve(requirement)
    plan = PlannerAgent().plan(requirement, knowledge)[0]
    if hyperparameters is not None:
        plan = plan.model_copy(update={"hyperparameters": hyperparameters})
    return CodeAgent().generate(plan, "run-test", tmp_path), requirement.dataset_path


def test_normal_candidate_executes_and_captures_logs(tmp_path: Path) -> None:
    artifact, data = _artifact(tmp_path)
    result = SubprocessRunner().run(artifact, data, timeout_seconds=20)
    assert result.process_status == "completed" and result.return_code == 0
    assert "success" in result.stdout
    assert Path(result.result_json_path).is_file()
    assert Path(result.model_path).is_file()
    assert (Path(artifact.code_path).parent / "stdout.log").is_file()
    assert result.command[0] == "<venv-python>"
    assert not any(str(Path.home()) in item for item in result.command)


def test_generated_train_passes_hyperparameters_to_runtime(tmp_path: Path) -> None:
    artifact, _ = _artifact(tmp_path)
    source = Path(artifact.code_path).read_text(encoding="utf-8")
    assert "train_candidate(ALGORITHM, data_path, model_path, HYPERPARAMETERS)" in source


@pytest.mark.parametrize(("algorithm", "requested", "parameter", "expected"), [
    ("logistic_regression", {"C": 0.25, "max_iter": 300, "class_weight": "balanced",
                             "solver": "lbfgs", "random_state": 42}, "C", 0.25),
    ("random_forest", {"n_estimators": 17, "max_depth": 6, "min_samples_split": 2,
                       "min_samples_leaf": 2, "max_features": "sqrt",
                       "class_weight": "balanced_subsample", "random_state": 42,
                       "n_jobs": 1}, "n_estimators", 17),
])
def test_whitelisted_parameters_reach_saved_estimator(
    tmp_path: Path, algorithm: str, requested: dict, parameter: str, expected,
) -> None:
    artifact, data = _artifact(tmp_path, algorithm, requested)
    result = SubprocessRunner().run(artifact, data, timeout_seconds=20)
    assert result.process_status == "completed"
    with Path(result.model_path).open("rb") as stream:
        bundle = pickle.load(stream)
    estimator = bundle["pipeline"].named_steps["model"]
    assert estimator.get_params()[parameter] == expected
    assert bundle["requested_hyperparameters"] == requested
    assert bundle["effective_hyperparameters"][parameter] == expected
    payload = json.loads(Path(result.result_json_path).read_text(encoding="utf-8"))
    assert payload["train_result"]["requested_hyperparameters"] == requested
    assert payload["evaluation"]["effective_hyperparameters"][parameter] == expected


def test_nonzero_exit_and_missing_artifacts_are_recorded(tmp_path: Path) -> None:
    artifact, data = _artifact(tmp_path)
    path = Path(artifact.code_path)
    source = path.read_text(encoding="utf-8").replace(
        "return train_candidate(ALGORITHM, data_path, model_path, HYPERPARAMETERS)",
        "raise RuntimeError('real child failure')",
    )
    path.write_text(source, encoding="utf-8")
    result = SubprocessRunner().run(artifact, data, timeout_seconds=20)
    assert result.return_code != 0 and result.process_status == "failed"
    assert "real child failure" in result.stderr
    assert not Path(result.result_json_path).exists()
    assert not Path(result.model_path).exists()


def test_timeout_is_recorded_without_hanging(tmp_path: Path) -> None:
    artifact, data = _artifact(tmp_path)
    path = Path(artifact.code_path)
    source = path.read_text(encoding="utf-8").replace(
        "return train_candidate(ALGORITHM, data_path, model_path, HYPERPARAMETERS)",
        "import time\n    time.sleep(5)\n    return {}",
    )
    path.write_text(source, encoding="utf-8")
    result = SubprocessRunner().run(artifact, data, timeout_seconds=0.1)
    assert result.timed_out and result.process_status == "timed_out"
    assert result.return_code is None


def test_runner_uses_shell_false_and_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact, data = _artifact(tmp_path)
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 1, stdout="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = SubprocessRunner().run(artifact, data, timeout_seconds=3.5)
    assert observed["shell"] is False
    assert observed["timeout"] == 3.5
    assert observed["capture_output"] is True
    assert result.process_status == "failed"
