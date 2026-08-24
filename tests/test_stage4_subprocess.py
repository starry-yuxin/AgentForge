from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agentforge.agents import CodeAgent, KnowledgeAgent, PlannerAgent, RequirementAgent
from agentforge.knowledge import KnowledgeGraphStore, KnowledgeRetriever
from agentforge.models import GeneratedArtifact
from agentforge.validation import SubprocessRunner


ROOT = Path(__file__).resolve().parents[1]
REQUEST = "Customer churn prediction using logistic regression with minimum F1 of 0.60"


def _artifact(tmp_path: Path) -> tuple[GeneratedArtifact, str]:
    requirement = RequirementAgent().parse(REQUEST)
    store = KnowledgeGraphStore.load_graphml(ROOT / "knowledge" / "knowledge_graph.graphml")
    knowledge = KnowledgeAgent(KnowledgeRetriever(store)).retrieve(requirement)
    plan = PlannerAgent().plan(requirement, knowledge)[0]
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


def test_nonzero_exit_and_missing_artifacts_are_recorded(tmp_path: Path) -> None:
    artifact, data = _artifact(tmp_path)
    path = Path(artifact.code_path)
    source = path.read_text(encoding="utf-8").replace(
        "return train_candidate(ALGORITHM, data_path, model_path)",
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
        "return train_candidate(ALGORITHM, data_path, model_path)",
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
