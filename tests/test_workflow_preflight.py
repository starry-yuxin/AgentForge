from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agentforge.agents import RequirementAgent
from agentforge.workflow import WorkflowOrchestrator
from agentforge.workflow.preflight import DatasetNotFound, validate_dataset


REQUEST = "请构建客户流失预测模型，以F1作为主要指标。"


def test_preflight_accepts_regular_csv_and_relative_path(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "input.csv"
    pd.DataFrame({"feature": [1], "churn": [0]}).to_csv(dataset, index=False)
    requirement = RequirementAgent().parse(REQUEST).model_copy(
        update={"dataset_path": str(dataset)}
    )
    assert validate_dataset(requirement) == dataset

    monkeypatch.chdir(tmp_path)
    relative = requirement.model_copy(update={"dataset_path": "input.csv"})
    assert validate_dataset(relative) == Path("input.csv")


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_preflight_rejects_unavailable_dataset_without_parent_path(tmp_path: Path, kind: str):
    path = tmp_path / ("missing.csv" if kind == "missing" else "dataset-dir")
    if kind == "directory":
        path.mkdir()
    requirement = RequirementAgent().parse(REQUEST).model_copy(
        update={"dataset_path": str(path)}
    )
    with pytest.raises(DatasetNotFound) as captured:
        validate_dataset(requirement)
    assert path.name in str(captured.value)
    assert str(path.parent) not in str(captured.value)


def test_missing_dataset_fails_before_knowledge_planning_generation_or_repair(tmp_path: Path):
    missing = tmp_path / "absent.csv"
    state = WorkflowOrchestrator(output_root=tmp_path / "runs").run(
        REQUEST, overrides={"dataset_path": str(missing)}, persist=False,
    )

    assert state.status == "failed"
    assert state.llm_call_count == 0
    assert state.retrieved_knowledge is None
    assert state.candidate_plans == []
    assert state.generated_artifacts == []
    assert state.candidate_results == []
    assert state.total_repair_attempts == 0
    assert any(error.startswith("DatasetNotFound:") for error in state.errors)
    failed = [event for event in state.events if event.agent_name == "DatasetPreflight"]
    assert [event.status for event in failed] == ["started", "failed"]
    assert not list((tmp_path / "runs").rglob("candidate.py"))
