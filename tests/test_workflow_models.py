from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentforge.models import AlgorithmRequirement


def _requirement(**changes) -> AlgorithmRequirement:
    values = {
        "request_id": "req-test", "raw_text": "customer churn prediction",
        "task_type": "binary_classification", "industry": "customer_churn",
        "dataset_path": "data/churn_sample.csv", "target_column": "churn",
        "primary_metric": "f1", "minimum_score": 0.6,
        "candidate_algorithms": ["logistic_regression", "random_forest"],
        "field_sources": {},
    }
    values.update(changes)
    return AlgorithmRequirement(**values)


def test_algorithm_requirement_is_json_serializable_and_has_default_interfaces() -> None:
    requirement = _requirement()
    assert requirement.required_interfaces == ["train", "predict", "evaluate"]
    assert '"task_type":"binary_classification"' in requirement.model_dump_json()


@pytest.mark.parametrize("task", ["", "text_classification"])
def test_invalid_task_type_is_rejected(task: str) -> None:
    with pytest.raises(ValidationError, match="task_type"):
        _requirement(task_type=task)


def test_invalid_primary_metric_and_empty_candidates_are_rejected() -> None:
    with pytest.raises(ValidationError, match="primary_metric"):
        _requirement(primary_metric="profit")
    with pytest.raises(ValidationError, match="candidate_algorithms"):
        _requirement(candidate_algorithms=[])


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_minimum_score_range_is_validated(score: float) -> None:
    with pytest.raises(ValidationError, match="minimum_score"):
        _requirement(minimum_score=score)
