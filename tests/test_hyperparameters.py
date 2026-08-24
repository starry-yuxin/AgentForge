from __future__ import annotations

import pytest

from agentforge.pipeline import build_estimator, validate_hyperparameters


@pytest.mark.parametrize(("algorithm", "parameters", "message"), [
    ("logistic_regression", {"unknown_parameter": 1}, "unsupported hyperparameter"),
    ("logistic_regression", {"max_iter": "1000"}, "max_iter"),
    ("logistic_regression", {"C": 0}, "C must"),
    ("logistic_regression", {"random_state": 7}, "random_state"),
    ("random_forest", {"n_estimators": 0}, "n_estimators"),
    ("random_forest", {"n_estimators": True}, "n_estimators"),
    ("random_forest", {"n_jobs": 2}, "n_jobs"),
    ("random_forest", {"max_depth": -1}, "max_depth"),
])
def test_invalid_hyperparameters_are_rejected(
    algorithm: str, parameters: dict, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_hyperparameters(algorithm, parameters)


def test_effective_parameters_include_trusted_defaults() -> None:
    estimator, effective = build_estimator("logistic_regression", {"C": 0.5})
    assert estimator.get_params()["C"] == 0.5
    assert effective == {
        "C": 0.5, "max_iter": 1000, "class_weight": "balanced",
        "solver": "lbfgs", "random_state": 42,
    }


def test_non_json_serializable_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        validate_hyperparameters("logistic_regression", {"C": object()})
