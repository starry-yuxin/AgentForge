import json
from pathlib import Path

import numpy as np
import pandas as pd

from agentforge.data import generate_churn_data
from agentforge.pipeline import METRIC_NAMES, run_benchmark, select_threshold, split_dataset


def test_benchmark_runs_both_models_and_writes_json(tmp_path: Path) -> None:
    data_path = tmp_path / "churn.csv"
    output_path = tmp_path / "result.json"
    generate_churn_data(data_path, rows=700, random_state=42)
    result = run_benchmark(data_path, output_path)

    assert result["status"] == "success"
    assert result["primary_metric"] == "f1"
    assert result["selection_split"] == "validation"
    assert result["train_size"] + result["validation_size"] + result["test_size"] == 700
    assert {item["algorithm"] for item in result["candidates"]} == {
        "logistic_regression",
        "random_forest",
    }
    for candidate in result["candidates"]:
        metrics = candidate["metrics"]
        assert all(0.0 <= metrics[name] <= 1.0 for name in METRIC_NAMES)
        assert metrics["runtime_seconds"] >= 0.0
        assert metrics["roc_auc"] >= 0.65
        assert 0.0 < candidate["selected_threshold"] < 1.0
        assert len(metrics["confusion_matrix"]) == 2
    validation_scores = {
        item["algorithm"]: item["validation_f1"] for item in result["candidates"]
    }
    assert result["best_algorithm"] == max(validation_scores, key=validation_scores.get)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["best_algorithm"] == result["best_algorithm"]


def test_threshold_selection_uses_only_validation_inputs() -> None:
    validation_labels = pd.Series([0, 0, 0, 1, 1, 1])
    validation_probabilities = np.array([0.10, 0.20, 0.45, 0.40, 0.60, 0.90])
    threshold, score = select_threshold(
        validation_labels, validation_probabilities, thresholds=(0.3, 0.5, 0.7)
    )
    assert threshold == 0.3
    assert score == 6 / 7

    # Changing unrelated would-be test labels cannot influence this pure function.
    unrelated_test_labels = pd.Series([1, 1, 0, 0])
    unrelated_test_labels[:] = 1 - unrelated_test_labels
    repeated = select_threshold(
        validation_labels, validation_probabilities, thresholds=(0.3, 0.5, 0.7)
    )
    assert repeated == (threshold, score)


def test_stratified_splits_are_disjoint_and_complete(tmp_path: Path) -> None:
    frame = generate_churn_data(tmp_path / "churn.csv", rows=1_000, random_state=42)
    features = frame.drop(columns=["customer_id", "churn"])
    target = frame["churn"]
    split = split_dataset(features, target, random_state=42)

    train_indices = set(split.x_train.index)
    validation_indices = set(split.x_validation.index)
    test_indices = set(split.x_test.index)
    assert train_indices.isdisjoint(validation_indices)
    assert train_indices.isdisjoint(test_indices)
    assert validation_indices.isdisjoint(test_indices)
    assert train_indices | validation_indices | test_indices == set(frame.index)
    assert (len(train_indices), len(validation_indices), len(test_indices)) == (600, 200, 200)
    overall_rate = target.mean()
    for labels in (split.y_train, split.y_validation, split.y_test):
        assert abs(labels.mean() - overall_rate) < 0.01


def test_benchmark_rejects_unknown_metric(tmp_path: Path) -> None:
    data_path = tmp_path / "churn.csv"
    generate_churn_data(data_path, rows=200)
    try:
        run_benchmark(data_path, tmp_path / "result.json", primary_metric="made_up")
    except ValueError as error:
        assert "supports only f1" in str(error)
    else:
        raise AssertionError("expected ValueError")

