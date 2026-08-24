"""Shared deterministic model benchmark used by CLI and future UI entry points."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")
THRESHOLDS = tuple(float(value) for value in np.arange(0.15, 0.81, 0.025))
MODEL_DEFAULTS = {
    "logistic_regression": {
        "C": 1.0, "max_iter": 1_000, "class_weight": "balanced",
        "solver": "lbfgs", "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 240, "max_depth": 10, "min_samples_split": 2,
        "min_samples_leaf": 3, "max_features": "sqrt",
        "class_weight": "balanced_subsample", "random_state": 42, "n_jobs": 1,
    },
}


@dataclass(frozen=True)
class DataSplit:
    """Three disjoint, stratified partitions retaining source row indices."""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    random_state: int = 42,
    validation_size: float = 0.20,
    test_size: float = 0.20,
) -> DataSplit:
    """Create disjoint train/validation/test sets using two stratified splits."""
    if not 0.0 < validation_size < 1.0 or not 0.0 < test_size < 1.0:
        raise ValueError("validation_size and test_size must be between 0 and 1")
    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size and test_size must sum to less than 1")

    x_development, x_test, y_development, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_development,
        y_development,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=y_development,
    )
    return DataSplit(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def select_threshold(
    validation_labels: pd.Series | np.ndarray,
    validation_probabilities: np.ndarray,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> tuple[float, float]:
    """Select a threshold exclusively from validation labels and probabilities."""
    if len(validation_labels) != len(validation_probabilities):
        raise ValueError("validation labels and probabilities must have equal length")
    if not thresholds:
        raise ValueError("at least one threshold is required")

    scored = []
    for threshold in thresholds:
        predictions = (validation_probabilities >= threshold).astype(int)
        score = f1_score(validation_labels, predictions, zero_division=0)
        scored.append((float(score), -abs(threshold - 0.5), threshold))
    best_score, _, best_threshold = max(scored)
    return float(best_threshold), float(best_score)


def _preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns)]
    )


def _plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _plain_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_hyperparameters(algorithm: str, requested: dict[str, Any] | None) -> dict[str, Any]:
    """Validate a small, JSON-safe estimator parameter surface and apply trusted defaults."""
    if algorithm not in MODEL_DEFAULTS:
        raise ValueError(f"unsupported registered algorithm: {algorithm}")
    if requested is None:
        requested = {}
    if not isinstance(requested, dict):
        raise TypeError("hyperparameters must be a dictionary")
    try:
        json.dumps(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError("hyperparameters must be JSON serializable") from exc
    unknown = sorted(set(requested) - set(MODEL_DEFAULTS[algorithm]))
    if unknown:
        raise ValueError(f"unsupported hyperparameter(s) for {algorithm}: {unknown}")
    effective = {**MODEL_DEFAULTS[algorithm], **requested}
    if effective["random_state"] != 42 or not _plain_int(effective["random_state"]):
        raise ValueError("random_state must be the fixed integer 42")
    if algorithm == "logistic_regression":
        if not _plain_number(effective["C"]) or not 0.0 < float(effective["C"]) <= 1_000.0:
            raise ValueError("C must be a number in (0, 1000]")
        if not _plain_int(effective["max_iter"]) or not 1 <= effective["max_iter"] <= 10_000:
            raise ValueError("max_iter must be an integer in [1, 10000]")
        if effective["class_weight"] not in {None, "balanced"}:
            raise ValueError("class_weight must be null or 'balanced'")
        if effective["solver"] not in {"lbfgs", "liblinear"}:
            raise ValueError("solver must be 'lbfgs' or 'liblinear'")
    else:
        if not _plain_int(effective["n_estimators"]) or not 1 <= effective["n_estimators"] <= 2_000:
            raise ValueError("n_estimators must be an integer in [1, 2000]")
        if effective["max_depth"] is not None and (
            not _plain_int(effective["max_depth"]) or not 1 <= effective["max_depth"] <= 100
        ):
            raise ValueError("max_depth must be null or an integer in [1, 100]")
        if not _plain_int(effective["min_samples_split"]) or not 2 <= effective["min_samples_split"] <= 100:
            raise ValueError("min_samples_split must be an integer in [2, 100]")
        if not _plain_int(effective["min_samples_leaf"]) or not 1 <= effective["min_samples_leaf"] <= 100:
            raise ValueError("min_samples_leaf must be an integer in [1, 100]")
        if effective["max_features"] not in {None, "sqrt", "log2"}:
            raise ValueError("max_features must be null, 'sqrt', or 'log2'")
        if effective["class_weight"] not in {None, "balanced", "balanced_subsample"}:
            raise ValueError("invalid random_forest class_weight")
        if effective["n_jobs"] != 1 or not _plain_int(effective["n_jobs"]):
            raise ValueError("n_jobs must be the fixed integer 1")
    return effective


def build_estimator(algorithm: str, hyperparameters: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    effective = validate_hyperparameters(algorithm, hyperparameters)
    estimator = (
        LogisticRegression(**effective)
        if algorithm == "logistic_regression"
        else RandomForestClassifier(**effective)
    )
    return estimator, effective


def _models(random_state: int) -> dict[str, Any]:
    if random_state != 42:
        raise ValueError("random_state must remain fixed at 42")
    return {
        algorithm: build_estimator(algorithm)[0]
        for algorithm in MODEL_DEFAULTS
    }


def _final_metrics(
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    runtime_seconds: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "runtime_seconds": runtime_seconds,
        "confusion_matrix": matrix.tolist(),
    }
    numeric_values = [metrics[name] for name in (*METRIC_NAMES, "runtime_seconds")]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("model produced a non-finite metric")
    return metrics


def _validation_metrics(
    labels: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Return validation metrics without a runtime field or test-set access."""
    metrics = _final_metrics(labels, probabilities, threshold, 0.0)
    return {name: float(metrics[name]) for name in METRIC_NAMES}


def _evaluate_model(name: str, estimator: Any, split: DataSplit) -> dict[str, Any]:
    pipeline = Pipeline(
        [("preprocessor", _preprocessor(split.x_train)), ("model", estimator)]
    )
    started = time.perf_counter()
    pipeline.fit(split.x_train, split.y_train)
    validation_probabilities = pipeline.predict_proba(split.x_validation)[:, 1]
    selected_threshold, validation_f1 = select_threshold(
        split.y_validation, validation_probabilities
    )
    # Test data is touched only after model fitting and threshold selection are complete.
    test_probabilities = pipeline.predict_proba(split.x_test)[:, 1]
    runtime = time.perf_counter() - started
    return {
        "algorithm": name,
        "selected_threshold": selected_threshold,
        "validation_f1": validation_f1,
        "validation_metrics": _validation_metrics(
            split.y_validation, validation_probabilities, selected_threshold
        ),
        "metrics": _final_metrics(
            split.y_test, test_probabilities, selected_threshold, runtime
        ),
    }


def evaluate_candidate(
    data_path: str | Path,
    algorithm: str,
    *,
    random_state: int = 42,
    validation_size: float = 0.20,
    test_size: float = 0.20,
) -> dict[str, Any]:
    """Independently execute one registered candidate on fresh train/validation/test splits."""
    models = _models(random_state)
    if algorithm not in models:
        raise ValueError(f"unsupported registered algorithm: {algorithm}")
    frame = pd.read_csv(Path(data_path))
    required = {TARGET_COLUMN, ID_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    features = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    target = frame[TARGET_COLUMN].astype(int)
    split = split_dataset(
        features, target, random_state=random_state,
        validation_size=validation_size, test_size=test_size,
    )
    result = _evaluate_model(algorithm, models[algorithm], split)
    result.update({
        "rows": len(frame),
        "positive_rate": float(target.mean()),
        "train_size": len(split.y_train),
        "validation_size": len(split.y_validation),
        "test_size": len(split.y_test),
        "random_state": random_state,
    })
    return result


def run_benchmark(
    data_path: str | Path,
    output_path: str | Path,
    *,
    primary_metric: str = "f1",
    random_state: int = 42,
    validation_size: float = 0.20,
    test_size: float = 0.20,
) -> dict[str, Any]:
    """Train candidates, select on validation F1, and evaluate once on test."""
    if primary_metric != "f1":
        raise ValueError("threshold-aware candidate selection currently supports only f1")

    source = Path(data_path)
    frame = pd.read_csv(source)
    required = {TARGET_COLUMN, ID_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    if frame[TARGET_COLUMN].nunique() != 2:
        raise ValueError("churn target must contain exactly two classes")

    features = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    target = frame[TARGET_COLUMN].astype(int)
    split = split_dataset(
        features,
        target,
        random_state=random_state,
        validation_size=validation_size,
        test_size=test_size,
    )
    candidates = [
        _evaluate_model(name, model, split) for name, model in _models(random_state).items()
    ]
    # Candidate selection is based on validation F1, never final test metrics.
    best = max(candidates, key=lambda result: result["validation_f1"])
    result = {
        "status": "success",
        "data_path": str(source),
        "rows": len(frame),
        "positive_rate": float(target.mean()),
        "train_size": len(split.y_train),
        "validation_size": len(split.y_validation),
        "test_size": len(split.y_test),
        "primary_metric": primary_metric,
        "selection_split": "validation",
        "candidates": candidates,
        "best_algorithm": best["algorithm"],
        "best_score": best["metrics"][primary_metric],
        "best_validation_score": best["validation_f1"],
        "random_state": random_state,
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
