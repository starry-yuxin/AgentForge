"""Trusted runtime used by deterministic generated candidate interfaces."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agentforge.pipeline import (
    ID_COLUMN,
    TARGET_COLUMN,
    _final_metrics,
    _models,
    _preprocessor,
    select_threshold,
    split_dataset,
)
from sklearn.pipeline import Pipeline


def train_candidate(algorithm: str, data_path: str, model_path: str) -> dict:
    frame = pd.read_csv(data_path)
    features = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    target = frame[TARGET_COLUMN].astype(int)
    split = split_dataset(features, target)
    estimator = _models(42).get(algorithm)
    if estimator is None:
        raise ValueError(f"unsupported registered algorithm: {algorithm}")
    pipeline = Pipeline([
        ("preprocessor", _preprocessor(split.x_train)), ("model", estimator)
    ])
    pipeline.fit(split.x_train, split.y_train)
    probabilities = pipeline.predict_proba(split.x_validation)[:, 1]
    threshold, validation_f1 = select_threshold(split.y_validation, probabilities)
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        pickle.dump({"pipeline": pipeline, "threshold": threshold, "algorithm": algorithm}, stream)
    return {"algorithm": algorithm, "selected_threshold": threshold,
            "validation_f1": validation_f1, "model_path": str(destination)}


def train_candidate_missing_imputer(algorithm: str, data_path: str, model_path: str) -> dict:
    """Intentional stage-four fault: fit without imputers so real NaN handling fails."""
    frame = pd.read_csv(data_path)
    features = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    target = frame[TARGET_COLUMN].astype(int)
    split = split_dataset(features, target)
    numeric_columns = split.x_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = split.x_train.select_dtypes(exclude="number").columns.tolist()
    unsafe_preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_columns),
    ])
    estimator = _models(42).get(algorithm)
    pipeline = Pipeline([("preprocessor", unsafe_preprocessor), ("model", estimator)])
    pipeline.fit(split.x_train, split.y_train)
    raise AssertionError("fault injection unexpectedly completed without a missing-value error")


def _load_bundle(model_path: str) -> dict:
    with Path(model_path).open("rb") as stream:
        return pickle.load(stream)  # noqa: S301 - only project-created trusted artifacts are allowed.


def predict_candidate(model_path: str, data_path: str) -> list:
    bundle = _load_bundle(model_path)
    frame = pd.read_csv(data_path)
    features = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    probabilities = bundle["pipeline"].predict_proba(features)[:, 1]
    return (probabilities >= bundle["threshold"]).astype(int).tolist()


def evaluate_candidate_model(model_path: str, data_path: str) -> dict:
    bundle = _load_bundle(model_path)
    frame = pd.read_csv(data_path)
    labels = frame[TARGET_COLUMN].astype(int)
    features = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    split = split_dataset(features, labels)
    validation_probabilities = bundle["pipeline"].predict_proba(split.x_validation)[:, 1]
    test_probabilities = bundle["pipeline"].predict_proba(split.x_test)[:, 1]
    validation = _final_metrics(
        split.y_validation, validation_probabilities, bundle["threshold"], 0.0
    )
    test = _final_metrics(split.y_test, test_probabilities, bundle["threshold"], 0.0)
    return {
        "algorithm": bundle["algorithm"],
        "selected_threshold": bundle["threshold"],
        "validation_metrics": validation,
        "test_metrics": test,
        "split_sizes": {
            "train": len(split.y_train), "validation": len(split.y_validation),
            "test": len(split.y_test),
        },
    }
