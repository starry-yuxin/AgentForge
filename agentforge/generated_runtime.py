"""Trusted runtime used by deterministic generated candidate interfaces."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

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
    probabilities = bundle["pipeline"].predict_proba(features)[:, 1]
    return _final_metrics(labels, probabilities, bundle["threshold"], 0.0)
