"""Offline duplicate-group and feature sensitivity audit for UCI churn data."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from agentforge.pipeline import (
    ID_COLUMN, TARGET_COLUMN, DataSplit, _final_metrics, _preprocessor,
    _validation_metrics, build_estimator, select_threshold, split_dataset,
)

ALGORITHMS = ("logistic_regression", "random_forest")
FEATURE_SCHEMES = {
    "full_features": (),
    "without_status": ("status",),
    "without_status_and_customer_value": ("status", "customer_value"),
}


def feature_signatures(frame: pd.DataFrame) -> pd.Series:
    """Hash only predictive values, explicitly excluding identifier and target."""
    columns = [column for column in frame.columns if column not in {ID_COLUMN, TARGET_COLUMN}]
    if not columns:
        raise ValueError("no predictive columns available for feature signatures")

    def signature(row: pd.Series) -> str:
        values = [None if pd.isna(value) else value.item() if hasattr(value, "item") else value
                  for value in row.tolist()]
        payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return frame[columns].apply(signature, axis=1).rename("feature_signature")


def group_isolated_indices(frame: pd.DataFrame, *, random_state: int = 42) -> dict[str, np.ndarray]:
    """Assign each feature-signature group to one stratified fold (60/20/20 target)."""
    signatures = feature_signatures(frame)
    target = frame[TARGET_COLUMN].astype(int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    folds: list[np.ndarray] = []
    placeholder = np.zeros((len(frame), 1))
    for _, held_out in splitter.split(placeholder, target, groups=signatures):
        folds.append(np.asarray(held_out, dtype=int))
    return {
        "train": np.sort(np.concatenate(folds[2:])),
        "validation": np.sort(folds[1]),
        "test": np.sort(folds[0]),
    }


def ordinary_indices(frame: pd.DataFrame, *, random_state: int = 42) -> dict[str, np.ndarray]:
    features = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    target = frame[TARGET_COLUMN].astype(int)
    split = split_dataset(features, target, random_state=random_state)
    return {
        "train": split.x_train.index.to_numpy(),
        "validation": split.x_validation.index.to_numpy(),
        "test": split.x_test.index.to_numpy(),
    }


def _split(frame: pd.DataFrame, indices: dict[str, np.ndarray], dropped: tuple[str, ...]) -> DataSplit:
    features = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN, *dropped])
    target = frame[TARGET_COLUMN].astype(int)
    return DataSplit(
        x_train=features.loc[indices["train"]],
        x_validation=features.loc[indices["validation"]],
        x_test=features.loc[indices["test"]],
        y_train=target.loc[indices["train"]],
        y_validation=target.loc[indices["validation"]],
        y_test=target.loc[indices["test"]],
    )


def _evaluate(algorithm: str, split: DataSplit) -> dict[str, Any]:
    estimator, _ = build_estimator(algorithm)
    pipeline = Pipeline([("preprocessor", _preprocessor(split.x_train)), ("model", estimator)])
    started = time.perf_counter()
    pipeline.fit(split.x_train, split.y_train)
    validation_probabilities = pipeline.predict_proba(split.x_validation)[:, 1]
    threshold, _ = select_threshold(split.y_validation, validation_probabilities)
    validation = _validation_metrics(split.y_validation, validation_probabilities, threshold)
    # Test is accessed only after threshold selection has completed.
    test_probabilities = pipeline.predict_proba(split.x_test)[:, 1]
    test = _final_metrics(split.y_test, test_probabilities, threshold, time.perf_counter() - started)
    transformer_columns = [column for _, _, columns in pipeline.named_steps[
        "preprocessor"
    ].transformers_ for column in columns]
    return {
        "algorithm": algorithm, "selected_threshold": threshold,
        "validation_metrics": validation, "test_metrics": test,
        "selection_basis": "validation_f1_only", "test_used_for_selection": False,
        "model_feature_columns": split.x_train.columns.tolist(),
        "column_transformer_columns": transformer_columns,
        "customer_id_excluded": ID_COLUMN not in transformer_columns,
    }


def duplicate_audit(frame: pd.DataFrame, ordinary: dict[str, np.ndarray]) -> dict[str, Any]:
    signatures = feature_signatures(frame)
    without_id = frame.drop(columns=[ID_COLUMN])
    signature_sizes = signatures.value_counts()
    duplicate_signatures = set(signature_sizes[signature_sizes > 1].index)
    conflicting = []
    for signature, group in frame.assign(feature_signature=signatures).groupby("feature_signature"):
        if group[TARGET_COLUMN].nunique() > 1:
            conflicting.append({
                "feature_signature": signature, "size": len(group),
                "labels": sorted(group[TARGET_COLUMN].astype(int).unique().tolist()),
            })
    split_signatures = {
        name: set(signatures.loc[index].tolist()) for name, index in ordinary.items()
    }
    crossings = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = split_signatures[left] & split_signatures[right]
        left_count = int(signatures.loc[ordinary[left]].isin(shared).sum())
        right_count = int(signatures.loc[ordinary[right]].isin(shared).sum())
        crossings[f"{left}_to_{right}"] = {
            "group_count": len(shared),
            "sample_count": left_count + right_count,
            "all_dataset_samples_with_shared_signatures": int(signatures.isin(shared).sum()),
            f"{left}_sample_count": left_count,
            f"{right}_sample_count": right_count,
        }
    return {
        "definitions": {
            "complete_row": "all processed columns, including customer_id and churn",
            "without_customer_id": "all official fields including churn",
            "feature_signature": "SHA-256 of ordered predictors excluding customer_id and churn",
        },
        "row_count": len(frame),
        "complete_duplicate_rows_beyond_first": int(frame.duplicated().sum()),
        "duplicates_without_customer_id_beyond_first": int(without_id.duplicated().sum()),
        "feature_duplicate_rows_beyond_first": int(signatures.duplicated().sum()),
        "feature_duplicate_group_count": len(duplicate_signatures),
        "feature_duplicate_involved_samples": int(signatures.isin(duplicate_signatures).sum()),
        "duplicate_group_size_distribution": {
            str(size): count for size, count in sorted(Counter(
                signature_sizes[signature_sizes > 1].astype(int).tolist()
            ).items())
        },
        "conflicting_label_group_count": len(conflicting),
        "conflicting_label_involved_samples": sum(item["size"] for item in conflicting),
        "conflicting_label_groups": conflicting,
        "ordinary_split_crossings": crossings,
    }


def _split_summary(frame: pd.DataFrame, indices: dict[str, np.ndarray]) -> dict[str, Any]:
    signatures = feature_signatures(frame)
    result = {}
    for name, index in indices.items():
        labels = frame.loc[index, TARGET_COLUMN].astype(int)
        result[name] = {
            "rows": len(index), "ratio": len(index) / len(frame),
            "positive_rate": float(labels.mean()), "signature_count": int(signatures.loc[index].nunique()),
        }
    result["signature_sets_disjoint"] = all(
        set(signatures.loc[indices[left]]).isdisjoint(set(signatures.loc[indices[right]]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    result["all_samples_assigned_once"] = len(set(np.concatenate(list(indices.values())))) == len(frame)
    return result


def _experiment(frame: pd.DataFrame, indices: dict[str, np.ndarray], schemes: dict[str, tuple[str, ...]]) -> dict:
    results = {}
    for scheme, dropped in schemes.items():
        split = _split(frame, indices, dropped)
        candidates = [_evaluate(algorithm, split) for algorithm in ALGORITHMS]
        best = max(candidates, key=lambda item: item["validation_metrics"]["f1"])
        results[scheme] = {
            "dropped_features": list(dropped), "candidates": candidates,
            "best_candidate": best["algorithm"], "selection_basis": "validation_f1_only",
        }
    return results


def run_uci_audit(data_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    frame = pd.read_csv(data_path)
    required = {ID_COLUMN, TARGET_COLUMN, "status", "customer_value"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"audit dataset missing columns: {sorted(missing)}")
    audit_id = f"audit-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    output = Path(output_root) / audit_id
    output.mkdir(parents=True, exist_ok=False)
    ordinary = ordinary_indices(frame)
    isolated = group_isolated_indices(frame)
    duplicates = duplicate_audit(frame, ordinary)
    sensitivity = {
        "audit_id": audit_id, "random_state": 42,
        "dataset_file": Path(data_path).name,
        "ordinary_split": _split_summary(frame, ordinary),
        "group_isolated_split": _split_summary(frame, isolated),
        "ordinary_full_features": _experiment(frame, ordinary, {"full_features": ()})["full_features"],
        "group_isolated_feature_sensitivity": _experiment(frame, isolated, FEATURE_SCHEMES),
        "interpretation": (
            "Sensitivity analysis only. Official Status and Customer Value fields are potential "
            "proxy/derived-variable risks, not established leakage fields. The main workflow remains unchanged."
        ),
    }
    (output / "duplicate_audit.json").write_text(
        json.dumps(duplicates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "sensitivity_results.json").write_text(
        json.dumps(sensitivity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# UCI churn sensitivity audit", "", f"- Audit ID: `{audit_id}`",
        "- Random state: `42`", "- LLM calls: `0`", "- Knowledge persistence: `false`", "",
        "## Duplicate audit", "",
        f"- Duplicate feature groups: `{duplicates['feature_duplicate_group_count']}`",
        f"- Samples in duplicate feature groups: `{duplicates['feature_duplicate_involved_samples']}`",
        f"- Conflicting-label groups: `{duplicates['conflicting_label_group_count']}`",
        f"- Ordinary split crossings: `{duplicates['ordinary_split_crossings']}`", "",
        "## Split summary", "", f"- Ordinary: `{sensitivity['ordinary_split']}`",
        f"- Group isolated: `{sensitivity['group_isolated_split']}`", "",
        "## Results", "", "| Split / feature scheme | Algorithm | Validation F1 | Test F1 | Test ROC-AUC | Threshold |",
        "|---|---|---:|---:|---:|---:|",
    ]
    rows = [("ordinary / full_features", sensitivity["ordinary_full_features"])]
    rows.extend((f"group_isolated / {name}", value)
                for name, value in sensitivity["group_isolated_feature_sensitivity"].items())
    for label, result in rows:
        for candidate in result["candidates"]:
            lines.append(
                f"| {label} | {candidate['algorithm']} | "
                f"{candidate['validation_metrics']['f1']:.6f} | "
                f"{candidate['test_metrics']['f1']:.6f} | "
                f"{candidate['test_metrics']['roc_auc']:.6f} | "
                f"{candidate['selected_threshold']:.3f} |"
            )
    lines.extend(["", "Test data was not used for threshold or candidate selection.", "",
                  sensitivity["interpretation"]])
    (output / "sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"audit_id": audit_id, "output_dir": str(output),
            "duplicate_audit": duplicates, "sensitivity": sensitivity}
