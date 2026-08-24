from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from agentforge.datasets.audit import (
    FEATURE_SCHEMES, _experiment, _split, duplicate_audit, feature_signatures,
    group_isolated_indices, ordinary_indices, run_uci_audit,
)
from agentforge.pipeline import _preprocessor, build_estimator
from sklearn.pipeline import Pipeline


def _frame(rows: int = 150) -> pd.DataFrame:
    records = []
    for index in range(rows):
        signature = index // 2
        records.append({
            "customer_id": f"ID-{index}", "usage": signature % 17,
            "status": f"status_{signature % 2}",
            "customer_value": float(signature % 23),
            "plan": f"plan_{signature % 3}", "churn": int(signature % 4 == 0),
        })
    return pd.DataFrame(records)


def test_feature_signature_excludes_id_and_target():
    frame = _frame()
    baseline = feature_signatures(frame)
    changed = frame.copy()
    changed["customer_id"] = [f"OTHER-{i}" for i in range(len(changed))]
    changed["churn"] = 1 - changed["churn"]
    assert baseline.equals(feature_signatures(changed))


def test_group_split_is_complete_disjoint_and_keeps_signatures_together():
    frame = _frame()
    indices = group_isolated_indices(frame)
    all_indices = np.concatenate(list(indices.values()))
    assert len(all_indices) == len(frame) == len(set(all_indices))
    signatures = feature_signatures(frame)
    sets = {name: set(signatures.loc[index]) for name, index in indices.items()}
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])


def test_three_feature_schemes_train_and_test_is_not_selection_input():
    frame = _frame()
    results = _experiment(frame, group_isolated_indices(frame), FEATURE_SCHEMES)
    assert set(results) == set(FEATURE_SCHEMES)
    for result in results.values():
        assert result["selection_basis"] == "validation_f1_only"
        assert result["best_candidate"] == max(
            result["candidates"], key=lambda item: item["validation_metrics"]["f1"]
        )["algorithm"]
        assert all(not item["test_used_for_selection"] for item in result["candidates"])


def test_customer_id_never_enters_features_or_transformer_and_cannot_affect_inputs():
    frame = _frame()
    indices = group_isolated_indices(frame)
    split = _split(frame, indices, ())
    assert "customer_id" not in split.x_train.columns
    transformer = _preprocessor(split.x_train)
    transformer.fit(split.x_train, split.y_train)
    columns = [column for _, _, selected in transformer.transformers_ for column in selected]
    assert "customer_id" not in columns
    changed = frame.copy()
    changed["customer_id"] = "changed"
    changed_split = _split(changed, indices, ())
    pd.testing.assert_frame_equal(split.x_test, changed_split.x_test)
    estimator, _ = build_estimator("logistic_regression")
    model = Pipeline([("preprocessor", _preprocessor(split.x_train)), ("model", estimator)])
    model.fit(split.x_train, split.y_train)
    np.testing.assert_array_equal(
        model.predict(split.x_test), model.predict(changed_split.x_test)
    )


def test_duplicate_audit_and_reports_are_generated(tmp_path: Path):
    frame = _frame()
    audit = duplicate_audit(frame, ordinary_indices(frame))
    assert audit["complete_duplicate_rows_beyond_first"] == 0
    assert audit["feature_duplicate_rows_beyond_first"] > 0
    dataset = tmp_path / "uci_iranian_churn.csv"
    frame.to_csv(dataset, index=False)
    result = run_uci_audit(dataset, tmp_path / "outputs")
    output = Path(result["output_dir"])
    assert {path.name for path in output.iterdir()} == {
        "duplicate_audit.json", "sensitivity_results.json", "sensitivity_report.md"
    }
    assert json.loads((output / "sensitivity_results.json").read_text())["random_state"] == 42
    assert "Test data was not used" in (output / "sensitivity_report.md").read_text()


def test_audit_does_not_modify_formal_knowledge_graph(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    graph_paths = [root / "knowledge/knowledge_graph.json", root / "knowledge/knowledge_graph.graphml"]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in graph_paths}
    dataset = tmp_path / "uci_iranian_churn.csv"
    _frame().to_csv(dataset, index=False)
    run_uci_audit(dataset, tmp_path / "outputs")
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in graph_paths}
    assert after == before
