from pathlib import Path

import pandas as pd

from agentforge.data import FEATURE_COLUMNS, generate_churn_data


def test_generate_churn_data_is_reproducible_and_learnable(tmp_path: Path) -> None:
    first = generate_churn_data(tmp_path / "first.csv", rows=600, random_state=42)
    second = generate_churn_data(tmp_path / "second.csv", rows=600, random_state=42)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [*FEATURE_COLUMNS, "churn"]
    assert first.isna().sum().sum() > 0
    assert 0.10 < first["churn"].mean() < 0.50
    assert first["churn"].nunique() == 2


def test_generate_churn_data_rejects_tiny_dataset(tmp_path: Path) -> None:
    try:
        generate_churn_data(tmp_path / "tiny.csv", rows=20)
    except ValueError as error:
        assert "at least 100" in str(error)
    else:
        raise AssertionError("expected ValueError")

