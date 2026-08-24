"""Deterministic sample data generation for the churn demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "customer_id",
    "age",
    "tenure_months",
    "monthly_charge",
    "total_charge",
    "contract_type",
    "payment_method",
    "internet_service",
    "support_calls",
    "late_payments",
]


def generate_churn_data(
    output_path: str | Path,
    *,
    rows: int = 1_200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a noisy, imbalanced dataset with learnable business patterns."""
    if rows < 100:
        raise ValueError("rows must be at least 100")

    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 81, rows)
    tenure = rng.integers(1, 73, rows)
    monthly = np.round(rng.uniform(20, 130, rows), 2)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"], rows, p=[0.55, 0.27, 0.18]
    )
    payment = rng.choice(
        ["electronic-check", "credit-card", "bank-transfer", "mailed-check"],
        rows,
        p=[0.35, 0.28, 0.25, 0.12],
    )
    internet = rng.choice(["fiber", "dsl", "none"], rows, p=[0.48, 0.42, 0.10])
    support_calls = np.clip(rng.poisson(1.5, rows), 0, 8)
    late_payments = np.clip(rng.poisson(0.9, rows), 0, 7)
    total = np.round(monthly * tenure * rng.uniform(0.88, 1.05, rows), 2)

    # Multiple independent business factors contribute to churn. Gaussian noise and
    # Bernoulli sampling keep the target stochastic and prevent deterministic labels.
    logit = (
        -5.70
        + 3.00 * (contract == "month-to-month")
        + 1.35 * (payment == "electronic-check")
        + 1.50 * (internet == "fiber")
        + 0.72 * support_calls
        + 1.05 * late_payments
        + 0.027 * (monthly - 70)
        - 0.0675 * tenure
        - 0.018 * (age - 40)
        + rng.normal(0.0, 0.35, rows)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST-{index:06d}" for index in range(1, rows + 1)],
            "age": age,
            "tenure_months": tenure,
            "monthly_charge": monthly,
            "total_charge": total,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            "support_calls": support_calls,
            "late_payments": late_payments,
            "churn": churn,
        }
    )

    missing_columns = [
        "age",
        "monthly_charge",
        "total_charge",
        "contract_type",
        "payment_method",
        "internet_service",
    ]
    for column in missing_columns:
        missing_count = max(1, int(rows * 0.02))
        indices = rng.choice(rows, size=missing_count, replace=False)
        frame.loc[indices, column] = np.nan

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame

