"""Deterministic UCI Iranian Churn download/extraction adapter."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable

import pandas as pd

from agentforge.datasets.models import DatasetMetadata, LeakageReview
from agentforge.datasets.registry import ROOT, get_dataset_spec

RAW_COLUMNS = [
    "Call  Failure", "Complains", "Subscription  Length", "Charge  Amount",
    "Seconds of Use", "Frequency of use", "Frequency of SMS",
    "Distinct Called Numbers", "Age Group", "Tariff Plan", "Status", "Age",
    "Customer Value", "Churn",
]
FEATURE_MAPPING = {
    "Call  Failure": "call_failure", "Complains": "complains",
    "Subscription  Length": "subscription_length", "Charge  Amount": "charge_amount",
    "Seconds of Use": "seconds_of_use", "Frequency of use": "frequency_of_use",
    "Frequency of SMS": "frequency_of_sms",
    "Distinct Called Numbers": "distinct_called_numbers", "Age Group": "age_group",
    "Tariff Plan": "tariff_plan", "Status": "status", "Age": "age",
    "Customer Value": "customer_value", "Churn": "churn",
}
FEATURE_TYPES = {
    "call_failure": "numeric_count", "complains": "binary_categorical",
    "subscription_length": "numeric_duration", "charge_amount": "ordinal_categorical",
    "seconds_of_use": "numeric_duration", "frequency_of_use": "numeric_count",
    "frequency_of_sms": "numeric_count", "distinct_called_numbers": "numeric_count",
    "age_group": "ordinal_categorical", "tariff_plan": "categorical",
    "status": "categorical", "age": "numeric", "customer_value": "numeric_derived",
}
CATEGORICAL_COLUMNS = ["complains", "charge_amount", "age_group", "tariff_plan", "status"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member: {name}")


def _download_atomic(
    url: str, destination: Path, *, timeout: float,
    opener: Callable[..., BinaryIO], force: bool,
) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with opener(url, timeout=timeout) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        with zipfile.ZipFile(temporary) as archive:
            for info in archive.infolist():
                _safe_member(info.filename)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_uci_iranian_churn(
    *, root: str | Path | None = None, force: bool = False, timeout: float = 30.0,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> tuple[Path, Path]:
    """Download, validate and normalize the registered dataset without fitting ML state."""
    spec = get_dataset_spec("uci_iranian_churn")
    base = Path(root) if root else ROOT / "data" / "external" / spec.dataset_id
    archive_path = base / "raw" / spec.archive_file_name
    raw_csv_path = base / "raw" / spec.archive_member_name
    processed_path = base / "processed" / "uci_iranian_churn.csv"
    metadata_path = base / "metadata" / "dataset_metadata.json"
    _download_atomic(spec.download_url, archive_path, timeout=timeout, opener=opener, force=force)

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for name in names:
            _safe_member(name)
        if names.count(spec.archive_member_name) != 1:
            raise ValueError(f"expected exactly one ZIP member {spec.archive_member_name!r}")
        raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        raw_csv_path.write_bytes(archive.read(spec.archive_member_name))

    frame = pd.read_csv(raw_csv_path)
    if list(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected UCI schema: {list(frame.columns)!r}")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    if numeric.isna().any().any():
        raise ValueError("official UCI dataset unexpectedly contains missing values")
    classes = set(numeric["Churn"].astype(int).unique().tolist())
    if not classes or not classes.issubset({0, 1}):
        raise ValueError(f"target must be binary 0/1, got {sorted(classes)}")

    processed = numeric.rename(columns=FEATURE_MAPPING)
    for column in CATEGORICAL_COLUMNS:
        processed[column] = processed[column].map(lambda value, name=column: f"{name}_{int(value)}")
    processed.insert(0, "customer_id", [f"UCI-{index:06d}" for index in range(1, len(processed) + 1)])
    processed = processed[["customer_id", *FEATURE_MAPPING.values()]]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False, lineterminator="\n")

    downloaded_at = datetime.fromtimestamp(archive_path.stat().st_mtime, timezone.utc)
    metadata = DatasetMetadata(
        **spec.model_dump(), original_target_column=spec.target_original,
        target_column=spec.target_processed, data_origin="external_public_dataset",
        archive_sha256=_sha256(archive_path),
        raw_file_name=raw_csv_path.name, raw_file_sha256=_sha256(raw_csv_path),
        processed_file_name=processed_path.name, processed_file_sha256=_sha256(processed_path),
        row_count=len(processed), feature_count=len(FEATURE_MAPPING) - 1,
        total_column_count=len(processed.columns), positive_label=1, negative_label=0,
        positive_rate=float(processed["churn"].mean()),
        class_distribution={str(key): int(value) for key, value in
                            processed["churn"].value_counts().sort_index().items()},
        missing_value_count=int(processed.isna().sum().sum()),
        duplicate_row_count=int(frame.duplicated().sum()), feature_mapping=FEATURE_MAPPING,
        feature_types=FEATURE_TYPES,
        transformations=[
            "Validated the exact official CSV member and 14-column schema.",
            "Normalized column names with an explicit mapping and renamed Churn to churn.",
            "Converted documented integer-coded categorical fields to stable category labels.",
            "Added stable customer_id values from source row order; no rows or feature values changed.",
            "Performed no imputation, scaling, encoding, feature fitting, or resampling.",
        ], downloaded_at=downloaded_at, processed_at=datetime.now(timezone.utc),
        leakage_review=LeakageReview(
            target_removed_from_features=True, direct_target_derivatives=[],
            future_information_fields=[], identifier_fields=["customer_id"],
            prediction_time_assessment=(
                "UCI documents attributes as aggregated over months 1-9 and Churn at month 12; "
                "features therefore precede the outcome under that documented setup."
            ),
            preprocessing_fit_scope="AgentForge fits sklearn preprocessing on the training split only.",
            split_policy="Fixed stratified train/validation/test split; test is excluded from selection.",
            potential_risks=[
                "Status may act as a strong operational proxy and should be reviewed for deployment-time availability.",
                "Customer Value is a derived business feature; its upstream calculation should be audited before deployment.",
            ],
        ),
        disclaimer=(
            "Repository results demonstrate reproducibility on this public dataset only; they do not "
            "establish production performance, business generalization, or deployment readiness."
        ),
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return processed_path, metadata_path
