"""Small explicit registry and report-time provenance resolver."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agentforge.datasets.models import DatasetMetadata, DatasetSpec

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "datasets"


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    if dataset_id != "uci_iranian_churn":
        raise ValueError(f"unsupported dataset_id: {dataset_id}")
    return DatasetSpec.model_validate_json(
        (CONFIG_DIR / "uci_iranian_churn.json").read_text(encoding="utf-8")
    )


def resolve_dataset_metadata(dataset_path: str | Path) -> dict[str, Any]:
    """Return validated real-data metadata, otherwise a conservative local label."""
    path = Path(dataset_path).resolve()
    metadata_path = path.parents[1] / "metadata" / "dataset_metadata.json"
    if path.name == "uci_iranian_churn.csv" and metadata_path.is_file():
        metadata = DatasetMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata.processed_file_sha256:
            raise ValueError("processed dataset SHA-256 does not match registered metadata")
        return metadata.model_dump(mode="json")
    return {
        "dataset_id": "local_or_synthetic_churn",
        "title": "Local customer churn dataset",
        "data_origin": "synthetic" if path.name == "churn_sample.csv" else "local_unregistered",
        "real_world_data": False,
        "processed_file_name": path.name,
        "disclaimer": "No external real-world provenance metadata is registered for this path.",
    }
