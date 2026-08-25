from __future__ import annotations

import hashlib
import io
import json
import ssl
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from agentforge.datasets.models import DatasetMetadata
from agentforge.datasets import get_dataset_spec, resolve_dataset_metadata
from agentforge.datasets.uci_iranian_churn import (
    FEATURE_MAPPING, RAW_COLUMNS, prepare_uci_iranian_churn,
)
from agentforge.workflow import WorkflowOrchestrator
from agentforge.config import LLMConfig


def _csv(rows: int = 120) -> bytes:
    data = []
    for index in range(rows):
        churn = int(index % 4 == 0)
        data.append([
            index % 8, index % 2, 8 + index % 30, index % 10,
            100 + index * 3, 5 + index % 20, index % 12, 3 + index % 25,
            1 + index % 5, 1 + index % 2, index % 2, 20 + index % 50,
            50.0 + index * 1.2, churn,
        ])
    return pd.DataFrame(data, columns=RAW_COLUMNS).to_csv(index=False).encode()


def _zip(payload: bytes | None = None, member: str = "Customer Churn.csv") -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(member, payload or _csv())
    return stream.getvalue()


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _opener(payload: bytes, calls: list[tuple[str, float]]):
    def open_url(url: str, *, timeout: float):
        calls.append((url, timeout))
        return Response(payload)
    return open_url


def test_registered_source_is_official_and_portable():
    spec = get_dataset_spec("uci_iranian_churn")
    assert spec.dataset_id == "uci_iranian_churn"
    assert spec.source_name == "UCI Machine Learning Repository"
    assert spec.source_url.startswith("https://archive.ics.uci.edu/")
    assert spec.download_url.startswith("https://archive.ics.uci.edu/")
    assert spec.doi == "10.24432/C5JW3Z"
    assert spec.license_name == "CC BY 4.0"
    assert spec.real_world_data is True
    assert "/Users/" not in spec.model_dump_json()


def test_prepare_maps_schema_and_writes_auditable_metadata(tmp_path: Path):
    calls: list[tuple[str, float]] = []
    processed, metadata_path = prepare_uci_iranian_churn(
        root=tmp_path, opener=_opener(_zip(), calls), timeout=4.5,
    )
    frame = pd.read_csv(processed)
    metadata = DatasetMetadata.model_validate_json(metadata_path.read_text())

    assert len(calls) == 1 and calls[0][1] == 4.5
    assert list(frame.columns) == ["customer_id", *FEATURE_MAPPING.values()]
    assert frame["customer_id"].iloc[[0, -1]].tolist() == ["UCI-000001", "UCI-000120"]
    assert set(frame["churn"]) == {0, 1}
    assert frame["tariff_plan"].str.startswith("tariff_plan_").all()
    assert frame["status"].str.startswith("status_").all()
    assert metadata.row_count == 120
    assert metadata.feature_count == 13
    assert metadata.total_column_count == 15
    assert metadata.target_column == "churn"
    assert metadata.positive_rate == 0.25
    assert metadata.missing_value_count == 0
    assert metadata.real_world_data is True
    assert metadata.processed_file_sha256 == hashlib.sha256(processed.read_bytes()).hexdigest()
    assert metadata.feature_mapping["Churn"] == "churn"
    assert metadata.leakage_review.target_removed_from_features


def test_default_download_uses_verified_certifi_context(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_urlopen(url: str, *, timeout: float, context: ssl.SSLContext):
        captured.update(url=url, timeout=timeout, context=context)
        return Response(_zip())

    monkeypatch.setattr("agentforge.datasets.uci_iranian_churn.urllib.request.urlopen", fake_urlopen)
    prepare_uci_iranian_churn(root=tmp_path, timeout=3.5)

    context = captured["context"]
    assert captured["url"].startswith("https://archive.ics.uci.edu/")
    assert captured["timeout"] == 3.5
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_existing_archive_is_not_downloaded_and_output_is_stable(tmp_path: Path):
    calls: list[tuple[str, float]] = []
    processed, _ = prepare_uci_iranian_churn(root=tmp_path, opener=_opener(_zip(), calls))
    first = hashlib.sha256(processed.read_bytes()).hexdigest()
    processed, _ = prepare_uci_iranian_churn(
        root=tmp_path, opener=lambda *_args, **_kwargs: pytest.fail("network called"),
    )
    assert hashlib.sha256(processed.read_bytes()).hexdigest() == first
    assert len(calls) == 1


def test_registry_rejects_processed_file_that_does_not_match_metadata(tmp_path: Path):
    processed, _ = prepare_uci_iranian_churn(root=tmp_path, opener=_opener(_zip(), []))
    processed.write_text(processed.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        resolve_dataset_metadata(processed)


def test_force_replaces_archive_and_timeout_propagates(tmp_path: Path):
    first_calls: list[tuple[str, float]] = []
    prepare_uci_iranian_churn(root=tmp_path, opener=_opener(_zip(), first_calls))
    second_calls: list[tuple[str, float]] = []
    prepare_uci_iranian_churn(
        root=tmp_path, force=True, timeout=1.25, opener=_opener(_zip(_csv(124)), second_calls),
    )
    assert second_calls[0][1] == 1.25
    assert len(pd.read_csv(tmp_path / "processed" / "uci_iranian_churn.csv")) == 124


def test_download_failure_leaves_no_archive(tmp_path: Path):
    def failing(*_args, **_kwargs):
        raise TimeoutError("offline fixture timeout")
    with pytest.raises(TimeoutError):
        prepare_uci_iranian_churn(root=tmp_path, opener=failing)
    assert not (tmp_path / "raw" / "iranian_churn.zip").exists()


@pytest.mark.parametrize("member", ["../Customer Churn.csv", "/Customer Churn.csv"])
def test_unsafe_zip_members_are_rejected(tmp_path: Path, member: str):
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        prepare_uci_iranian_churn(root=tmp_path, opener=_opener(_zip(member=member), []))


def test_wrong_zip_member_and_schema_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="expected exactly one ZIP member"):
        prepare_uci_iranian_churn(root=tmp_path / "member", opener=_opener(_zip(member="other.csv"), []))
    malformed = pd.DataFrame([[1]], columns=["Churn"]).to_csv(index=False).encode()
    with pytest.raises(ValueError, match="unexpected UCI schema"):
        prepare_uci_iranian_churn(root=tmp_path / "schema", opener=_opener(_zip(malformed), []))


def test_real_dataset_workflow_reports_provenance_without_persistence(tmp_path: Path, monkeypatch):
    processed, metadata_path = prepare_uci_iranian_churn(
        root=tmp_path / "source", opener=_opener(_zip(), []),
    )
    # The registry deliberately recognizes only its canonical location; redirect it for this fixture.
    import agentforge.datasets.registry as registry
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    canonical = tmp_path / "data/external/uci_iranian_churn"
    canonical.parent.mkdir(parents=True)
    (canonical / "processed").mkdir(parents=True)
    (canonical / "metadata").mkdir(parents=True)
    canonical_processed = canonical / "processed/uci_iranian_churn.csv"
    canonical_processed.write_bytes(processed.read_bytes())
    (canonical / "metadata/dataset_metadata.json").write_bytes(metadata_path.read_bytes())

    state = WorkflowOrchestrator(
        output_root=tmp_path / "runs", llm_config=LLMConfig(mode="deterministic")
    ).run(
        "请为客户流失数据比较Logistic Regression和Random Forest，以F1作为主要指标。",
        overrides={"dataset_path": str(canonical_processed)}, persist=False,
    )
    assert state.status == "completed"
    assert state.llm_call_count == 0
    assert len(state.candidate_results) == 2
    assert state.dataset_metadata["dataset_id"] == "uci_iranian_churn"
    report = json.loads(Path(state.final_report_paths["json"]).read_text())
    assert report["dataset_metadata"]["real_world_data"] is True
    assert "UCI Machine Learning Repository" in Path(state.final_report_paths["markdown"]).read_text()
    assert not state.knowledge_persisted
