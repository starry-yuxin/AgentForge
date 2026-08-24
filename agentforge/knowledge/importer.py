"""Import and validate structured knowledge extracted from example documents."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from agentforge.knowledge.models import Capability


def load_capabilities(path: str | Path) -> list[Capability]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    capabilities = TypeAdapter(list[Capability]).validate_python(payload)
    ids = [item.id for item in capabilities]
    if len(ids) != len(set(ids)):
        raise ValueError("capability ids must be unique")
    return capabilities


def validate_sources(capabilities: list[Capability], documents_dir: str | Path) -> None:
    root = Path(documents_dir)
    missing = sorted(
        {item.source_document for item in capabilities if not (root / item.source_document).is_file()}
    )
    if missing:
        raise ValueError(f"missing source documents: {missing}")

