"""Workflow input validation performed before knowledge retrieval and planning."""

from __future__ import annotations

import os
from pathlib import Path

from agentforge.models import AlgorithmRequirement


class DatasetNotFound(ValueError):
    """The requested dataset is unavailable as a readable regular file."""


def validate_dataset(requirement: AlgorithmRequirement) -> Path:
    """Validate the dataset without exposing its containing user directories."""
    path = Path(requirement.dataset_path).expanduser()
    display_name = path.name or "<unnamed>"
    if not path.exists():
        raise DatasetNotFound(f"dataset file does not exist: {display_name}")
    if not path.is_file():
        raise DatasetNotFound(f"dataset path is not a regular file: {display_name}")
    if not os.access(path, os.R_OK):
        raise DatasetNotFound(f"dataset file is not readable: {display_name}")
    return path
