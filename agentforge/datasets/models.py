"""Strict contracts for dataset provenance and preparation results."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetSpec(DatasetModel):
    dataset_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    download_url: str = Field(min_length=1)
    doi: str = Field(min_length=1)
    license_name: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    task_type: Literal["binary_classification"]
    target_original: str
    target_processed: str
    archive_file_name: str
    archive_member_name: str
    real_world_data: Literal[True]


class LeakageReview(DatasetModel):
    target_removed_from_features: bool
    direct_target_derivatives: list[str]
    future_information_fields: list[str]
    identifier_fields: list[str]
    prediction_time_assessment: str
    preprocessing_fit_scope: str
    split_policy: str
    potential_risks: list[str]


class DatasetMetadata(DatasetModel):
    dataset_id: str
    title: str
    source_name: str
    source_url: str
    download_url: str
    doi: str
    license_name: str
    license_url: str
    task_type: Literal["binary_classification"]
    target_original: str
    target_processed: str
    original_target_column: str
    target_column: str
    data_origin: Literal["external_public_dataset"]
    real_world_data: Literal[True]
    archive_file_name: str
    archive_member_name: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_file_name: str
    raw_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processed_file_name: str
    processed_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(gt=0)
    feature_count: int = Field(gt=0)
    total_column_count: int = Field(gt=0)
    positive_label: int | str
    negative_label: int | str
    positive_rate: float = Field(ge=0.0, le=1.0)
    class_distribution: dict[str, int]
    missing_value_count: int = Field(ge=0)
    duplicate_row_count: int = Field(ge=0)
    feature_mapping: dict[str, str]
    feature_types: dict[str, str]
    transformations: list[str]
    downloaded_at: datetime
    processed_at: datetime
    leakage_review: LeakageReview
    disclaimer: str
