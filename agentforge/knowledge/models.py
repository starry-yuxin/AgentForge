"""Validated data contracts for the knowledge layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


NodeType = Literal[
    "Task", "Algorithm", "Preprocessor", "Metric", "Constraint", "Dataset",
    "ValidationRun", "FailureExperience", "Dependency", "SourceDocument",
]


class Capability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9_\-]+$")
    name: str = Field(min_length=1)
    type: NodeType
    description: str = Field(min_length=1)
    inputs: list[str]
    outputs: list[str]
    applicable_tasks: list[str]
    applicable_conditions: list[str]
    constraints: list[str]
    dependencies: list[str]
    metrics: list[str]
    source_document: str = Field(min_length=1)
    source_section: str = Field(min_length=1)
    version: str = Field(min_length=1)


class Recommendation(BaseModel):
    id: str
    name: str
    node_type: str
    score: int
    reasons: list[str]
    source_document: str
    source_section: str


class RetrievalResult(BaseModel):
    query: dict[str, Any]
    algorithms: list[Recommendation]
    preprocessors: list[Recommendation]
    metrics: list[Recommendation]
    failure_experiences: list[Recommendation]

