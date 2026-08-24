"""Pydantic data contracts for the deterministic multi-agent workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SUPPORTED_TASKS = {"binary_classification"}
SUPPORTED_METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc"}
SUPPORTED_ALGORITHMS = {"logistic_regression", "random_forest"}


class WorkflowModel(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_timedelta="float")


class AlgorithmRequirement(WorkflowModel):
    request_id: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    industry: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    primary_metric: str
    minimum_score: float = Field(ge=0.0, le=1.0)
    candidate_algorithms: list[str] = Field(min_length=1)
    data_characteristics: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    required_interfaces: list[str] = Field(
        default_factory=lambda: ["train", "predict", "evaluate"]
    )
    max_runtime_seconds: float = Field(default=120.0, gt=0.0)
    field_sources: dict[str, str] = Field(default_factory=dict)

    @field_validator("task_type")
    @classmethod
    def validate_task(cls, value: str) -> str:
        if value not in SUPPORTED_TASKS:
            raise ValueError(f"unsupported task_type: {value}")
        return value

    @field_validator("primary_metric")
    @classmethod
    def validate_metric(cls, value: str) -> str:
        if value not in SUPPORTED_METRICS:
            raise ValueError(f"unsupported primary_metric: {value}")
        return value

    @field_validator("candidate_algorithms")
    @classmethod
    def normalize_candidates(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("candidate_algorithms cannot be empty")
        unknown = sorted(set(values) - SUPPORTED_ALGORITHMS)
        if unknown:
            raise ValueError(f"unsupported candidate algorithms: {unknown}")
        return list(dict.fromkeys(values))

    @field_validator("required_interfaces")
    @classmethod
    def validate_interfaces(cls, values: list[str]) -> list[str]:
        required = {"train", "predict", "evaluate"}
        if not required.issubset(values):
            raise ValueError("required_interfaces must include train, predict, and evaluate")
        return list(dict.fromkeys(values))


class RetrievedCapability(WorkflowModel):
    capability_id: str
    name: str
    capability_type: str
    description: str = ""
    match_reasons: list[str] = Field(default_factory=list)
    source_document: str
    source_section: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedKnowledge(WorkflowModel):
    algorithms: list[RetrievedCapability] = Field(default_factory=list)
    preprocessors: list[RetrievedCapability] = Field(default_factory=list)
    metrics: list[RetrievedCapability] = Field(default_factory=list)
    failure_experiences: list[RetrievedCapability] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    retrieval_summary: str = ""
    total_matches: int = Field(default=0, ge=0)


class CandidatePlan(WorkflowModel):
    plan_id: str
    algorithm: str
    preprocessing_steps: list[str]
    hyperparameters: dict[str, Any]
    threshold_strategy: str
    evaluation_metrics: list[str]
    rationale: str
    supporting_capability_ids: list[str] = Field(min_length=1)
    expected_interfaces: list[str]


class GeneratedArtifact(WorkflowModel):
    plan_id: str
    algorithm: str
    code_path: str
    model_output_path: str
    result_output_path: str
    generator_mode: Literal["deterministic_template"] = "deterministic_template"
    interface_spec: list[str]
    source_plan: CandidatePlan
    syntax_valid: bool


class CandidateResult(WorkflowModel):
    plan_id: str
    algorithm: str
    status: Literal["completed", "failed"]
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    test_metrics: dict[str, Any] = Field(default_factory=dict)
    selected_threshold: float | None = None
    runtime_seconds: float = Field(default=0.0, ge=0.0)
    generated_code_path: str
    minimum_score_met: bool = False
    selection_metric_name: str
    selection_metric_value: float | None = None
    error: str | None = None
    validation_messages: list[str] = Field(default_factory=list)


class WorkflowEvent(WorkflowModel):
    event_id: str
    agent_name: str
    status: Literal["started", "completed", "failed", "skipped"]
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float = Field(default=0.0, ge=0.0)
    input_summary: str = ""
    output_summary: str = ""
    message: str = ""


class WorkflowState(WorkflowModel):
    run_id: str
    request: AlgorithmRequirement | None = None
    retrieved_knowledge: RetrievedKnowledge | None = None
    candidate_plans: list[CandidatePlan] = Field(default_factory=list)
    generated_artifacts: list[GeneratedArtifact] = Field(default_factory=list)
    candidate_results: list[CandidateResult] = Field(default_factory=list)
    best_candidate: CandidateResult | None = None
    events: list[WorkflowEvent] = Field(default_factory=list)
    status: Literal[
        "pending", "running", "completed", "partially_completed", "failed"
    ] = "pending"
    final_report_paths: dict[str, str] = Field(default_factory=dict)
    knowledge_persisted: bool = False
    errors: list[str] = Field(default_factory=list)
