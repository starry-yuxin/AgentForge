"""Structured contracts shared by the deterministic workflow agents."""

from agentforge.models.workflow import (
    AlgorithmRequirement,
    CandidatePlan,
    CandidateResult,
    GeneratedArtifact,
    RetrievedCapability,
    RetrievedKnowledge,
    WorkflowEvent,
    WorkflowState,
    SecurityFinding,
    SecurityCheckResult,
    InterfaceCheckResult,
    ExecutionResult,
    ValidationCheck,
    RepairRecord,
)

__all__ = [
    "AlgorithmRequirement", "CandidatePlan", "CandidateResult", "GeneratedArtifact",
    "RetrievedCapability", "RetrievedKnowledge", "WorkflowEvent", "WorkflowState",
    "SecurityFinding", "SecurityCheckResult", "InterfaceCheckResult", "ExecutionResult",
    "ValidationCheck", "RepairRecord",
]
