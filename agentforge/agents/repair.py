"""Knowledge-grounded deterministic candidate repair."""

from __future__ import annotations

import difflib
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agentforge.agents.code import CodeAgent
from agentforge.knowledge import KnowledgeRetriever
from agentforge.models import (
    AlgorithmRequirement, CandidatePlan, GeneratedArtifact, RepairRecord, ValidationCheck,
)


REPAIRABLE = {"MissingValueError", "MissingInterface", "InvalidReturnFormat"}


class RepairAgent:
    def __init__(self, retriever: KnowledgeRetriever, code_agent: CodeAgent | None = None) -> None:
        self.retriever = retriever
        self.code_agent = code_agent or CodeAgent()

    def repair(
        self,
        requirement: AlgorithmRequirement,
        plan: CandidatePlan,
        artifact: GeneratedArtifact,
        failure_type: str,
        validation_before: list[ValidationCheck],
        *,
        attempt: int,
        run_id: str,
        run_dir: str | Path,
        error_summary: str,
    ) -> tuple[RepairRecord, GeneratedArtifact]:
        if attempt not in {1, 2}:
            raise ValueError("repair attempt must be 1 or 2")
        if failure_type not in REPAIRABLE:
            raise ValueError(f"failure is not deterministically repairable: {failure_type}")
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        retrieval = self.retriever.query(
            task_type=requirement.task_type,
            data_characteristics=requirement.data_characteristics,
            required_metric=requirement.primary_metric,
            failure_type=failure_type,
        )
        experiences = [item for item in retrieval.failure_experiences
                       if item.id == failure_type.lower().replace("error", "_error")
                       or item.name.lower() == failure_type.lower()]
        if failure_type == "MissingValueError":
            experiences = [item for item in retrieval.failure_experiences
                           if item.id == "missing_value_error"]
        experience_ids = [item.id for item in experiences]
        if failure_type == "MissingValueError" and not experience_ids:
            raise LookupError("knowledge graph did not return MissingValueError experience")
        improved_by = []
        for experience_id in experience_ids:
            for _, target, attributes in self.retriever.store.graph.out_edges(
                experience_id, data=True
            ):
                if attributes.get("relation") == "IMPROVED_BY":
                    improved_by.append(target)
        strategy = {
            "MissingValueError": (
                "Regenerate trusted template with NumericalImputation and "
                "CategoricalImputation before encoding/scaling."
            ),
            "MissingInterface": "Regenerate all train/predict/evaluate wrapper functions.",
            "InvalidReturnFormat": "Regenerate the constrained result adapter and interface.",
        }[failure_type]
        repaired = self.code_agent.generate(
            plan, run_id, run_dir, attempt=attempt, failure_injection=None
        )
        before = Path(artifact.code_path).read_text(encoding="utf-8").splitlines(keepends=True)
        after = Path(repaired.code_path).read_text(encoding="utf-8").splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(
            before, after, fromfile=f"attempt-{artifact.attempt}/candidate.py",
            tofile=f"attempt-{attempt}/candidate.py",
        ))
        diff_path = Path(repaired.code_path).parent / "repair.diff"
        diff_path.write_text(diff, encoding="utf-8")
        finished_at = datetime.now(timezone.utc)
        record = RepairRecord(
            repair_id=f"repair-{uuid4().hex[:12]}", plan_id=plan.plan_id,
            attempt=attempt, failure_type=failure_type, error_summary=error_summary[-1000:],
            retrieved_experience_ids=experience_ids,
            repair_strategy=strategy + (f" Graph improvements: {sorted(set(improved_by))}."
                                        if improved_by else ""),
            source_code_path=artifact.code_path, repaired_code_path=repaired.code_path,
            diff_path=str(diff_path), validation_before=validation_before,
            started_at=started_at, finished_at=finished_at,
            duration_seconds=time.perf_counter() - started,
        )
        return record, repaired
