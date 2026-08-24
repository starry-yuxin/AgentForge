"""Trusted in-process execution of one independently evaluated candidate."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agentforge.models import (
    AlgorithmRequirement, CandidatePlan, CandidateResult, GeneratedArtifact,
)
from agentforge.pipeline import evaluate_candidate


class ValidationAgent:
    def validate(
        self,
        artifact: GeneratedArtifact,
        requirement: AlgorithmRequirement,
        plan: CandidatePlan,
    ) -> CandidateResult:
        started = time.perf_counter()
        try:
            execution = evaluate_candidate(requirement.dataset_path, plan.algorithm)
            validation_metrics = execution["validation_metrics"]
            test_metrics = execution["metrics"]
            selection_value = float(validation_metrics[requirement.primary_metric])
            result = CandidateResult(
                plan_id=plan.plan_id, algorithm=plan.algorithm, status="completed",
                validation_metrics=validation_metrics, test_metrics=test_metrics,
                selected_threshold=execution["selected_threshold"],
                runtime_seconds=time.perf_counter() - started,
                generated_code_path=artifact.code_path,
                minimum_score_met=selection_value >= requirement.minimum_score,
                selection_metric_name=requirement.primary_metric,
                selection_metric_value=selection_value,
                validation_messages=[
                    "Model fitted on train split only.",
                    "Threshold selected using validation labels and probabilities only.",
                    "Test metrics computed only after threshold selection.",
                ],
            )
        except Exception as exc:  # candidate isolation is an explicit workflow requirement
            result = CandidateResult(
                plan_id=plan.plan_id, algorithm=plan.algorithm, status="failed",
                runtime_seconds=time.perf_counter() - started,
                generated_code_path=artifact.code_path,
                selection_metric_name=requirement.primary_metric,
                error=f"{type(exc).__name__}: {exc}",
                validation_messages=["Candidate failed independently; remaining candidates may continue."],
            )
        destination = Path(artifact.result_output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result
