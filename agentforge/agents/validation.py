"""Trusted in-process execution of one independently evaluated candidate."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from agentforge.models import (
    AlgorithmRequirement, CandidatePlan, CandidateResult, GeneratedArtifact,
)
from agentforge.pipeline import evaluate_candidate
from agentforge.models import ExecutionResult, InterfaceCheckResult, SecurityCheckResult, ValidationCheck


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

    def assess(
        self,
        artifact: GeneratedArtifact,
        requirement: AlgorithmRequirement,
        plan: CandidatePlan,
        security: SecurityCheckResult,
        interface: InterfaceCheckResult,
        execution: ExecutionResult,
    ) -> CandidateResult:
        """Validate controlled subprocess artifacts without importing candidate code."""
        checks = [
            ValidationCheck(name="syntax", category="syntax", passed=security.passed or not any(
                finding.category == "syntax" for finding in security.findings
            ), expected="valid Python AST", actual="valid" if security.passed else "findings",
                            message="Candidate source must parse."),
            ValidationCheck(name="security_policy", category="security", passed=security.passed,
                            expected="no blocking findings", actual=len(security.findings),
                            message="Lightweight AST policy; not a production sandbox."),
            ValidationCheck(name="interface_contract", category="interface", passed=interface.passed,
                            expected=interface.required_functions,
                            actual=interface.discovered_functions,
                            message="Generated module must expose the unified interface."),
            ValidationCheck(name="subprocess_exit", category="execution",
                            passed=execution.process_status == "completed",
                            expected="return code 0 with result and model files",
                            actual={"status": execution.process_status,
                                    "return_code": execution.return_code},
                            message="Candidate executes in a timeout-bounded child process."),
            ValidationCheck(name="timeout", category="resource", passed=not execution.timed_out,
                            expected=False, actual=execution.timed_out,
                            message="Execution must finish before max_runtime_seconds."),
        ]
        payload = None
        result_path = Path(execution.result_json_path)
        if execution.process_status == "completed" and result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = None
        checks.append(ValidationCheck(
            name="result_json", category="functionality", passed=isinstance(payload, dict),
            expected="valid result schema", actual=type(payload).__name__,
            message="Harness must emit machine-readable result JSON.",
        ))
        model_exists = Path(execution.model_path).is_file()
        checks.append(ValidationCheck(
            name="model_artifact", category="functionality", passed=model_exists,
            expected=True, actual=model_exists,
            message="train must save a reloadable model used by predict and evaluate.",
        ))

        validation_metrics: dict[str, float] = {}
        test_metrics: dict[str, object] = {}
        selected_threshold = None
        requested_hyperparameters: dict[str, object] = {}
        effective_hyperparameters: dict[str, object] = {}
        if payload:
            evaluation = payload.get("evaluation", {})
            train_result = payload.get("train_result", {})
            validation_metrics = evaluation.get("validation_metrics", {})
            test_metrics = evaluation.get("test_metrics", {})
            selected_threshold = evaluation.get("selected_threshold")
            requested_hyperparameters = train_result.get(
                "requested_hyperparameters", evaluation.get("requested_hyperparameters", {})
            )
            effective_hyperparameters = train_result.get(
                "effective_hyperparameters", evaluation.get("effective_hyperparameters", {})
            )
            expected_rows = sum(evaluation.get("split_sizes", {}).values())
            prediction_count = payload.get("prediction_count")
            labels = payload.get("prediction_labels", [])
            checks.extend([
                ValidationCheck(name="prediction_length", category="functionality",
                                passed=prediction_count == expected_rows,
                                expected=expected_rows, actual=prediction_count,
                                message="predict must return one label per input row."),
                ValidationCheck(name="prediction_labels", category="functionality",
                                passed=set(labels).issubset({0, 1}), expected=[0, 1], actual=labels,
                                message="Predictions must contain binary labels only."),
                ValidationCheck(name="split_separation", category="metrics",
                                passed=bool(validation_metrics) and bool(test_metrics),
                                expected=["validation_metrics", "test_metrics"],
                                actual=list(evaluation),
                                message="Validation and test metrics must remain separate."),
                ValidationCheck(name="selected_threshold", category="metrics",
                                passed=isinstance(selected_threshold, (int, float)) and
                                       0.0 <= float(selected_threshold) <= 1.0,
                                expected="0 <= threshold <= 1", actual=selected_threshold,
                                message="Threshold is selected on validation only."),
            ])
            required_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
            present = required_metrics.issubset(validation_metrics) and required_metrics.issubset(test_metrics)
            finite = present and all(
                isinstance(metrics[name], (int, float)) and math.isfinite(float(metrics[name]))
                and 0.0 <= float(metrics[name]) <= 1.0
                for metrics in (validation_metrics, test_metrics) for name in required_metrics
            )
            checks.append(ValidationCheck(
                name="metric_schema", category="metrics", passed=finite,
                expected=sorted(required_metrics),
                actual={"validation": sorted(validation_metrics), "test": sorted(test_metrics)},
                message="Required metrics must be finite and within [0, 1].",
            ))
        selection_value = validation_metrics.get(requirement.primary_metric)
        minimum_met = isinstance(selection_value, (int, float)) and \
            float(selection_value) >= requirement.minimum_score
        checks.append(ValidationCheck(
            name="minimum_score", category="metrics", passed=minimum_met,
            expected=f">={requirement.minimum_score}", actual=selection_value,
            message="Minimum score is evaluated on validation, not test.",
        ))
        blocking_names = {"syntax", "security_policy", "interface_contract", "subprocess_exit",
                          "timeout", "result_json", "model_artifact", "prediction_length",
                          "prediction_labels", "split_separation", "selected_threshold", "metric_schema"}
        passed = all(check.passed for check in checks if check.name in blocking_names)
        return CandidateResult(
            plan_id=plan.plan_id, algorithm=plan.algorithm,
            status="completed" if passed else "failed",
            validation_metrics={key: float(value) for key, value in validation_metrics.items()
                               if isinstance(value, (int, float))},
            test_metrics=test_metrics, selected_threshold=selected_threshold,
            runtime_seconds=execution.duration_seconds,
            generated_code_path=artifact.code_path, minimum_score_met=minimum_met,
            selection_metric_name=requirement.primary_metric,
            selection_metric_value=float(selection_value) if isinstance(selection_value, (int, float)) else None,
            error=None if passed else (execution.stderr[-1000:] or "validation checks failed"),
            validation_messages=[check.message for check in checks],
            attempts=[execution], security_result=security, interface_result=interface,
            validation_checks=checks, final_code_path=artifact.code_path if passed else None,
            execution_logs=[str(Path(artifact.code_path).parent / "stdout.log"),
                            str(Path(artifact.code_path).parent / "stderr.log")],
            requested_hyperparameters=requested_hyperparameters,
            effective_hyperparameters=effective_hyperparameters,
        )
