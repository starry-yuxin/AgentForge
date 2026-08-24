"""Classify candidate failures from structured checks and real process output."""

from __future__ import annotations

from agentforge.models import (
    ExecutionResult, InterfaceCheckResult, SecurityCheckResult, ValidationCheck,
)


class ErrorClassifier:
    def classify(
        self,
        *,
        execution: ExecutionResult,
        security: SecurityCheckResult,
        interface: InterfaceCheckResult,
        checks: list[ValidationCheck],
    ) -> str:
        if any(item.category == "syntax" for item in security.findings):
            return "SyntaxError"
        if not security.passed:
            return "SecurityViolation"
        if not interface.passed:
            return "MissingInterface"
        if execution.timed_out:
            return "TimeoutError"
        combined = f"{execution.stderr}\n{execution.stdout}".lower()
        if "input x contains nan" in combined or "does not accept missing values" in combined \
                or ("nan" in combined and "simpleimputer" in combined):
            return "MissingValueError"
        if "could not convert string to float" in combined or "unknown categories" in combined:
            return "CategoricalEncodingError"
        failed_names = {check.name for check in checks if not check.passed}
        if {"result_json", "prediction_length", "prediction_labels"} & failed_names:
            return "InvalidReturnFormat"
        if {"metric_schema", "selected_threshold", "split_separation"} & failed_names:
            return "MetricValidationError"
        if execution.return_code not in (0, None):
            return "UnknownExecutionError"
        return "UnknownExecutionError"
