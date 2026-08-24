"""Export small, sanitized, deterministic examples without reading dotenv or calling an LLM."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge.cli import DEMO_REQUEST
from agentforge.config import LLMConfig
from agentforge.workflow import WorkflowOrchestrator


def _run(root: Path, *, inject_failure: str | None = None):
    return WorkflowOrchestrator(
        output_root=root,
        llm_config=LLMConfig(mode="deterministic"),
    ).run(DEMO_REQUEST, persist=False, inject_failure=inject_failure)


def _candidate_summary(result) -> dict:
    return {
        "algorithm": result.algorithm,
        "status": result.status,
        "selection_metric_on_validation": {
            "name": result.selection_metric_name,
            "value": result.selection_metric_value,
        },
        "validation_metrics": result.validation_metrics,
        "final_metrics_on_test": result.test_metrics,
        "selected_threshold": result.selected_threshold,
        "minimum_score_met": result.minimum_score_met,
        "security_passed": result.security_result.passed if result.security_result else None,
        "interface_passed": result.interface_result.passed if result.interface_result else None,
        "attempt_count": len(result.attempts),
    }


def export(destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="agentforge-examples-") as temporary:
        temporary_root = Path(temporary)
        normal = _run(temporary_root / "normal")
        fault = _run(temporary_root / "fault", inject_failure="missing_imputer")
        if normal.status != "completed" or fault.status != "completed":
            raise RuntimeError("deterministic example workflows did not complete")

        request_dir = destination / "requests"
        generated_dir = destination / "generated"
        report_dir = destination / "reports"
        repair_dir = destination / "repair_case"
        for directory in (request_dir, generated_dir, report_dir, repair_dir):
            directory.mkdir(parents=True, exist_ok=True)
        (request_dir / "churn_request.txt").write_text(DEMO_REQUEST + "\n", encoding="utf-8")

        first_artifacts = {artifact.algorithm: artifact for artifact in normal.generated_artifacts
                           if artifact.attempt == 0}
        for algorithm in ("logistic_regression", "random_forest"):
            shutil.copyfile(first_artifacts[algorithm].code_path, generated_dir / f"{algorithm}.py")

        payload = {
            "schema_version": "1.0",
            "example": True,
            "data_notice": "Synthetic customer churn data; not evidence of business generalization.",
            "execution_mode": "deterministic",
            "llm_call_count": normal.llm_call_count,
            "request": {
                "task_type": normal.request.task_type,
                "dataset_path": "data/churn_sample.csv",
                "target_column": normal.request.target_column,
                "primary_metric": normal.request.primary_metric,
                "minimum_score": normal.request.minimum_score,
                "candidate_algorithms": normal.request.candidate_algorithms,
            },
            "selection_policy": "Candidates and thresholds are selected on validation only.",
            "test_policy": "Test data is used only for final reporting after selection.",
            "candidates": [_candidate_summary(result) for result in normal.candidate_results],
            "best_candidate": normal.best_candidate.algorithm,
            "knowledge_persisted": normal.knowledge_persisted,
        }
        (report_dir / "workflow_report.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        by_name = {item["algorithm"]: item for item in payload["candidates"]}
        logistic = by_name["logistic_regression"]
        forest = by_name["random_forest"]
        markdown = f"""# AgentForge deterministic example report

> This committed example uses synthetic customer churn data and makes zero LLM calls.

- Mode: `deterministic`
- Dataset: `data/churn_sample.csv`
- Primary metric: `f1`
- Best candidate: `{payload['best_candidate']}`
- Selection rule: validation metrics only
- Test role: final reporting only; never candidate selection

| Candidate | Validation F1 | Test F1 | Test ROC-AUC | Threshold |
|---|---:|---:|---:|---:|
| Logistic Regression | {logistic['validation_metrics']['f1']:.6f} | {logistic['final_metrics_on_test']['f1']:.6f} | {logistic['final_metrics_on_test']['roc_auc']:.6f} | {logistic['selected_threshold']:.3f} |
| Random Forest | {forest['validation_metrics']['f1']:.6f} | {forest['final_metrics_on_test']['f1']:.6f} | {forest['final_metrics_on_test']['roc_auc']:.6f} | {forest['selected_threshold']:.3f} |

Both candidates passed the AST security policy, interface contract and controlled subprocess execution. These synthetic-data metrics do not demonstrate real-business generalization.
"""
        (report_dir / "workflow_report.md").write_text(markdown, encoding="utf-8")

        repaired = next(result for result in fault.candidate_results
                        if result.algorithm == "logistic_regression")
        repair = repaired.repair_records[0]
        shutil.copyfile(repair.diff_path, repair_dir / "repair.diff")
        repair_readme = """# Missing-imputer repair example

The deterministic fault injection routes Logistic Regression attempt-0 to a training helper without imputers. Execution fails on real missing values and is classified as `MissingValueError`.

AgentForge retrieves the `missing_value_error` FailureExperience and its `IMPROVED_BY` links, regenerates attempt-1 with numerical and categorical imputation, and reruns AST, interface and subprocess validation. The sanitized `repair.diff` is copied from that real deterministic repair run.
"""
        (repair_dir / "README.md").write_text(repair_readme, encoding="utf-8")


def main() -> None:
    export(ROOT / "examples")
    print("Exported sanitized deterministic examples to examples/")


if __name__ == "__main__":
    main()
