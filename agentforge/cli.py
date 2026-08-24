"""Command-line entry point for deterministic AgentForge workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentforge.workflow import WorkflowOrchestrator


DEMO_REQUEST = (
    "请构建客户流失预测模型，比较Logistic Regression和Random Forest，"
    "以F1作为主要指标，最低要求为0.60，数据包含缺失值、类别特征和类别不平衡"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgentForge deterministic workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the built-in deterministic demo")
    demo.add_argument("--inject-failure", choices=["missing_imputer"])
    run = subparsers.add_parser("run", help="run a natural-language request")
    run.add_argument("--request", required=True)
    run.add_argument("--dataset")
    run.add_argument("--metric")
    run.add_argument("--minimum-score", type=float)
    run.add_argument("--output-root")
    run.add_argument("--no-persist", action="store_true")
    run.add_argument("--inject-failure", choices=["missing_imputer"])
    return parser


def _print_summary(state) -> None:
    print(f"run_id: {state.run_id}")
    if state.request:
        print(f"requirement: task={state.request.task_type}, metric={state.request.primary_metric}")
    print(f"retrieved_capabilities: {state.retrieved_knowledge.total_matches if state.retrieved_knowledge else 0}")
    for plan in state.candidate_plans:
        print(f"plan: {plan.algorithm} ({plan.plan_id})")
    for artifact in state.generated_artifacts:
        print(f"generated: {artifact.code_path}")
    for result in state.candidate_results:
        print(
            f"result: {result.algorithm} status={result.status} "
            f"validation_f1={result.validation_metrics.get('f1')} "
            f"test_f1={result.test_metrics.get('f1')} "
            f"roc_auc={result.test_metrics.get('roc_auc')} "
            f"threshold={result.selected_threshold}"
        )
        for attempt in result.attempts:
            print(
                f"attempt: {result.algorithm} #{attempt.attempt} "
                f"security={result.security_result.passed if result.security_result else None} "
                f"interface={result.interface_result.passed if result.interface_result else None} "
                f"returncode={attempt.return_code} status={attempt.process_status}"
            )
        if result.failure_type:
            print(f"failure_type: {result.algorithm} {result.failure_type}")
        for repair in result.repair_records:
            print(
                f"repair: attempt={repair.attempt} mode={repair.repair_mode} "
                f"experiences={repair.retrieved_experience_ids} "
                f"strategy={repair.repair_strategy} status={repair.status}"
            )
    if state.best_candidate:
        print(
            f"best_candidate: {state.best_candidate.algorithm} selected by validation "
            f"{state.best_candidate.selection_metric_name}="
            f"{state.best_candidate.selection_metric_value}"
        )
        print(f"minimum_score_met: {state.best_candidate.minimum_score_met}")
    print(f"json_report: {state.final_report_paths.get('json', '')}")
    print(f"markdown_report: {state.final_report_paths.get('markdown', '')}")
    print(f"knowledge_persisted: {state.knowledge_persisted}")
    print(f"failure_injection: {state.failure_injection}")
    print(f"total_repair_attempts: {state.total_repair_attempts}")
    print(f"status: {state.status}")
    for error in state.errors:
        print(f"error: {error}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "demo":
        orchestrator = WorkflowOrchestrator()
        state = orchestrator.run(
            DEMO_REQUEST, persist=False, inject_failure=args.inject_failure
        )
    else:
        overrides = {
            "dataset_path": str(Path(args.dataset).resolve()) if args.dataset else None,
            "primary_metric": args.metric,
            "minimum_score": args.minimum_score,
        }
        orchestrator = WorkflowOrchestrator(output_root=args.output_root)
        state = orchestrator.run(
            args.request, overrides=overrides, persist=not args.no_persist,
            inject_failure=args.inject_failure,
        )
    _print_summary(state)
    return 0 if state.best_candidate is not None and state.status in {
        "completed", "partially_completed"
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
