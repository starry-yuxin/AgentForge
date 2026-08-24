"""Human- and machine-readable workflow reporting."""

from __future__ import annotations

from pathlib import Path

from agentforge.models import WorkflowState


class ReportAgent:
    def generate(self, state: WorkflowState, run_dir: str | Path) -> dict[str, str]:
        report_dir = Path(run_dir) / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / "workflow_report.json"
        markdown_path = report_dir / "workflow_report.md"
        json_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

        request = state.request
        knowledge = state.retrieved_knowledge
        lines = [
            "# AgentForge deterministic workflow report", "", f"- Run ID: `{state.run_id}`",
            f"- Status: `{state.status}`", f"- Original request: {request.raw_text if request else ''}",
            f"- Generator mode: `deterministic_template`", "",
            "## Structured requirement", "",
            f"- Task: `{request.task_type}`", f"- Dataset: `{request.dataset_path}`",
            f"- Target: `{request.target_column}`", f"- Primary metric: `{request.primary_metric}`",
            f"- Minimum score: `{request.minimum_score}`", "",
            "### Field sources", "",
        ]
        lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(request.field_sources.items()))
        lines.extend(["", "## Retrieved knowledge", "",
                      knowledge.retrieval_summary if knowledge else "No knowledge retrieved."])
        if knowledge:
            for item in [*knowledge.algorithms, *knowledge.preprocessors,
                         *knowledge.metrics, *knowledge.failure_experiences]:
                lines.append(
                    f"- **{item.name}** (`{item.capability_id}`): "
                    f"{'; '.join(item.match_reasons)} — {item.source_document} / {item.source_section}"
                )
        lines.extend(["", "## Candidate plans and results", ""])
        by_plan = {result.plan_id: result for result in state.candidate_results}
        artifacts = {artifact.plan_id: artifact for artifact in state.generated_artifacts}
        for plan in state.candidate_plans:
            result = by_plan.get(plan.plan_id)
            artifact = artifacts.get(plan.plan_id)
            lines.extend([
                f"### {plan.algorithm}", "", f"- Rationale: {plan.rationale}",
                f"- Preprocessing: {', '.join(plan.preprocessing_steps)}",
                f"- Supporting capabilities: {', '.join(plan.supporting_capability_ids)}",
                f"- Generated code: `{artifact.code_path if artifact else ''}`",
                f"- Validation metrics: `{result.validation_metrics if result else {}}`",
                f"- Test metrics: `{result.test_metrics if result else {}}`",
                f"- Selected threshold: `{result.selected_threshold if result else None}`",
                f"- Minimum score met: `{result.minimum_score_met if result else False}`", "",
            ])
        lines.extend([
            "## Selection", "",
            f"- Best candidate: `{state.best_candidate.algorithm if state.best_candidate else None}`",
            f"- Selection metric on validation: "
            f"`{state.best_candidate.selection_metric_value if state.best_candidate else None}`",
            f"- Final metrics on test: `{state.best_candidate.test_metrics if state.best_candidate else {}}`",
            "", "## Execution trace", "",
        ])
        for event in state.events:
            lines.append(
                f"- {event.agent_name}: {event.status} ({event.duration_seconds:.6f}s) — {event.message}"
            )
        lines.extend([
            "", "## Persistence and limitations", "",
            f"- Knowledge persisted: `{state.knowledge_persisted}`",
            "- Generated code uses deterministic trusted templates; it is not LLM-generated.",
            "- No subprocess sandbox, AST security policy, automatic repair, real LLM, SQLite, or Web UI.",
            "- Metrics use fixed-random-seed synthetic data and do not demonstrate real-business generalization.",
        ])
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"json": str(json_path), "markdown": str(markdown_path)}
