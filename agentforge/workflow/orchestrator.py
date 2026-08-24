"""Deterministic multi-agent workflow orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentforge.agents import (
    CodeAgent, KnowledgeAgent, PersistenceAgent, PlannerAgent, ReportAgent,
    RepairAgent, RequirementAgent, ValidationAgent,
)
from agentforge.knowledge import KnowledgeGraphStore, KnowledgeRetriever
from agentforge.models import WorkflowState
from agentforge.models import ExecutionResult, ValidationCheck
from agentforge.repair import ErrorClassifier
from agentforge.validation import AstSecurityChecker, InterfaceChecker, SubprocessRunner
from agentforge.workflow.events import invoke_traced, record_skipped
from agentforge.config import LLMConfig
from agentforge.llm import create_llm_client
from agentforge.datasets import resolve_dataset_metadata


ROOT = Path(__file__).resolve().parents[2]


class WorkflowOrchestrator:
    def __init__(
        self,
        *,
        output_root: str | Path | None = None,
        graphml_path: str | Path | None = None,
        graph_json_path: str | Path | None = None,
        requirement_agent: RequirementAgent | None = None,
        validation_agent: ValidationAgent | None = None,
        llm_config: LLMConfig | None = None,
        llm_client=None,
    ) -> None:
        self.output_root = Path(output_root or ROOT / "outputs" / "runs")
        self.graphml_path = Path(graphml_path or ROOT / "knowledge" / "knowledge_graph.graphml")
        self.graph_json_path = Path(
            graph_json_path or self.graphml_path.with_suffix(".json")
        )
        self.requirement_agent = requirement_agent or RequirementAgent()
        self.validation_agent = validation_agent or ValidationAgent()
        self.planner_agent = PlannerAgent()
        self.code_agent = CodeAgent()
        self.report_agent = ReportAgent()
        self.persistence_agent = PersistenceAgent()
        self.security_checker = AstSecurityChecker()
        self.interface_checker = InterfaceChecker()
        self.subprocess_runner = SubprocessRunner()
        self.error_classifier = ErrorClassifier()
        self.llm_config = llm_config or LLMConfig()
        self.llm_client = llm_client

    @staticmethod
    def _track_llm(state, call) -> None:
        state.llm_calls.append(call.model_dump(mode="json"))
        state.llm_call_count += 1
        if call.status == "failed":
            state.llm_failures.append(f"{call.purpose}: {call.error_type}: {call.error_message}")

    def _fallback(self, state, purpose: str) -> None:
        state.llm_fallback_count += 1
        record_skipped(state, "LLMFallback", f"{purpose}: deterministic fallback selected.")

    def _new_run(self) -> tuple[str, Path]:
        while True:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_id = f"run-{stamp}-{uuid4().hex[:8]}"
            run_dir = self.output_root / run_id
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return run_id, run_dir

    def run(
        self,
        request_text: str,
        *,
        overrides: dict[str, Any] | None = None,
        persist: bool = True,
        inject_failure: str | None = None,
    ) -> WorkflowState:
        run_id, run_dir = self._new_run()
        state = WorkflowState(
            run_id=run_id, status="running", failure_injection=inject_failure,
            execution_mode=self.llm_config.mode, llm_provider="deterministic",
            llm_model=self.llm_config.model,
        )
        try:
            client = self.llm_client
            if self.llm_config.mode != "deterministic" and client is None:
                try:
                    client = create_llm_client(self.llm_config)
                except Exception as exc:
                    state.llm_failures.append(f"configuration: {type(exc).__name__}: {exc}")
                    if not self.llm_config.allow_fallback: raise
                    self._fallback(state, "configuration")
            self._active_llm_client = client
            if client is not None and self.llm_config.mode != "deterministic":
                parsed, call = invoke_traced(state, "LLMRequirementParser", "structured requirement",
                    self.requirement_agent.parse_with_llm, request_text, client, overrides)
                self._track_llm(state, call)
                if parsed is not None:
                    state.request, state.llm_provider = parsed, call.provider
                elif self.llm_config.allow_fallback:
                    self._fallback(state, "requirement_parsing")
                    state.request = self.requirement_agent.parse(request_text, overrides)
                else: raise RuntimeError(call.error_message or "LLM requirement parsing failed")
            else:
                state.request = invoke_traced(
                    state, "RequirementAgent", "natural-language request and explicit overrides",
                    self.requirement_agent.parse, request_text, overrides,
                )
            state.dataset_metadata = resolve_dataset_metadata(state.request.dataset_path)
            store = KnowledgeGraphStore.load_graphml(self.graphml_path)
            knowledge_agent = KnowledgeAgent(KnowledgeRetriever(store))
            state.retrieved_knowledge = invoke_traced(
                state, "KnowledgeAgent",
                f"task={state.request.task_type}, metric={state.request.primary_metric}",
                knowledge_agent.retrieve, state.request,
            )
            if client is not None and self.llm_config.mode != "deterministic":
                plans, call = invoke_traced(state, "LLMPlanner", "knowledge-grounded candidates",
                    self.planner_agent.plan_with_llm, state.request, state.retrieved_knowledge, client)
                self._track_llm(state, call)
                if plans is not None: state.candidate_plans, state.llm_provider = plans, call.provider
                elif self.llm_config.allow_fallback:
                    self._fallback(state, "candidate_planning")
                    state.candidate_plans = self.planner_agent.plan(state.request, state.retrieved_knowledge)
                else: raise RuntimeError(call.error_message or "LLM planning failed")
            else:
                state.candidate_plans = invoke_traced(
                    state, "PlannerAgent",
                    f"requested_candidates={len(state.request.candidate_algorithms)}",
                    self.planner_agent.plan, state.request, state.retrieved_knowledge,
                )
            for plan in state.candidate_plans:
                artifact = None
                if client is not None and self.llm_config.enable_code_generation:
                    artifact, call = invoke_traced(state, f"LLMCodeGenerator[{plan.algorithm}]",
                        f"plan_id={plan.plan_id}", self.code_agent.generate_with_llm,
                        plan, run_id, run_dir, client, attempt=0)
                    self._track_llm(state, call)
                    if artifact is None and self.llm_config.allow_fallback:
                        self._fallback(state, f"code_generation:{plan.algorithm}")
                    elif artifact is None: raise RuntimeError(call.error_message or "LLM code generation failed")
                if artifact is None:
                    artifact = invoke_traced(
                        state, f"CodeAgent[{plan.algorithm}]", f"plan_id={plan.plan_id}",
                        self.code_agent.generate, plan, run_id, run_dir,
                        attempt=0, failure_injection=inject_failure,
                    )
                state.generation_modes[plan.algorithm] = artifact.generator_mode
                state.generated_artifacts.append(artifact)
                if type(self.validation_agent) is ValidationAgent:
                    result, final_artifact = self._execute_with_repairs(
                        state, plan, artifact, run_id, run_dir, knowledge_agent,
                    )
                    if final_artifact.code_path != artifact.code_path:
                        state.generated_artifacts.append(final_artifact)
                else:
                    # Backward-compatible test/custom agent extension point from stage three.
                    result = invoke_traced(
                        state, f"ValidationAgent[{plan.algorithm}]",
                        f"algorithm={plan.algorithm}, dataset={Path(state.request.dataset_path).name}",
                        self.validation_agent.validate, artifact, state.request, plan,
                    )
                state.candidate_results.append(result)

            successful = [result for result in state.candidate_results if result.status == "completed"]
            if not successful:
                state.status = "failed"
            else:
                state.best_candidate = max(
                    successful,
                    key=lambda result: float(result.selection_metric_value or float("-inf")),
                )
                state.status = (
                    "completed" if len(successful) == len(state.candidate_results)
                    else "partially_completed"
                )
                self._check_best_stability(state)
            state.security_summary = {
                "checked_attempts": sum(len(result.attempts) for result in state.candidate_results),
                "blocking_findings": sum(
                    len([finding for finding in (result.security_result.findings
                                                 if result.security_result else [])
                         if finding.blocking])
                    for result in state.candidate_results
                ),
                "limitations": (
                    "Subprocess isolation and AST checks reduce accidental risk but do not "
                    "constitute a production-grade security sandbox."
                ),
            }
            state.final_report_paths = invoke_traced(
                state, "ReportAgent", f"candidate_results={len(state.candidate_results)}",
                self.report_agent.generate, state, run_dir,
            )
            if persist and successful:
                try:
                    state.knowledge_persisted = invoke_traced(
                        state, "PersistenceAgent", f"run_id={state.run_id}",
                        self.persistence_agent.persist, state,
                        graphml_path=self.graphml_path, json_path=self.graph_json_path,
                    )
                except Exception as exc:
                    state.errors.append(f"PersistenceAgent: {type(exc).__name__}: {exc}")
                    if state.status == "completed":
                        state.status = "partially_completed"
            else:
                record_skipped(
                    state, "PersistenceAgent",
                    "Persistence disabled." if not persist else "No successful candidate to persist.",
                )
            # Refresh both reports with final persistence status and trace.
            state.final_report_paths = self.report_agent.generate(state, run_dir)
            return state
        except Exception as exc:
            state.errors.append(f"{type(exc).__name__}: {exc}")
            state.status = "failed"
            try:
                state.final_report_paths = self.report_agent.generate(state, run_dir)
            except Exception as report_exc:
                state.errors.append(f"ReportAgent: {type(report_exc).__name__}: {report_exc}")
            return state

    def _execute_with_repairs(
        self, state, plan, artifact, run_id, run_dir, knowledge_agent,
    ):
        current = artifact
        attempts = []
        repair_records = []
        final_result = None
        repair_agent = RepairAgent(knowledge_agent.retriever, self.code_agent)
        for attempt in range(3):
            security = invoke_traced(
                state, f"SecurityChecker[{plan.algorithm}:attempt-{attempt}]",
                f"file=attempt-{attempt}/candidate.py",
                self.security_checker.check, current.code_path,
            )
            interface = invoke_traced(
                state, f"InterfaceChecker[{plan.algorithm}:attempt-{attempt}]",
                f"file=attempt-{attempt}/candidate.py",
                self.interface_checker.check, current.code_path,
            )
            if security.passed and interface.passed:
                execution = invoke_traced(
                    state, f"SubprocessRunner[{plan.algorithm}:attempt-{attempt}]",
                    f"timeout={state.request.max_runtime_seconds}s",
                    self.subprocess_runner.run, current, state.request.dataset_path,
                    timeout_seconds=state.request.max_runtime_seconds,
                )
            else:
                record_skipped(
                    state, f"SubprocessRunner[{plan.algorithm}:attempt-{attempt}]",
                    "Execution blocked by security or interface policy.",
                )
                execution = ExecutionResult(
                    attempt=attempt, command=[],
                    cwd=str(Path("generated") / plan.algorithm / f"attempt-{attempt}"),
                    result_json_path=current.result_output_path,
                    model_path=current.model_output_path, process_status="not_started",
                )
            result = invoke_traced(
                state, f"ValidationAgent[{plan.algorithm}:attempt-{attempt}]",
                f"checks for attempt-{attempt}", self.validation_agent.assess,
                current, state.request, plan, security, interface, execution,
            )
            attempts.append(execution)
            if repair_records and repair_records[-1].status == "created":
                repair_records[-1].validation_after = result.validation_checks
                repair_records[-1].status = "validated" if result.status == "completed" else "failed"
            result.attempts = list(attempts)
            result.repair_records = list(repair_records)
            final_result = result
            if result.status == "completed":
                result.final_code_path = current.code_path
                if repair_records:
                    result.failure_type = repair_records[0].failure_type
                    state.repaired_candidates.append(plan.algorithm)
                return result, current
            failure_type = invoke_traced(
                state, f"ErrorClassifier[{plan.algorithm}:attempt-{attempt}]",
                f"return_code={execution.return_code}, timed_out={execution.timed_out}",
                self.error_classifier.classify, execution=execution, security=security,
                interface=interface, checks=result.validation_checks,
            )
            result.failure_type = failure_type
            if attempt >= 2 or failure_type not in {
                "MissingValueError", "MissingInterface", "InvalidReturnFormat"
            }:
                state.unrepaired_failures.append(plan.algorithm)
                return result, current
            record = repaired = None
            llm_repair_fallback = False
            if getattr(self, "_active_llm_client", None) is not None and self.llm_config.enable_code_repair:
                record, repaired, call = invoke_traced(
                    state, f"LLMRepairer[{plan.algorithm}:attempt-{attempt + 1}]",
                    f"failure_type={failure_type}", repair_agent.repair_with_llm,
                    state.request, plan, current, failure_type, result.validation_checks,
                    self._active_llm_client, attempt=attempt + 1, run_id=run_id, run_dir=run_dir,
                    error_summary=result.error or "validation failure")
                self._track_llm(state, call)
                if record is None and self.llm_config.allow_fallback:
                    self._fallback(state, f"code_repair:{plan.algorithm}")
                    llm_repair_fallback = True
                elif record is None: raise RuntimeError(call.error_message or "LLM repair failed")
            if record is None:
                record, repaired = invoke_traced(
                    state, f"RepairAgent[{plan.algorithm}:attempt-{attempt + 1}]",
                    f"failure_type={failure_type}", repair_agent.repair,
                    state.request, plan, current, failure_type, result.validation_checks,
                    attempt=attempt + 1, run_id=run_id, run_dir=run_dir,
                    error_summary=result.error or "validation failure",
                )
                if llm_repair_fallback:
                    record = record.model_copy(update={"fallback_used": True})
            repair_records.append(record)
            state.repair_modes.append(record.repair_mode)
            state.total_repair_attempts += 1
            current = repaired
        return final_result, current

    def _check_best_stability(self, state: WorkflowState) -> None:
        if state.best_candidate is None or not state.best_candidate.final_code_path:
            return
        artifact = next(
            item for item in reversed(state.generated_artifacts)
            if item.algorithm == state.best_candidate.algorithm
            and item.code_path == state.best_candidate.final_code_path
        )
        stability_artifact = artifact.model_copy(update={
            "model_output_path": str(Path(artifact.model_output_path).with_name("stability-model.pkl")),
            "result_output_path": str(Path(artifact.result_output_path).with_name("stability-result.json")),
        })
        execution = invoke_traced(
            state, f"SubprocessRunner[{artifact.algorithm}:stability]",
            "repeat best candidate with fixed seed",
            self.subprocess_runner.run, stability_artifact, state.request.dataset_path,
            timeout_seconds=state.request.max_runtime_seconds,
        )
        passed = False
        if execution.process_status == "completed":
            import json
            payload = json.loads(Path(execution.result_json_path).read_text(encoding="utf-8"))
            evaluation = payload["evaluation"]
            passed = (
                evaluation["validation_metrics"][state.request.primary_metric]
                == state.best_candidate.validation_metrics[state.request.primary_metric]
                and evaluation["test_metrics"]["f1"] == state.best_candidate.test_metrics["f1"]
            )
        state.best_candidate.validation_checks.append(ValidationCheck(
            name="fixed_seed_stability", category="stability", passed=passed,
            expected="identical validation selection metric and test F1",
            actual=execution.process_status,
            message="Only the final best candidate is repeated for stability.",
        ))
        if not passed and state.status == "completed":
            state.status = "partially_completed"
