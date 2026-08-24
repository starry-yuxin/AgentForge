"""Constrained deterministic source-code generation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from agentforge.models import CandidatePlan, GeneratedArtifact
from agentforge.llm.parsing import strip_python_fence
from agentforge.llm.prompts import CODE


TEMPLATE = '''"""Generated deterministic interface for {algorithm}."""

from agentforge.generated_runtime import evaluate_candidate_model, predict_candidate, {train_function}

ALGORITHM = "{algorithm}"
HYPERPARAMETERS = {hyperparameters!r}


def train(data_path: str, model_path: str) -> dict:
    return {train_function}(ALGORITHM, data_path, model_path, HYPERPARAMETERS)


def predict(model_path: str, data_path: str) -> list:
    return predict_candidate(model_path, data_path)


def evaluate(model_path: str, data_path: str) -> dict:
    return evaluate_candidate_model(model_path, data_path)
'''


class CodeAgent:
    def generate(
        self, plan: CandidatePlan, run_id: str, run_dir: str | Path,
        *, attempt: int = 0, failure_injection: str | None = None,
    ) -> GeneratedArtifact:
        root = Path(run_dir)
        generated_dir = root / "generated" / plan.algorithm / f"attempt-{attempt}"
        model_dir = root / "models" / plan.algorithm / f"attempt-{attempt}"
        result_dir = root / "results" / plan.algorithm / f"attempt-{attempt}"
        for directory in (generated_dir, model_dir, result_dir):
            directory.mkdir(parents=True, exist_ok=True)
        code_path = generated_dir / "candidate.py"
        train_function = (
            "train_candidate_missing_imputer"
            if failure_injection == "missing_imputer" and plan.algorithm == "logistic_regression"
            else "train_candidate"
        )
        source = TEMPLATE.format(
            algorithm=plan.algorithm, hyperparameters=plan.hyperparameters,
            train_function=train_function,
        )
        ast.parse(source)
        code_path.write_text(source, encoding="utf-8")
        return GeneratedArtifact(
            plan_id=plan.plan_id, algorithm=plan.algorithm, code_path=str(code_path),
            model_output_path=str(model_dir / f"{plan.algorithm}.pkl"),
            result_output_path=str(result_dir / f"{plan.algorithm}.json"),
            interface_spec=plan.expected_interfaces, source_plan=plan, syntax_valid=True,
            attempt=attempt, failure_injection=failure_injection,
        )

    def generate_with_llm(self, plan, run_id, run_dir, client, *, attempt=0):
        call = client.call(purpose="code_generation", prompt_version=CODE.version,
            system=CODE.system, user="Generate Python source for this validated plan:\n" +
            json.dumps(plan.model_dump(mode="json"), ensure_ascii=False))
        if call.status != "success": return None, call
        source = strip_python_fence(call.response_text)
        try:
            ast.parse(source)
        except SyntaxError as exc:
            call.status, call.error_type, call.error_message = "failed", "SyntaxError", str(exc)
            return None, call
        root = Path(run_dir)
        generated_dir = root / "generated" / plan.algorithm / f"attempt-{attempt}"
        model_dir = root / "models" / plan.algorithm / f"attempt-{attempt}"
        result_dir = root / "results" / plan.algorithm / f"attempt-{attempt}"
        for directory in (generated_dir, model_dir, result_dir): directory.mkdir(parents=True, exist_ok=True)
        code_path = generated_dir / "candidate.py"
        code_path.write_text(source, encoding="utf-8")
        artifact = GeneratedArtifact(plan_id=plan.plan_id, algorithm=plan.algorithm,
            code_path=str(code_path), model_output_path=str(model_dir / f"{plan.algorithm}.pkl"),
            result_output_path=str(result_dir / f"{plan.algorithm}.json"),
            generator_mode="llm", interface_spec=plan.expected_interfaces, source_plan=plan,
            syntax_valid=True, attempt=attempt, llm_call_id=call.call_id,
            prompt_version=CODE.version)
        return artifact, call
