"""Constrained deterministic source-code generation."""

from __future__ import annotations

import ast
from pathlib import Path

from agentforge.models import CandidatePlan, GeneratedArtifact


TEMPLATE = '''"""Generated deterministic interface for {algorithm}."""

from agentforge.generated_runtime import evaluate_candidate_model, predict_candidate, train_candidate

ALGORITHM = "{algorithm}"
HYPERPARAMETERS = {hyperparameters!r}


def train(data_path: str, model_path: str) -> dict:
    return train_candidate(ALGORITHM, data_path, model_path)


def predict(model_path: str, data_path: str) -> list:
    return predict_candidate(model_path, data_path)


def evaluate(model_path: str, data_path: str) -> dict:
    return evaluate_candidate_model(model_path, data_path)
'''


class CodeAgent:
    def generate(self, plan: CandidatePlan, run_id: str, run_dir: str | Path) -> GeneratedArtifact:
        root = Path(run_dir)
        generated_dir = root / "generated"
        model_dir = root / "models"
        result_dir = root / "results"
        for directory in (generated_dir, model_dir, result_dir):
            directory.mkdir(parents=True, exist_ok=True)
        code_path = generated_dir / f"{plan.algorithm}.py"
        source = TEMPLATE.format(
            algorithm=plan.algorithm, hyperparameters=plan.hyperparameters
        )
        ast.parse(source)
        code_path.write_text(source, encoding="utf-8")
        return GeneratedArtifact(
            plan_id=plan.plan_id, algorithm=plan.algorithm, code_path=str(code_path),
            model_output_path=str(model_dir / f"{plan.algorithm}.pkl"),
            result_output_path=str(result_dir / f"{plan.algorithm}.json"),
            interface_spec=plan.expected_interfaces, source_plan=plan, syntax_valid=True,
        )
