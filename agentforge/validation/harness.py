"""Trusted subprocess harness that loads and invokes a generated candidate module."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("agentforge_generated_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(candidate: Path, data: Path, model_output: Path, result_output: Path) -> dict:
    module = _load_candidate(candidate)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    result_output.parent.mkdir(parents=True, exist_ok=True)
    train_result = module.train(str(data), str(model_output))
    predictions = module.predict(str(model_output), str(data))
    evaluation = module.evaluate(str(model_output), str(data))
    prediction_path = result_output.with_name("predictions.json")
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
    payload = {
        "schema_version": "1.0", "train_result": train_result,
        "prediction_count": len(predictions),
        "prediction_labels": sorted(set(predictions), key=str),
        "prediction_output": str(prediction_path), "evaluation": evaluation,
    }
    result_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "success", "algorithm": evaluation.get("algorithm"),
        "prediction_count": len(predictions),
    }))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    execute(
        args.candidate.resolve(), args.data.resolve(), args.model_output.resolve(),
        args.result_output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
