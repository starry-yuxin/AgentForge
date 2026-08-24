"""Deterministic natural-language requirement parsing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentforge.models import AlgorithmRequirement
from agentforge.llm.parsing import extract_json
from agentforge.llm.prompts import REQUIREMENT


ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "task_type": "binary_classification",
    "industry": "customer_churn",
    "dataset_path": str(ROOT / "data" / "churn_sample.csv"),
    "target_column": "churn",
    "primary_metric": "f1",
    "minimum_score": 0.60,
    "candidate_algorithms": ["logistic_regression", "random_forest"],
    "data_characteristics": [
        "missing_values", "categorical_features", "imbalanced_data", "numerical_features"
    ],
    "constraints": ["train_validation_test_isolation", "validation_only_threshold"],
    "required_interfaces": ["train", "predict", "evaluate"],
    "max_runtime_seconds": 120.0,
}


class RequirementAgent:
    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path else ROOT / "configs" / "default.json"
        self.project_config = json.loads(path.read_text(encoding="utf-8"))

    def parse(self, text: str, overrides: dict[str, Any] | None = None) -> AlgorithmRequirement:
        if not text.strip():
            raise ValueError("request text cannot be empty")
        lowered = text.lower()
        parsed: dict[str, Any] = {}
        sources = {key: "default_config" for key in DEFAULTS}

        supported_task = any(token in lowered for token in (
            "客户流失", "customer churn", "binary_classification", "binary classification"
        ))
        unsupported_task = any(token in lowered for token in (
            "图像分类", "image classification", "文本分类", "text classification", "regression task"
        ))
        if unsupported_task and not supported_task:
            parsed["task_type"] = "unsupported"
            sources["task_type"] = "deterministic_parser"
        elif supported_task:
            parsed["task_type"] = "binary_classification"
            sources["task_type"] = "user_input"

        algorithms = []
        if re.search(r"logistic[ _-]?regression|逻辑回归", lowered):
            algorithms.append("logistic_regression")
        if re.search(r"random[ _-]?forest|随机森林", lowered):
            algorithms.append("random_forest")
        if algorithms:
            parsed["candidate_algorithms"] = algorithms
            sources["candidate_algorithms"] = "user_input"

        metric_patterns = [
            ("roc_auc", r"roc[ _-]?auc"), ("f1", r"(?<![a-z0-9_])f1(?![a-z0-9_])"),
            ("accuracy", r"accuracy|准确率"), ("precision", r"precision|精确率"),
            ("recall", r"recall|召回率"),
        ]
        for metric, pattern in metric_patterns:
            if re.search(pattern, lowered):
                parsed["primary_metric"] = metric
                sources["primary_metric"] = "user_input"
                break
        explicit_metric = re.search(
            r"(?:primary metric|主要指标)(?:\s+is|\s*为|\s*[:：=])?\s*([a-z0-9_-]+)",
            lowered,
        )
        if explicit_metric and "primary_metric" not in parsed:
            parsed["primary_metric"] = explicit_metric.group(1).replace("-", "_")
            sources["primary_metric"] = "deterministic_parser"

        score_match = re.search(
            r"(?:最低要求(?:为)?|minimum(?:\s+\w+)?(?:\s+of)?|f1\s*>=)\s*([0-9]+(?:\.[0-9]+)?)",
            lowered,
        )
        if score_match:
            parsed["minimum_score"] = float(score_match.group(1))
            sources["minimum_score"] = "user_input"

        runtime_match = re.search(
            r"(?:最大运行时间|max(?:imum)? runtime(?: seconds)?)\s*(?:为|[:=]|of)?\s*([0-9]+(?:\.[0-9]+)?)",
            lowered,
        )
        if runtime_match:
            parsed["max_runtime_seconds"] = float(runtime_match.group(1))
            sources["max_runtime_seconds"] = "user_input"

        dataset_match = re.search(
            r"(?:dataset(?:\s+path)?|数据集(?:路径)?)(?:\s+is|\s*为|\s*[:：=])?\s*[\"']?([^\s\"']+\.csv)",
            text,
            flags=re.IGNORECASE,
        )
        if dataset_match:
            parsed["dataset_path"] = dataset_match.group(1)
            sources["dataset_path"] = "user_input"

        target_match = re.search(
            r"(?:target(?:\s+column)?|目标列)(?:\s+is|\s*为|\s*[:：=])?\s*[\"']?([a-zA-Z_][\w-]*)",
            text,
            flags=re.IGNORECASE,
        )
        if target_match:
            parsed["target_column"] = target_match.group(1)
            sources["target_column"] = "user_input"

        requested_interfaces = [
            name for name in ("train", "predict", "evaluate")
            if re.search(rf"\b{name}\b", lowered)
        ]
        if len(requested_interfaces) == 3:
            parsed["required_interfaces"] = requested_interfaces
            sources["required_interfaces"] = "user_input"

        characteristic_patterns = {
            "missing_values": r"missing values?|缺失值",
            "categorical_features": r"categorical features?|类别特征",
            "imbalanced_data": r"imbalanced data|class imbalance|类别不平衡",
            "numerical_features": r"numerical features?|数值特征",
        }
        found_characteristics = [
            key for key, pattern in characteristic_patterns.items() if re.search(pattern, lowered)
        ]
        if found_characteristics:
            parsed["data_characteristics"] = found_characteristics
            sources["data_characteristics"] = "user_input"

        configured_defaults = dict(DEFAULTS)
        configured_defaults["target_column"] = self.project_config.get(
            "target_column", configured_defaults["target_column"]
        )
        configured_defaults["primary_metric"] = self.project_config.get(
            "primary_metric", configured_defaults["primary_metric"]
        )
        values = {**configured_defaults, **parsed}
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    if key not in DEFAULTS:
                        raise ValueError(f"unknown requirement override: {key}")
                    values[key] = value
                    sources[key] = "explicit_override"
        values.update({
            "request_id": f"req-{uuid4().hex[:10]}",
            "raw_text": text,
            "field_sources": sources,
        })
        return AlgorithmRequirement.model_validate(values)

    def parse_with_llm(self, text: str, client, overrides=None):
        call = client.call(purpose="requirement_parsing", prompt_version=REQUIREMENT.version,
            system=REQUIREMENT.system,
            user=("Extract a JSON object containing only: task_type, industry, dataset_path, "
                  "target_column, primary_metric, minimum_score, candidate_algorithms, "
                  "data_characteristics, constraints, required_interfaces, max_runtime_seconds.\n"
                  f"Request: {text}"))
        if call.status != "success":
            return None, call
        try:
            payload = extract_json(call.response_text)
            if not isinstance(payload, dict): raise ValueError("requirement output must be an object")
            unknown = set(payload) - set(DEFAULTS)
            if unknown: raise ValueError(f"unknown requirement fields: {sorted(unknown)}")
            values = {**DEFAULTS, **payload}
            sources = {key: ("llm" if key in payload else "default_config") for key in DEFAULTS}
            for key, value in (overrides or {}).items():
                if value is not None:
                    if key not in DEFAULTS: raise ValueError(f"unknown requirement override: {key}")
                    values[key], sources[key] = value, "explicit_override"
            dataset = Path(values["dataset_path"])
            if not dataset.is_absolute(): dataset = (ROOT / dataset).resolve()
            if not dataset.is_file(): raise ValueError("LLM-selected dataset does not exist")
            values["dataset_path"] = str(dataset)
            values.update(request_id=f"req-{uuid4().hex[:10]}", raw_text=text, field_sources=sources)
            result = AlgorithmRequirement.model_validate(values)
            call.parsed_output = result.model_dump(mode="json")
            return result, call
        except Exception as exc:
            call.status, call.error_type, call.error_message = "failed", type(exc).__name__, str(exc)[:500]
            return None, call
