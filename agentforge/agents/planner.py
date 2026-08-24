"""Knowledge-grounded deterministic candidate planning."""

from __future__ import annotations

from uuid import uuid4

from agentforge.models import AlgorithmRequirement, CandidatePlan, RetrievedKnowledge
from agentforge.llm.parsing import extract_json
from agentforge.llm.prompts import PLANNER
import json


ALGORITHM_REGISTRY = {
    "logistic_regression": {
        "steps": ["numerical_imputation", "categorical_imputation", "one_hot_encoding",
                  "standard_scaling", "class_weight_balancing",
                  "validation_threshold_optimization"],
        "hyperparameters": {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
        "rationale": "Interpretable linear baseline with scaled numeric features.",
    },
    "random_forest": {
        "steps": ["numerical_imputation", "categorical_imputation", "one_hot_encoding",
                  "class_weight_balancing", "validation_threshold_optimization"],
        "hyperparameters": {"n_estimators": 240, "max_depth": 10, "min_samples_leaf": 3,
                            "max_features": "sqrt", "class_weight": "balanced_subsample",
                            "random_state": 42, "n_jobs": 1},
        "rationale": "Tree ensemble for nonlinear tabular relationships without scaling.",
    },
}


class PlannerAgent:
    def plan(
        self, requirement: AlgorithmRequirement, knowledge: RetrievedKnowledge
    ) -> list[CandidatePlan]:
        supported_by_knowledge = {item.capability_id for item in knowledge.algorithms}
        capability_ids = {
            item.capability_id for group in (
                knowledge.algorithms, knowledge.preprocessors, knowledge.metrics,
                knowledge.failure_experiences,
            ) for item in group
        }
        plans = []
        for algorithm in requirement.candidate_algorithms:
            if algorithm not in ALGORITHM_REGISTRY or algorithm not in supported_by_knowledge:
                continue
            spec = ALGORITHM_REGISTRY[algorithm]
            supporting = [algorithm, *[step for step in spec["steps"] if step in capability_ids]]
            metric_ids = [item.capability_id for item in knowledge.metrics]
            supporting.extend(item for item in metric_ids if item not in supporting)
            plans.append(CandidatePlan(
                plan_id=f"plan-{algorithm}-{uuid4().hex[:8]}", algorithm=algorithm,
                preprocessing_steps=spec["steps"], hyperparameters=spec["hyperparameters"],
                threshold_strategy="select on validation split only",
                evaluation_metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
                rationale=spec["rationale"] + " Knowledge retrieval supports this candidate.",
                supporting_capability_ids=supporting,
                expected_interfaces=requirement.required_interfaces,
            ))
        if not plans:
            raise LookupError("no candidate is supported by both knowledge and algorithm registry")
        return plans

    def plan_with_llm(self, requirement, knowledge, client):
        allowed = [a for a in requirement.candidate_algorithms if a in ALGORITHM_REGISTRY]
        call = client.call(purpose="candidate_planning", prompt_version=PLANNER.version,
            system=PLANNER.system,
            user=("Return a JSON object with key algorithms, an ordered list chosen only from "
                  f"{allowed}. Do not invent algorithms or capabilities. Context: "
                  f"{json.dumps([x.capability_id for x in knowledge.algorithms])}"))
        if call.status != "success": return None, call
        try:
            payload = extract_json(call.response_text)
            algorithms = payload["algorithms"]
            if not isinstance(algorithms, list) or not algorithms or any(a not in allowed for a in algorithms):
                raise ValueError("planner returned unsupported algorithms")
            narrowed = requirement.model_copy(update={"candidate_algorithms": list(dict.fromkeys(algorithms))})
            plans = self.plan(narrowed, knowledge)
            call.parsed_output = [p.model_dump(mode="json") for p in plans]
            return plans, call
        except Exception as exc:
            call.status, call.error_type, call.error_message = "failed", type(exc).__name__, str(exc)[:500]
            return None, call
