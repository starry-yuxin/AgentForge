"""Versioned prompts for narrow, auditable AgentForge tasks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str


POLICY = (
    "AgentForge supports tabular binary classification only. Return only the requested JSON or "
    "Python source; never claim unimplemented capabilities. Generated code must expose train, "
    "predict and evaluate and must not use network access, subprocesses, eval/exec, destructive "
    "file operations, or absolute paths. Do not reveal private reasoning."
)
REQUIREMENT = Prompt("requirement_parser", "requirement_parser_v1", POLICY)
PLANNER = Prompt("candidate_planner", "candidate_planner_v1", POLICY)
CODE = Prompt("code_generator", "code_generator_v1", POLICY)
ERROR = Prompt("error_analyzer", "error_analyzer_v1", POLICY)
REPAIR = Prompt("code_repair", "code_repair_v1", POLICY)
