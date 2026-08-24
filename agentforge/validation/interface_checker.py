"""AST-only validation of the generated candidate interface."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from agentforge.models import InterfaceCheckResult


EXPECTED = {
    "train": (["data_path", "model_path"], "dict"),
    "predict": (["model_path", "data_path"], "list"),
    "evaluate": (["model_path", "data_path"], "dict"),
}


class InterfaceChecker:
    def check(self, path: str | Path) -> InterfaceCheckResult:
        source = Path(path).read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return InterfaceCheckResult(
                passed=False, required_functions=list(EXPECTED), discovered_functions=[],
                missing_functions=list(EXPECTED), signature_errors=[str(exc)],
                messages=["File has invalid Python syntax."],
            )
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        counts = Counter(node.name for node in functions)
        discovered = [node.name for node in functions]
        missing = [name for name in EXPECTED if counts[name] == 0]
        errors = []
        for name, count in counts.items():
            if name in EXPECTED and count > 1:
                errors.append(f"duplicate function definition: {name}")
        for node in functions:
            if node.name not in EXPECTED or counts[node.name] > 1:
                continue
            expected_args, expected_return = EXPECTED[node.name]
            positional = [arg.arg for arg in [*node.args.posonlyargs, *node.args.args]]
            if positional != expected_args:
                errors.append(
                    f"{node.name} parameters must be {expected_args}, found {positional}"
                )
            if node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
                errors.append(f"{node.name} cannot define additional required or variadic parameters")
            if node.returns is not None:
                annotation = ast.unparse(node.returns)
                if annotation not in {expected_return, f"builtins.{expected_return}"}:
                    errors.append(
                        f"{node.name} return annotation must be {expected_return}, found {annotation}"
                    )
        passed = not missing and not errors
        return InterfaceCheckResult(
            passed=passed, required_functions=list(EXPECTED),
            discovered_functions=discovered, missing_functions=missing,
            signature_errors=errors,
            messages=["Interface contract satisfied."] if passed else ["Interface contract failed."],
        )
