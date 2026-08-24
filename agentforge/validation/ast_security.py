"""Lightweight AST policy for generated code; not a production sandbox."""

from __future__ import annotations

import ast
from pathlib import Path

from agentforge.models import SecurityCheckResult, SecurityFinding


BLOCKED_IMPORT_ROOTS = {"socket", "requests", "urllib", "http", "subprocess"}
BLOCKED_CALLS = {
    "eval": "AF001", "exec": "AF002", "compile": "AF003", "__import__": "AF004",
    "os.system": "AF010", "os.popen": "AF011", "subprocess.run": "AF012",
    "subprocess.Popen": "AF013", "subprocess.call": "AF014",
    "shutil.rmtree": "AF020", "os.remove": "AF021", "os.unlink": "AF022",
    "os.rmdir": "AF023", "Path.unlink": "AF024", "pathlib.Path.unlink": "AF024",
    "importlib.import_module": "AF030",
}


def _symbol(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _symbol(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


class AstSecurityChecker:
    def check(self, path: str | Path) -> SecurityCheckResult:
        source_path = Path(path)
        findings: list[SecurityFinding] = []
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        except SyntaxError as exc:
            findings.append(SecurityFinding(
                rule_id="AF000", severity="critical", category="syntax",
                message=str(exc), line_number=exc.lineno, symbol=None, blocking=True,
            ))
            return SecurityCheckResult(
                passed=False, findings=findings, checked_file=str(source_path)
            )
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    aliases[alias.asname or root] = alias.name
                    if root in BLOCKED_IMPORT_ROOTS:
                        findings.append(SecurityFinding(
                            rule_id="AF040", severity="critical", category="network_or_process",
                            message=f"blocked import: {alias.name}", line_number=node.lineno,
                            symbol=alias.name,
                        ))
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                if root in BLOCKED_IMPORT_ROOTS:
                    findings.append(SecurityFinding(
                        rule_id="AF040", severity="critical", category="network_or_process",
                        message=f"blocked import: {node.module}", line_number=node.lineno,
                        symbol=node.module,
                    ))
            elif isinstance(node, ast.Call):
                symbol = _symbol(node.func)
                first = symbol.split(".")[0]
                resolved = symbol.replace(first, aliases.get(first, first), 1)
                rule = BLOCKED_CALLS.get(resolved) or BLOCKED_CALLS.get(symbol)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "unlink":
                    rule = "AF024"
                if rule:
                    findings.append(SecurityFinding(
                        rule_id=rule, severity="critical", category="dangerous_call",
                        message=f"blocked call: {resolved}", line_number=node.lineno,
                        symbol=resolved,
                    ))
                for keyword in node.keywords:
                    if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) \
                            and keyword.value.value is True:
                        findings.append(SecurityFinding(
                            rule_id="AF050", severity="critical", category="shell",
                            message="shell=True is forbidden", line_number=node.lineno,
                            symbol=resolved or symbol,
                        ))
                if resolved in {"open", "Path", "pathlib.Path"} and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str) \
                            and Path(argument.value).is_absolute():
                        findings.append(SecurityFinding(
                            rule_id="AF060", severity="high", category="absolute_path",
                            message="repository-external absolute paths are forbidden in generated code",
                            line_number=node.lineno, symbol=resolved,
                        ))
        return SecurityCheckResult(
            passed=not any(item.blocking for item in findings), findings=findings,
            checked_file=str(source_path),
        )
