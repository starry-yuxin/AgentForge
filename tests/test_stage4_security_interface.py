from __future__ import annotations

from pathlib import Path

import pytest

from agentforge.validation import AstSecurityChecker, InterfaceChecker


VALID = '''
def train(data_path: str, model_path: str) -> dict:
    return {}
def predict(model_path: str, data_path: str) -> list:
    return []
def evaluate(model_path: str, data_path: str) -> dict:
    return {}
'''


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "candidate.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_normal_template_passes_security_and_interface(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID)
    assert AstSecurityChecker().check(path).passed
    assert InterfaceChecker().check(path).passed


@pytest.mark.parametrize(("source", "rule"), [
    ("eval('1')", "AF001"),
    ("exec('x=1')", "AF002"),
    ("import os\nos.system('echo unsafe')", "AF010"),
    ("import subprocess\nsubprocess.run(['x'])", "AF040"),
    ("from pathlib import Path\nPath('x').unlink()", "AF024"),
    ("import os\nos.remove('x')", "AF021"),
    ("import requests\nrequests.get('https://example.com')", "AF040"),
    ("import socket\nsocket.socket()", "AF040"),
    ("__import__('os')", "AF004"),
])
def test_dangerous_constructs_are_blocked(tmp_path: Path, source: str, rule: str) -> None:
    result = AstSecurityChecker().check(_write(tmp_path, source))
    assert not result.passed
    assert rule in {finding.rule_id for finding in result.findings}
    assert all(finding.line_number for finding in result.findings)


def test_syntax_error_is_blocking(tmp_path: Path) -> None:
    result = AstSecurityChecker().check(_write(tmp_path, "def broken(:\n"))
    assert not result.passed
    assert result.findings[0].rule_id == "AF000"


@pytest.mark.parametrize(("source", "message"), [
    (VALID.replace("def train", "def fit"), "train"),
    (VALID.replace("model_path: str, data_path: str", "data_path: str, model_path: str", 1),
     "parameters"),
    (VALID.replace("data_path: str, model_path: str", "data_path: str, model_path: str, extra", 1),
     "parameters"),
    (VALID + "\ndef train(data_path: str, model_path: str) -> dict:\n    return {}\n", "duplicate"),
])
def test_interface_contract_failures(tmp_path: Path, source: str, message: str) -> None:
    result = InterfaceChecker().check(_write(tmp_path, source))
    assert not result.passed
    combined = " ".join([*result.missing_functions, *result.signature_errors])
    assert message in combined
