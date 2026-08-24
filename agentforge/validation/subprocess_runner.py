"""Timeout-bounded subprocess execution with a minimal explicit environment."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from agentforge.models import ExecutionResult, GeneratedArtifact


ROOT = Path(__file__).resolve().parents[2]
LOG_LIMIT = 8000


class SubprocessRunner:
    def __init__(self, python_executable: str | None = None) -> None:
        self.python_executable = python_executable or sys.executable

    def run(
        self,
        artifact: GeneratedArtifact,
        data_path: str | Path,
        *,
        timeout_seconds: float,
    ) -> ExecutionResult:
        attempt_dir = Path(artifact.code_path).parent
        attempt_dir.mkdir(parents=True, exist_ok=True)
        actual_command = [
            self.python_executable, "-m", "agentforge.validation.harness",
            "--candidate", str(Path(artifact.code_path).resolve()),
            "--data", str(Path(data_path).resolve()),
            "--model-output", str(Path(artifact.model_output_path).resolve()),
            "--result-output", str(Path(artifact.result_output_path).resolve()),
        ]
        safe_command = [
            "<venv-python>", "-m", "agentforge.validation.harness",
            "--candidate", Path(artifact.code_path).name,
            "--data", Path(data_path).name,
            "--model-output", Path(artifact.model_output_path).name,
            "--result-output", Path(artifact.result_output_path).name,
        ]
        environment = {
            "PATH": str(Path(self.python_executable).resolve().parent),
            "PYTHONPATH": str(ROOT),
            "PYTHONHASHSEED": "0",
        }
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                actual_command, cwd=attempt_dir, env=environment,
                capture_output=True, text=True, timeout=timeout_seconds,
                check=False, shell=False,
            )
            duration = time.perf_counter() - started
            stdout, stderr = completed.stdout, completed.stderr
            (attempt_dir / "stdout.log").write_text(stdout, encoding="utf-8")
            (attempt_dir / "stderr.log").write_text(stderr, encoding="utf-8")
            result_exists = Path(artifact.result_output_path).is_file()
            model_exists = Path(artifact.model_output_path).is_file()
            status = "completed" if completed.returncode == 0 and result_exists and model_exists else "failed"
            return ExecutionResult(
                attempt=artifact.attempt, command=safe_command,
                cwd=str(Path("generated") / artifact.algorithm / f"attempt-{artifact.attempt}"),
                return_code=completed.returncode, duration_seconds=duration,
                stdout=stdout[-LOG_LIMIT:], stderr=stderr[-LOG_LIMIT:],
                result_json_path=artifact.result_output_path,
                model_path=artifact.model_output_path, process_status=status,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - started
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            (attempt_dir / "stdout.log").write_text(stdout, encoding="utf-8")
            (attempt_dir / "stderr.log").write_text(stderr, encoding="utf-8")
            return ExecutionResult(
                attempt=artifact.attempt, command=safe_command,
                cwd=str(Path("generated") / artifact.algorithm / f"attempt-{artifact.attempt}"),
                return_code=None, timed_out=True, duration_seconds=duration,
                stdout=stdout[-LOG_LIMIT:], stderr=stderr[-LOG_LIMIT:],
                result_json_path=artifact.result_output_path,
                model_path=artifact.model_output_path, process_status="timed_out",
            )
