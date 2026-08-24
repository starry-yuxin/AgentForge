"""Event creation helpers with UTC timestamps and monotonic durations."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from agentforge.models import WorkflowEvent, WorkflowState


def invoke_traced(
    state: WorkflowState,
    agent_name: str,
    input_summary: str,
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    event_prefix = f"event-{uuid4().hex[:12]}"
    state.events.append(WorkflowEvent(
        event_id=f"{event_prefix}-started", agent_name=agent_name, status="started",
        started_at=started_at, input_summary=input_summary,
        message=f"{agent_name} started.",
    ))
    try:
        output = function(*args, **kwargs)
    except Exception as exc:
        finished_at = datetime.now(timezone.utc)
        duration = time.perf_counter() - started
        state.events.append(WorkflowEvent(
            event_id=f"{event_prefix}-failed", agent_name=agent_name, status="failed",
            started_at=started_at, finished_at=finished_at, duration_seconds=duration,
            input_summary=input_summary,
            output_summary=f"{type(exc).__name__}",
            message=f"{type(exc).__name__}: {exc}",
        ))
        raise
    finished_at = datetime.now(timezone.utc)
    duration = time.perf_counter() - started
    if isinstance(output, list):
        output_summary = f"items={len(output)}"
    elif isinstance(output, bool):
        output_summary = f"success={output}"
    else:
        output_summary = type(output).__name__
    state.events.append(WorkflowEvent(
        event_id=f"{event_prefix}-completed", agent_name=agent_name, status="completed",
        started_at=started_at, finished_at=finished_at, duration_seconds=duration,
        input_summary=input_summary, output_summary=output_summary,
        message=f"{agent_name} completed.",
    ))
    return output


def record_skipped(state: WorkflowState, agent_name: str, message: str) -> None:
    now = datetime.now(timezone.utc)
    state.events.append(WorkflowEvent(
        event_id=f"event-{uuid4().hex[:12]}-skipped", agent_name=agent_name,
        status="skipped", started_at=now, finished_at=now,
        input_summary="", output_summary="skipped", message=message,
    ))
