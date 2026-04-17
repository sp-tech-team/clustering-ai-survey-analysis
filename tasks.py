"""
Background task registry.

Uses Python threading so Phase-1 processing (embed + UMAP) runs without
blocking the Dash server.  A dcc.Interval on the analysis page polls
get_task(session_id) every POLL_INTERVAL_MS milliseconds.

Task lifecycle:
  idle → running → done
                 → error
"""

from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Step:
    name:   str
    status: str = "pending"   # pending | running | done | error
    detail: str = ""


@dataclass
class Task:
    status:   str         = "idle"     # idle | running | done | error
    progress: int         = 0          # 0-100
    message:  str         = ""
    steps:    List[Step]  = field(default_factory=list)
    error:    str         = ""
    result:   Any         = None


_registry: Dict[str, Task] = {}
_lock      = threading.Lock()


def get_task(session_id: str) -> Task:
    with _lock:
        return _registry.get(session_id, Task())


def _set(session_id: str, **kwargs) -> None:
    with _lock:
        if session_id not in _registry:
            _registry[session_id] = Task()
        for k, v in kwargs.items():
            setattr(_registry[session_id], k, v)


def clear(session_id: str) -> None:
    with _lock:
        _registry.pop(session_id, None)


def run_in_background(session_id: str, fn: Callable, *args, **kwargs) -> None:
    """
    Run `fn(*args, **kwargs)` in a daemon thread.
    `fn` should use `update_progress()` to report progress and must
    call `mark_done()` or `mark_error()` when finished.
    """
    _set(session_id, status="running", progress=0, message="Starting…", steps=[], error="", result=None)

    def _run():
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            mark_error(session_id, str(exc))

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def update_progress(session_id: str, progress: int, message: str, step_name: str = "") -> None:
    with _lock:
        if session_id not in _registry:
            return
        task = _registry[session_id]
        task.progress = progress
        task.message  = message
        if step_name:
            # Update or append step
            for s in task.steps:
                if s.name == step_name:
                    s.status = "running"
                    s.detail = message
                    return
            task.steps.append(Step(name=step_name, status="running", detail=message))


def mark_step_done(session_id: str, step_name: str, detail: str = "") -> None:
    """Mark an existing step done, or append a new done step if not found."""
    with _lock:
        task = _registry.get(session_id)
        if not task:
            return
        for s in task.steps:
            if s.name == step_name:
                s.status = "done"
                if detail:
                    s.detail = detail
                return
        # Step not yet in list — add it already-done
        task.steps.append(Step(name=step_name, status="done", detail=detail))


def mark_done(session_id: str, result: Any = None) -> None:
    _set(session_id, status="done", progress=100, message="Complete", result=result)


def mark_error(session_id: str, error: str) -> None:
    _set(session_id, status="error", message=f"Error: {error}", error=error)
