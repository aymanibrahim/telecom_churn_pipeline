"""Skip pipeline tasks when their outputs are already fresh.

A task is "fresh" when **all** its declared output paths exist and
their newest mtime is within ``max_age`` of now. Used inside Airflow
tasks::

    from src.idempotency import skip_if_fresh

    @skip_if_fresh(outputs=[paths.EXPRESSO_SAMPLE], max_age_hours=24*7)
    def task_sample_expresso(...):
        ...

Force a re-run by setting the env var ``CHURN_FORCE_REBUILD=1`` or by
deleting the output file.
"""
from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from .logging_setup import get_logger

log = get_logger(__name__)


def _is_fresh(outputs: Iterable[Path], max_age_seconds: float) -> bool:
    paths = [Path(p) for p in outputs]
    if not paths:
        return False
    if any(not p.exists() for p in paths):
        return False
    newest_mtime = max(p.stat().st_mtime for p in paths)
    return (time.time() - newest_mtime) <= max_age_seconds


def skip_if_fresh(
    outputs: Iterable[Path | str],
    max_age_hours: float = 24.0 * 7,
    sentinel: str = "skipped:fresh",
) -> Callable:
    """Decorator: skip the wrapped function when ``outputs`` are recent enough.

    Returns ``sentinel`` (a string) instead of executing when skipped, so
    Airflow XCom shows a clear marker. The downstream task can branch on
    that sentinel if it cares.
    """
    paths = [Path(p) for p in outputs]
    max_age = max_age_hours * 3600.0

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if os.environ.get("CHURN_FORCE_REBUILD") == "1":
                log.info("CHURN_FORCE_REBUILD=1 -> running %s", fn.__name__)
            elif _is_fresh(paths, max_age):
                log.info(
                    "%s skipped — outputs fresh (< %.1f h): %s",
                    fn.__name__, max_age_hours, [str(p) for p in paths],
                )
                return sentinel
            return fn(*args, **kwargs)
        return wrapper
    return decorator
