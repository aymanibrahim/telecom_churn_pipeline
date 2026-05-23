"""Tests for src.idempotency.skip_if_fresh."""
from __future__ import annotations

import os
import time
from pathlib import Path

from src import idempotency


def test_skips_when_output_fresh(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    out.write_text("hello", encoding="utf-8")

    calls = {"n": 0}

    @idempotency.skip_if_fresh(outputs=[out], max_age_hours=24)
    def fn():
        calls["n"] += 1
        return "ran"

    assert fn() == "skipped:fresh"
    assert calls["n"] == 0


def test_runs_when_output_missing(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    calls = {"n": 0}

    @idempotency.skip_if_fresh(outputs=[out], max_age_hours=24)
    def fn():
        calls["n"] += 1
        out.write_text("x", encoding="utf-8")
        return "ran"

    assert fn() == "ran"
    assert calls["n"] == 1


def test_runs_when_output_stale(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    out.write_text("old", encoding="utf-8")
    # Backdate the file to 1000 hours ago.
    old_time = time.time() - 1000 * 3600
    os.utime(out, (old_time, old_time))

    calls = {"n": 0}

    @idempotency.skip_if_fresh(outputs=[out], max_age_hours=24)
    def fn():
        calls["n"] += 1
        return "ran"

    assert fn() == "ran"
    assert calls["n"] == 1


def test_force_rebuild_env_var_bypasses_cache(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    out.write_text("fresh", encoding="utf-8")

    os.environ["CHURN_FORCE_REBUILD"] = "1"
    calls = {"n": 0}

    @idempotency.skip_if_fresh(outputs=[out], max_age_hours=24 * 30)
    def fn():
        calls["n"] += 1
        return "ran"

    try:
        assert fn() == "ran"
        assert calls["n"] == 1
    finally:
        del os.environ["CHURN_FORCE_REBUILD"]
