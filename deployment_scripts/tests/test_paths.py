"""Verify path resolution honours CHURN_BASE_DIR and produces expected paths."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _fresh_paths(monkeypatch, base_dir: Path | None):
    """Re-import src.paths after setting CHURN_BASE_DIR."""
    if base_dir is None:
        monkeypatch.delenv("CHURN_BASE_DIR", raising=False)
    else:
        monkeypatch.setenv("CHURN_BASE_DIR", str(base_dir))
    # Drop both the package and the submodule so the env var is re-read.
    for mod in ("src.paths", "src"):
        sys.modules.pop(mod, None)
    return importlib.import_module("src.paths")


def test_base_dir_defaults_to_project_root(monkeypatch, project_root: Path) -> None:
    paths = _fresh_paths(monkeypatch, None)
    assert project_root == paths.BASE_DIR


def test_env_var_overrides_base_dir(monkeypatch, tmp_path: Path) -> None:
    paths = _fresh_paths(monkeypatch, tmp_path)
    assert tmp_path == paths.BASE_DIR
    assert tmp_path / "datasets" == paths.DATASETS_DIR
    assert (
        tmp_path / "outputs" / "predictions" / "churn_predictions.csv"
    ) == paths.PREDICTIONS_CSV


def test_bootstrap_creates_output_dirs(monkeypatch, tmp_path: Path) -> None:
    paths = _fresh_paths(monkeypatch, tmp_path)
    paths.bootstrap()
    for d in paths.all_dirs():
        assert d.exists() and d.is_dir(), d
