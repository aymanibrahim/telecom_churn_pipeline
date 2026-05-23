"""Thin wrappers around CSV and joblib I/O.

These helpers exist so every script logs the same way (size + row count)
and so directory creation is centralised.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .logging_setup import get_logger

log = get_logger(__name__)


def read_csv(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Read CSV with consistent logging."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path, **kwargs)
    size_mb = path.stat().st_size / 1024 / 1024
    log.info("read_csv %s: %s rows x %s cols (%.1f MB)",
             path.name, f"{df.shape[0]:,}", df.shape[1], size_mb)
    return df


def write_csv(df: pd.DataFrame, path: Path | str, index: bool = False, **kwargs: Any) -> Path:
    """Write CSV ensuring the parent directory exists, with logging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **kwargs)
    size_mb = path.stat().st_size / 1024 / 1024
    log.info("write_csv %s: %s rows (%.1f MB) -> %s",
             path.name, f"{len(df):,}", size_mb, path)
    return path


def dump_joblib(payload: Any, path: Path | str) -> Path:
    """Persist a Python object via joblib with parent-dir creation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    size_mb = path.stat().st_size / 1024 / 1024
    log.info("dump_joblib %s: %.1f MB -> %s", path.name, size_mb, path)
    return path


def load_joblib(path: Path | str) -> Any:
    """Load a joblib payload."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"joblib file not found: {path}")
    return joblib.load(path)
