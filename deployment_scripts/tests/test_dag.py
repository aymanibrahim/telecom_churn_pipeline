"""Smoke-test that the DAG file parses and exposes the right tasks.

Skipped if Airflow isn't installed (e.g., on a developer laptop without
the Docker image).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

airflow = pytest.importorskip("airflow", reason="Airflow not installed")


def test_dag_imports_and_has_expected_tasks() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dags_dir = project_root / "dags"

    if str(dags_dir) not in sys.path:
        sys.path.insert(0, str(dags_dir))

    spec = importlib.util.spec_from_file_location(
        "churn_pipeline", dags_dir / "churn_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    from airflow.models import DagBag
    bag = DagBag(dag_folder=str(dags_dir), include_examples=False)
    assert "telecom_churn_production_pipeline" in bag.dags, bag.import_errors
    dag = bag.dags["telecom_churn_production_pipeline"]
    expected = {
        "sample_expresso", "build_opencellid", "build_telecom_churn",
        "run_eda", "train_model", "predict",
    }
    assert expected.issubset(set(dag.task_ids))
