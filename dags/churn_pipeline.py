"""Telecom Customer Churn — full production DAG (TaskFlow API).

Task graph
----------

    sample_expresso ─┐
                     ├─→ build_telecom_churn_100k ─┐
    build_opencellid ┤                              ├─→ collect_telecom_paths ─→ run_eda ─→ train_model ─→ predict
                     └─→ build_telecom_churn_full ─┘

* ``build_telecom_churn_100k`` feeds ``EXPRESSO_SAMPLE`` (100k rows) into
  the spatial join → writes ``TELECOM_CHURN_100K`` (~100k rows).
* ``build_telecom_churn_full`` feeds ``EXPRESSO_RAW`` (2 M rows) into the
  same spatial join → writes ``TELECOM_CHURN_FULL`` (~2.15 M rows).
* ``collect_telecom_paths`` packages both paths into a dict that
  downstream tasks unpick via the ``USE_FULL_DATASET`` Airflow Variable.

Each task uses Airflow 2.x's @task decorator (TaskFlow API), which
gives:

* Automatic XCom for return values
* Type hints visible in the UI
* Cleaner Python (no PythonOperator boilerplate)

The tasks delegate to ``src/`` modules so the same code runs in CLI,
CI, and Airflow.

Configuration
-------------
* ``CHURN_BASE_DIR``     — env var. Project root. Resolved by ``src.paths``.
* ``USE_FULL_DATASET``   — Airflow Variable. ``"true"`` to train on the
                           full ``telecom_churn.csv`` instead of the 100k sample.
* ``CHURN_FORCE_REBUILD`` — env var. ``"1"`` to bypass idempotency caching.

The Streamlit dashboard runs as a separate service (see
docker-compose.yaml). The DAG's last task (``predict``) refreshes the
prediction CSV and the dashboard reads it on next page load.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models import Variable

# ---------------------------------------------------------------------------
# Make the project's ``src`` package importable from the DAG.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(
    os.environ.get("CHURN_BASE_DIR", Path(__file__).resolve().parents[1])
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# DAG default args
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "data-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


@dag(
    dag_id="telecom_churn_production_pipeline",
    description=(
        "OpenCellID + Expresso → telecom_churn dataset → EDA → ML → "
        "predictions consumed by the Streamlit dashboard."
    ),
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 5, 1),
    schedule=None,                 # manual trigger; switch to "@weekly" once stable
    catchup=False,
    max_active_runs=1,
    tags=["churn", "telecom", "ml"],
    doc_md=__doc__,
)
def telecom_churn_pipeline():
    """TaskFlow DAG for the Telecom Customer Churn pipeline."""

    # -----------------------------------------------------------------------
    # Helper resolvers (lazy — imported inside tasks to avoid heavy imports
    # at DAG-parse time)
    # -----------------------------------------------------------------------
    def _use_full_dataset() -> bool:
        return str(Variable.get("USE_FULL_DATASET", default_var="false")).lower() == "true"

    # -----------------------------------------------------------------------
    # Task 1 — sample expresso
    # -----------------------------------------------------------------------
    @task(
        task_id="sample_expresso",
        execution_timeout=timedelta(minutes=30),
        retries=2,
    )
    def sample_expresso() -> str:
        """Stratified sample of expresso.csv → expresso_sample_100k.csv."""
        from src import idempotency, paths, sampling, validation

        @idempotency.skip_if_fresh(
            outputs=[paths.EXPRESSO_SAMPLE], max_age_hours=24 * 7,
        )
        def _run():
            out = sampling.run(
                input_path=paths.EXPRESSO_RAW,
                output_path=paths.EXPRESSO_SAMPLE,
            )
            validation.validate_csv(out, validation.EXPRESSO_SAMPLE)
            return out

        return str(_run() or paths.EXPRESSO_SAMPLE)

    # -----------------------------------------------------------------------
    # Task 2 — build OpenCellID Senegal
    # -----------------------------------------------------------------------
    @task(
        task_id="build_opencellid",
        execution_timeout=timedelta(hours=2),
        retries=2,
    )
    def build_opencellid() -> str:
        """Filter Africa_towers.csv to Senegal/Expresso (90-day window)."""
        from src import geo, idempotency, paths, validation

        @idempotency.skip_if_fresh(
            outputs=[paths.OPENCELLID_SENEGAL], max_age_hours=24 * 30,
        )
        def _run():
            out = geo.build_senegal_cells(
                input_path=paths.OPENCELLID_RAW,
                output_path=paths.OPENCELLID_SENEGAL,
            )
            validation.validate_csv(out, validation.OPENCELLID_SENEGAL)
            return out

        return str(_run() or paths.OPENCELLID_SENEGAL)

    # -----------------------------------------------------------------------
    # Task 3a — build telecom_churn (100k, from EXPRESSO_SAMPLE)
    # -----------------------------------------------------------------------
    @task(
        task_id="build_telecom_churn_100k",
        execution_timeout=timedelta(minutes=30),
        retries=1,
    )
    def build_telecom_churn_100k(expresso_sample_path: str,
                                  cells_path: str) -> str:
        """100k spatial join: EXPRESSO_SAMPLE + cells → TELECOM_CHURN_100K."""
        from src import geo, idempotency, paths, validation

        scratch = paths.SCRATCH_DIR / "telecom_churn_100k_aux.csv"
        scratch.parent.mkdir(parents=True, exist_ok=True)

        @idempotency.skip_if_fresh(
            outputs=[paths.TELECOM_CHURN_100K], max_age_hours=24 * 7,
        )
        def _run():
            # geo.build_telecom_churn always writes BOTH out_full and out_100k.
            # Since the input here is already 100k, we want the function's
            # "full" channel as our canonical 100k output, and we discard the
            # secondary stratified-sample sidecar.
            out_full, _ = geo.build_telecom_churn(
                cells_path=Path(cells_path),
                expresso_path=Path(expresso_sample_path),
                gadm_path=paths.GADM_GPKG,
                out_full=paths.TELECOM_CHURN_100K,
                out_100k=scratch,
            )
            validation.validate_csv(out_full, validation.TELECOM_CHURN)
            return out_full

        return str(_run() or paths.TELECOM_CHURN_100K)

    # -----------------------------------------------------------------------
    # Task 3b — build telecom_churn (Full 2M, from EXPRESSO_RAW)
    # -----------------------------------------------------------------------
    @task(
        task_id="build_telecom_churn_full",
        execution_timeout=timedelta(hours=1),
        retries=1,
    )
    def build_telecom_churn_full(cells_path: str) -> str:
        """Full 2M spatial join: EXPRESSO_RAW + cells → TELECOM_CHURN_FULL."""
        from src import geo, idempotency, paths

        scratch = paths.SCRATCH_DIR / "telecom_churn_full_aux.csv"
        scratch.parent.mkdir(parents=True, exist_ok=True)

        @idempotency.skip_if_fresh(
            outputs=[paths.TELECOM_CHURN_FULL], max_age_hours=24 * 7,
        )
        def _run():
            # Use the raw 2M expresso. The function's auxiliary stratified
            # sample goes to scratch (the 100k task owns the canonical sample
            # output to avoid concurrent-write races).
            out_full, _ = geo.build_telecom_churn(
                cells_path=Path(cells_path),
                expresso_path=paths.EXPRESSO_RAW,
                gadm_path=paths.GADM_GPKG,
                out_full=paths.TELECOM_CHURN_FULL,
                out_100k=scratch,
            )
            return out_full

        return str(_run() or paths.TELECOM_CHURN_FULL)

    # -----------------------------------------------------------------------
    # Task 3c — collect both paths so downstream tasks can pick by Variable
    # -----------------------------------------------------------------------
    @task(task_id="collect_telecom_paths")
    def collect_telecom_paths(p_100k: str, p_full: str) -> dict[str, str]:
        """Pack the two canonical telecom_churn paths for downstream use."""
        return {"sample": p_100k, "full": p_full}

    # -----------------------------------------------------------------------
    # Task 4 — EDA
    # -----------------------------------------------------------------------
    @task(
        task_id="run_eda",
        execution_timeout=timedelta(minutes=30),
        retries=1,
    )
    def run_eda(telecom_paths: dict[str, str]) -> str:
        """Headless EDA report → outputs/eda/."""
        from src import eda, paths
        eda.run_eda(
            input_path=Path(telecom_paths["full"] if _use_full_dataset()
                            else telecom_paths["sample"]),
            output_dir=paths.EDA_DIR,
        )
        return str(paths.EDA_DIR)

    # -----------------------------------------------------------------------
    # Task 5 — train
    # -----------------------------------------------------------------------
    @task(
        task_id="train_model",
        execution_timeout=timedelta(hours=1),
        retries=1,
    )
    def train_model(telecom_paths: dict[str, str]) -> dict:
        """Train + calibrate LightGBM, write model + metrics."""
        from src import modeling, paths
        return modeling.train(
            input_path=Path(telecom_paths["full"] if _use_full_dataset()
                            else telecom_paths["sample"]),
            model_dir=paths.MODELS_DIR,
            metrics_dir=paths.METRICS_DIR,
            run_shap=True,
        )

    # -----------------------------------------------------------------------
    # Task 6 — predict
    # -----------------------------------------------------------------------
    @task(
        task_id="predict",
        execution_timeout=timedelta(minutes=30),
        retries=1,
    )
    def predict(train_result: dict, telecom_paths: dict[str, str]) -> str:
        """Batch inference → churn_predictions.csv (dashboard input)."""
        from src import inference, paths, validation
        out = inference.predict(
            model_path=Path(train_result["model_path"]),
            input_path=Path(telecom_paths["full"] if _use_full_dataset()
                            else telecom_paths["sample"]),
            output_path=paths.PREDICTIONS_CSV,
        )
        validation.validate_csv(out, validation.CHURN_PREDICTIONS)
        return str(out)

    # -----------------------------------------------------------------------
    # Wire the graph
    # -----------------------------------------------------------------------
    expresso_out = sample_expresso()
    cells_out = build_opencellid()

    tc_100k = build_telecom_churn_100k(expresso_out, cells_out)
    tc_full = build_telecom_churn_full(cells_out)
    telecom_paths = collect_telecom_paths(tc_100k, tc_full)

    eda_out = run_eda(telecom_paths)
    train_result = train_model(telecom_paths)

    # ``run_eda`` is a quality gate but doesn't gate training. Keep both as
    # downstream of telecom_paths; force training to wait until EDA finishes
    # so EDA artefacts are ready before the dashboard refresh.
    eda_out >> train_result

    predict(train_result, telecom_paths)


telecom_churn_pipeline()
