"""
telecom_churn_dag.py
--------------------
PART 2 – Airflow DAG: Telecom AI-Enhanced Data Pipeline

DAG Structure
~~~~~~~~~~~~~

    validate_inputs
          │
    ingest_data  ──────────────────────────────────────────────────────────────┐
          │                                                                    │
    transform_data                                                  send_notification_email
          │
    run_model_inference
          │
    serve_predictions
          │
    pipeline_success_alert  (on_success_callback)

Operator types used
~~~~~~~~~~~~~~~~~~~
* ``PythonOperator``   — all four pipeline stages + notification.
* ``EmailOperator``    — pipeline-success email (Airflow native).
* DAG-level ``on_failure_callback`` — notifies team on any task failure.

All heavy logic lives in the four stage modules under ``src/``.  The DAG
file itself contains *only* orchestration wiring — import the callables,
wrap them in operators, and define the dependency graph.

Scheduling
~~~~~~~~~~
Runs ``@daily`` by default.  Override via the ``TELECOM_DAG_SCHEDULE``
environment variable or set ``schedule`` directly below.

Prediction parity
~~~~~~~~~~~~~~~~~
Because this DAG calls the same Python functions used in the notebooks
(with no logic duplication), the predictions are guaranteed to be
identical to notebook-generated results.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

from airflow import DAG
from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ── Project root on the Airflow worker ─────────────────────────────────────
# Set TELECOM_BASE_DIR in your docker-compose / ECS task definition.
import sys
PROJECT_ROOT = Path(os.environ.get("TELECOM_PROJECT_ROOT", "/opt/airflow"))
sys.path.insert(0, str(PROJECT_ROOT))

# ── Stage callables ─────────────────────────────────────────────────────────
from src.ingestion.ingestion  import ingest_data, send_notification_email
from src.processing.processing import transform_data
from src.modeling.modeling     import run_model_inference
from src.serving.serving       import serve_predictions

# ── Config ──────────────────────────────────────────────────────────────────
from config.config import (
    DAG_EMAIL,
    DAG_OWNER,
    DAG_RETRIES,
    DAG_RETRY_DELAY_MINUTES,
    DAG_SCHEDULE,
    DAG_START_DATE_STR,
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DAG-LEVEL CALLBACKS                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _on_failure_callback(context: dict) -> None:
    """
    Called when ANY task in the DAG fails.
    Logs the failure details; extend to post to Slack or PagerDuty as needed.
    """
    task_id  = context["task_instance"].task_id
    dag_id   = context["dag"].dag_id
    run_id   = context["run_id"]
    exc      = context.get("exception", "unknown error")

    import logging
    log = logging.getLogger("airflow.task")
    log.error(
        "[DAG] !! FAILURE in dag=%s task=%s run=%s\nException: %s",
        dag_id, task_id, run_id, exc
    )

    # Optional Slack / webhook notification on failure
    slack_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if slack_webhook:
        try:
            import requests
            requests.post(
                slack_webhook,
                json={
                    "text": (
                        f":x: *Telecom Pipeline FAILED*\n"
                        f"DAG: `{dag_id}` | Task: `{task_id}` | Run: `{run_id}`\n"
                        f"Error: `{exc}`"
                    )
                },
                timeout=10,
            )
        except Exception:          # noqa: BLE001
            pass


def _on_success_callback(context: dict) -> None:
    """Log pipeline completion when the final task succeeds."""
    import logging
    log = logging.getLogger("airflow.task")
    log.info(
        "[DAG] ✓ Pipeline completed successfully. DAG=%s | Run=%s",
        context["dag"].dag_id, context["run_id"]
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DEFAULT TASK ARGUMENTS                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_START_DATE = datetime.datetime.strptime(DAG_START_DATE_STR, "%Y-%m-%d")

DEFAULT_ARGS = {
    "owner":            DAG_OWNER,
    "depends_on_past":  False,
    "email":            DAG_EMAIL,
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          DAG_RETRIES,
    "retry_delay":      datetime.timedelta(minutes=DAG_RETRY_DELAY_MINUTES),
    "on_failure_callback": _on_failure_callback,
}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DAG DEFINITION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with DAG(
    dag_id="telecom_ai_churn_pipeline",
    description=(
        "End-to-end Telecom AI-Enhanced Data Pipeline: "
        "Ingest → Process → Model → Serve"
    ),
    default_args=DEFAULT_ARGS,
    start_date=_START_DATE,
    schedule=os.environ.get("TELECOM_DAG_SCHEDULE", DAG_SCHEDULE),
    catchup=False,
    max_active_runs=1,
    tags=["telecom", "churn", "ml", "production"],
    doc_md=__doc__,
) as dag:

    # ──────────────────────────────────────────────────────────────────────
    # TASK 1 — Ingest
    # ──────────────────────────────────────────────────────────────────────
    t_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
        doc_md="""
        **Stage 1 — Data Ingestion**

        * Validates that ``Africa_towers.csv`` and ``expresso.csv`` exist
          and have the expected schema.
        * Loads Africa_towers.csv with Dask, filters to Senegal (MCC=608)
          and Expresso (MNC=3) towers within the 90-day observation window,
          and saves ``opencellid_senegal_90d.csv``.
        * Chunk-samples ``expresso.csv`` to 100 k rows and saves
          ``expresso_sample_100k.csv``.
        * Downloads the GADM GeoPackage (cached on first run).
        * Fires an email + Slack notification on success.
        """,
    )

    # ──────────────────────────────────────────────────────────────────────
    # TASK 2 — Notification (parallel to heavy processing)
    # ──────────────────────────────────────────────────────────────────────
    t_notify = PythonOperator(
        task_id="send_notification_email",
        python_callable=send_notification_email,
        op_kwargs={
            "subject": "Telecom Pipeline: New Data Ingested ✓",
            "body": (
                "The Telecom AI-Enhanced Data Pipeline has successfully "
                "ingested new data.\n\n"
                "The pipeline will now proceed to the processing stage."
            ),
        },
        doc_md="""
        **Notification — Ingestion Success**

        Sends an email (and optional Slack message) to the configured
        recipients informing them that new data has been ingested.
        Runs in parallel with the processing stage so it does not block
        the pipeline.
        """,
    )

    # ──────────────────────────────────────────────────────────────────────
    # TASK 3 — Process / Transform
    # ──────────────────────────────────────────────────────────────────────
    t_process = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
        execution_timeout=datetime.timedelta(hours=2),
        doc_md="""
        **Stage 2 — Data Processing & Transformation**

        * Loads GADM administrative boundaries (ADM1 / ADM2 / ADM3).
        * Builds a tower GeoDataFrame from the 90-day OpenCellID snapshot.
        * Runs point-in-polygon spatial joins at all three levels.
        * Computes network KPIs (tower count, avg range, avg signal,
          coverage index, signal strength index, network quality score).
        * Rolls up Department and Arrondissement KPIs to the Region level.
        * Loads the full Expresso dataset with Dask and broadcast-merges
          all KPI tables.
        * Computes the integrated ``telecom_churn`` dataset (≈2 M rows)
          and saves it to disk.
        * Generates the ``telecom_churn_100k`` sample.
        * Validates output row counts and schema.
        """,
    )

    # ──────────────────────────────────────────────────────────────────────
    # TASK 4 — Model Inference
    # ──────────────────────────────────────────────────────────────────────
    t_model = PythonOperator(
        task_id="run_model_inference",
        python_callable=run_model_inference,
        execution_timeout=datetime.timedelta(hours=3),
        doc_md="""
        **Stage 3 — AI Modeling (Training, Calibration & Inference)**

        * Loads the 100 k-row training dataset.
        * Applies the notebook-identical feature engineering pipeline.
        * Splits data 70% train / 15% val / 15% test (stratified).
        * Builds a leak-free ImbPipeline:
            - Numerical: median impute → StandardScaler
            - Categorical: constant impute → TargetEncoder → StandardScaler
            - SMOTE (fit-only)
            - LightGBM classifier
        * Runs RandomizedSearchCV (20 iterations, 5-fold stratified CV,
          optimised for ROC-AUC).
        * Calibrates the best model with isotonic regression on the val set.
        * Tunes the decision threshold on the val set (maximises F1).
        * Evaluates on the held-out test set.
        * Segments customers into risk tiers and action segments.
        * Saves a versioned ``.joblib`` payload and updates the stable
          symlink ``churn_model.joblib``.
        * Saves evaluation metrics CSV.

        **Prediction parity**: all steps replicate the notebook exactly,
        guaranteeing identical results when run via the DAG.
        """,
    )

    # ──────────────────────────────────────────────────────────────────────
    # TASK 5 — Serve / Export
    # ──────────────────────────────────────────────────────────────────────
    t_serve = PythonOperator(
        task_id="serve_predictions",
        python_callable=serve_predictions,
        on_success_callback=_on_success_callback,
        doc_md="""
        **Stage 4 — Model Serving & Output**

        * Loads the predictions CSV from the modeling stage.
        * Validates output schema (guaranteed contract for Streamlit / Flask).
        * Enriches predictions with human-readable labels and model version.
        * Saves the final ``churn_predictions.csv``.
        * Exports the ``high_risk_customers.csv`` subset for CRM integration.
        * Writes ``model_info.json`` consumed by the Flask REST API.
        * Produces and saves a plain-text executive summary report.
        """,
    )

    # ──────────────────────────────────────────────────────────────────────
    # TASK 6 — Pipeline-complete email notification
    # ──────────────────────────────────────────────────────────────────────
    t_pipeline_email = EmailOperator(
        task_id="pipeline_complete_email",
        to=DAG_EMAIL,
        subject="Telecom Pipeline: Full Run Complete ✓ {{ ds }}",
        html_content="""
        <h2>Telecom AI-Enhanced Data Pipeline — Run Complete</h2>
        <p>
          The full Ingest → Process → Model → Serve pipeline has completed
          successfully for execution date <strong>{{ ds }}</strong>.
        </p>
        <h3>Outputs</h3>
        <ul>
          <li><code>data/processed/telecom_churn.csv</code> — full 2 M dataset</li>
          <li><code>data/processed/telecom_churn_100k.csv</code> — 100 k sample</li>
          <li><code>outputs/churn_predictions.csv</code> — scored predictions</li>
          <li><code>outputs/high_risk_customers.csv</code> — CRM integration</li>
          <li><code>outputs/model_info.json</code> — Flask API model info</li>
          <li><code>outputs/executive_summary.txt</code> — business report</li>
          <li><code>models/churn_model.joblib</code> — stable model symlink</li>
        </ul>
        <p>Review the Airflow UI for detailed task logs.</p>
        """,
        doc_md="""
        **Pipeline Complete Notification**

        Sends a rich HTML email summarising the run to the configured
        recipient list.  Triggered only after all four pipeline stages
        complete successfully.
        """,
    )

    # ╔════════════════════════════════════════════════════════════════════╗
    # ║  DEPENDENCY GRAPH                                                 ║
    # ║                                                                   ║
    # ║  t_ingest ──► t_process ──► t_model ──► t_serve ──► t_pipeline_email
    # ║       └──────────────────────────────────────────► t_notify       ║
    # ╚════════════════════════════════════════════════════════════════════╝

    # Main pipeline chain
    t_ingest >> t_process >> t_model >> t_serve >> t_pipeline_email

    # Notification fires right after ingestion (non-blocking for pipeline)
    t_ingest >> t_notify
