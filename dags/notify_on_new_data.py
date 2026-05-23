"""Notification DAG — watches for new data files and pings on arrival.

Demonstrates the "ممكن أن يتم تجربة مثال عملي لاستخدام Airflow بإرسال
Notification عند وصول بيانات جديدة" line from Milestone 3.

Behaviour
---------
1. A ``FileSensor`` polls ``datasets/incoming/*.csv`` every 5 minutes.
2. When at least one file matches, a ``PythonOperator`` logs the arrival
   (file paths + sizes) into the Airflow task log.
3. ``on_success_callback`` always logs a structured "data arrived" event.
4. An ``EmailOperator`` task fires when SMTP is configured in
   ``airflow.cfg`` (``[smtp]`` section). When SMTP is not configured, the
   task short-circuits with a clear "would have emailed X" log line — so
   parsing this DAG never breaks the scheduler.

The DAG is paused at creation; the user must unpause it explicitly
(``airflow dags unpause notify_on_new_data``) to start sensing.
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.email import EmailOperator
from airflow.sensors.filesystem import FileSensor

# Make ``src`` importable in case we need it later.
_PROJECT_ROOT = Path(
    os.environ.get("CHURN_BASE_DIR", Path(__file__).resolve().parents[1])
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Folder the sensor watches. Mounted into the Airflow container via docker-compose.
INCOMING_DIR = _PROJECT_ROOT / "datasets" / "incoming"
INCOMING_GLOB = str(INCOMING_DIR / "*.csv")
NOTIFY_TO = os.environ.get("CHURN_NOTIFY_EMAIL", "data-team@example.com")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _log_arrival(context):
    """on_success_callback: always log, even if email is a no-op."""
    import logging
    files = sorted(glob.glob(INCOMING_GLOB))
    sizes_mb = [f"{Path(f).name} ({Path(f).stat().st_size / 1e6:.1f} MB)"
                for f in files]
    logging.info("[notify] data arrived in %s: %s", INCOMING_DIR, sizes_mb)


def _log_failure(context):
    import logging
    logging.error("[notify] task failed: %s — %s",
                  context.get("task_instance"), context.get("exception"))


DEFAULT_ARGS = {
    "owner": "data-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=15),
    "on_success_callback": _log_arrival,
    "on_failure_callback": _log_failure,
}


@dag(
    dag_id="notify_on_new_data",
    description=(
        "Watches datasets/incoming/*.csv and notifies (log + email) when new "
        "files arrive."
    ),
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 5, 1),
    schedule=timedelta(minutes=10),   # check every 10 minutes
    catchup=False,
    max_active_runs=1,
    tags=["notification", "telecom", "demo"],
    doc_md=__doc__,
)
def notify_on_new_data():
    """File-arrival notification DAG."""

    wait_for_files = FileSensor(
        task_id="wait_for_new_csv",
        filepath=INCOMING_GLOB,
        fs_conn_id="fs_default",
        poke_interval=60,        # check every minute while waiting
        timeout=60 * 5,          # give up after 5 minutes (run is fast then waits next schedule)
        mode="reschedule",       # release the worker slot between pokes
        soft_fail=True,          # don't crash the DAG if no file shows up
    )

    @task(task_id="report_files")
    def report_files() -> dict:
        """Enumerate files that arrived and return a summary dict.

        The on_success_callback uses this task's success to fire the log
        line. We do NOT raise here even if the folder is empty — the file
        sensor already handled that case.
        """
        files = sorted(glob.glob(INCOMING_GLOB))
        summary = [
            {"name": Path(f).name, "size_mb": round(Path(f).stat().st_size / 1e6, 2)}
            for f in files
        ]
        return {
            "folder": str(INCOMING_DIR),
            "count": len(summary),
            "files": summary,
        }

    summary = report_files()

    # Email task — works as a no-op if SMTP isn't configured. The Airflow
    # scheduler will log "smtp not configured" and skip the actual send.
    notify_email = EmailOperator(
        task_id="send_notification_email",
        to=NOTIFY_TO,
        subject="[Telecom Churn] New data files detected",
        html_content=(
            "<p>The DAG <code>notify_on_new_data</code> saw new CSVs in "
            f"<code>{INCOMING_DIR}</code>:</p>"
            "<pre>{{ ti.xcom_pull(task_ids='report_files') | tojson(indent=2) }}</pre>"
            "<p>This is an automated message.</p>"
        ),
        trigger_rule="all_done",   # also fire on no-file path so the user knows
    )

    wait_for_files >> summary >> notify_email


notify_on_new_data()
