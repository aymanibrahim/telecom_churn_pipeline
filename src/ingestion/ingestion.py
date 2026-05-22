"""
ingestion.py
------------
PART 1 – STAGE 1: Data Ingestion

Responsibilities
~~~~~~~~~~~~~~~~
* Validate that all required raw source files exist and are readable.
* Ingest the OpenCellID Africa tower dataset and produce the filtered
  Senegal/Expresso 90-day snapshot  (``opencellid_senegal_90d.csv``).
* Ingest the Expresso subscriber dataset and produce a reproducible
  100 k-row sample  (``expresso_sample_100k.csv``).
* Validate schema consistency of both raw files.
* Log every step and raise descriptive exceptions on failure.
* Fire a notification (email + optional Slack webhook) when ingestion
  succeeds.

All logic lives inside importable functions so that Airflow
``PythonOperator`` tasks can call them directly without side-effects at
import time.
"""

from __future__ import annotations

import os
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import requests
from dask.diagnostics import ProgressBar

# ── Internal imports ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import (
    AFRICA_TOWERS_FILE,
    EXPRESSO_APPROX_ROWS,
    EXPRESSO_CAT_COLS,
    EXPRESSO_DTYPES,
    EXPRESSO_FILE,
    EXPRESSO_SAMPLE_100K_FILE,
    GADM_GPKG_FILE,
    GADM_URL,
    NOTIFICATION_EMAIL_FROM,
    NOTIFICATION_EMAIL_TO,
    NOTIFICATION_SLACK_WEBHOOK,
    NOTIFICATION_SMTP_HOST,
    NOTIFICATION_SMTP_PASSWORD,
    NOTIFICATION_SMTP_PORT,
    NOTIFICATION_SMTP_USER,
    OPENCELLID_90D_FILE,
    OPENCELLID_DTYPES,
    PROCESSED_DIR,
    RANDOM_STATE,
    SAMPLE_CHUNKSIZE,
    SENEGAL_MCC,
    EXPRESSO_MNC,
    TARGET_SAMPLE_SIZE,
    TOWER_WINDOW_END,
    TOWER_WINDOW_START,
)
from src.logger import get_logger

log = get_logger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1.  FILE VALIDATION HELPERS                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _check_file_exists(path: Path, label: str) -> None:
    """Raise ``FileNotFoundError`` if *path* does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"[Ingestion] Required file '{label}' not found at: {path}\n"
            "Place the file in the correct location before running the pipeline."
        )
    size_mb = path.stat().st_size / 1_048_576
    log.info("[Ingestion] ✓ %s found  (%s, %.1f MB)", label, path.name, size_mb)


def _validate_csv_schema(path: Path, required_cols: list[str], label: str) -> None:
    """
    Read the first row of a CSV and assert that all *required_cols* are
    present.  Raises ``ValueError`` on mismatch.
    """
    header = pd.read_csv(path, nrows=0)
    actual = set(header.columns.str.strip())
    missing = set(required_cols) - actual
    if missing:
        raise ValueError(
            f"[Ingestion] Schema error in '{label}': columns {missing} are missing.\n"
            f"Expected  : {sorted(required_cols)}\n"
            f"Got       : {sorted(actual)}"
        )
    log.info("[Ingestion] ✓ Schema OK for %s  (%d columns verified)", label, len(required_cols))


def validate_raw_inputs() -> None:
    """
    Entry point called by the DAG's ingestion task to verify all source
    files before any heavy computation starts.
    """
    log.info("=" * 60)
    log.info("[Ingestion] Validating raw input files …")

    _check_file_exists(AFRICA_TOWERS_FILE, "Africa_towers.csv (OpenCellID)")
    _check_file_exists(EXPRESSO_FILE, "expresso.csv (Expresso subscribers)")
    # GADM GeoPackage is downloaded lazily; only warn if absent
    if not GADM_GPKG_FILE.exists():
        log.warning(
            "[Ingestion] GADM GeoPackage not found at %s — will be downloaded by "
            "the processing stage.", GADM_GPKG_FILE
        )

    _validate_csv_schema(
        AFRICA_TOWERS_FILE,
        required_cols=list(OPENCELLID_DTYPES.keys()),
        label="Africa_towers.csv",
    )
    _validate_csv_schema(
        EXPRESSO_FILE,
        required_cols=list(EXPRESSO_DTYPES.keys()),
        label="expresso.csv",
    )

    log.info("[Ingestion] All raw input validations passed ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2.  OPENCELLID INGESTION  →  opencellid_senegal_90d.csv               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _load_opencellid_dask(source: Path) -> dd.DataFrame:
    """Load Africa_towers.csv with Dask using pre-defined dtypes."""
    columns = list(OPENCELLID_DTYPES.keys())
    log.info("[Ingestion] Loading Africa_towers with Dask …")
    ddf = dd.read_csv(
        str(source),
        usecols=columns,
        dtype=OPENCELLID_DTYPES,
        blocksize="32MB",
    )
    log.info(
        "[Ingestion] Dask partitions: %d  |  columns: %d",
        ddf.npartitions, len(ddf.columns)
    )
    return ddf


def _filter_senegal_expresso(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Filter to Senegal (MCC=608) and Expresso (MNC=3) towers and persist
    the result in distributed memory to avoid recomputation.
    """
    log.info(
        "[Ingestion] Filtering Senegal (MCC=%d) + Expresso (MNC=%d) …",
        SENEGAL_MCC, EXPRESSO_MNC,
    )
    filtered = ddf[
        (ddf["MCC"] == SENEGAL_MCC) & (ddf["MNC"] == EXPRESSO_MNC)
    ].persist()
    n = len(filtered)
    log.info("[Ingestion] Filtered tower rows: %d", n)
    return filtered


def _compute_to_pandas(ddf: dd.DataFrame) -> pd.DataFrame:
    """Trigger Dask computation and return a clean pandas DataFrame."""
    log.info("[Ingestion] Computing Dask graph → pandas …")
    with ProgressBar():
        df = ddf.compute()
    for col in ["Country", "Network", "Continent"]:
        if col in df.columns and hasattr(df[col], "cat"):
            df[col] = df[col].cat.remove_unused_categories()
    df = df.reset_index(drop=True)
    log.info("[Ingestion] Computed shape: %s", df.shape)
    return df


def _add_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert UNIX timestamps to human-readable datetime columns."""
    df["created_date"] = pd.to_datetime(df["created"], unit="s")
    df["updated_date"] = pd.to_datetime(df["updated"], unit="s")
    return df


def _filter_90d_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only towers that were active inside the 90-day observation window
    (``TOWER_WINDOW_START`` → ``TOWER_WINDOW_END``) to align with the
    Expresso churn definition.
    """
    start = pd.to_datetime(TOWER_WINDOW_START)
    end   = pd.to_datetime(TOWER_WINDOW_END)
    log.info(
        "[Ingestion] Filtering 90-day window: %s → %s",
        TOWER_WINDOW_START, TOWER_WINDOW_END
    )
    mask = (df["created_date"] <= end) & (df["updated_date"] >= start)
    df90 = df[mask].reset_index(drop=True)
    log.info(
        "[Ingestion] Towers in 90-day window: %d  (from %d)",
        len(df90), len(df)
    )
    return df90


def ingest_opencellid() -> str:
    """
    Full OpenCellID ingestion pipeline.

    Reads ``Africa_towers.csv``, filters to Senegal/Expresso towers within
    the 90-day observation window, and saves the result to
    ``OPENCELLID_90D_FILE``.

    Returns
    -------
    str
        Absolute path to the saved file (for XCom passing in Airflow).
    """
    log.info("=" * 60)
    log.info("[Ingestion] Starting OpenCellID ingestion …")
    t0 = time.time()

    _check_file_exists(AFRICA_TOWERS_FILE, "Africa_towers.csv")

    ddf = _load_opencellid_dask(AFRICA_TOWERS_FILE)
    ddf = _filter_senegal_expresso(ddf)
    df  = _compute_to_pandas(ddf)
    df  = _add_datetime_cols(df)
    df  = _filter_90d_window(df)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = str(OPENCELLID_90D_FILE)
    df.to_csv(out, index=False)
    size_mb = os.path.getsize(out) / 1_048_576
    elapsed = time.time() - t0
    log.info(
        "[Ingestion] ✓ opencellid_senegal_90d saved → %s  (%d rows, %.1f MB) [%.1fs]",
        out, len(df), size_mb, elapsed,
    )
    return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3.  EXPRESSO INGESTION  →  expresso_sample_100k.csv                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def ingest_expresso_sample() -> str:
    """
    Produce a reproducible 100 k-row sample from the full Expresso dataset
    using chunk-based sampling (preserves the row distribution across the
    file without loading 2 M rows into RAM at once).

    Returns
    -------
    str
        Absolute path to the saved sample file.
    """
    log.info("=" * 60)
    log.info("[Ingestion] Starting Expresso sample (100k) ingestion …")
    t0 = time.time()

    _check_file_exists(EXPRESSO_FILE, "expresso.csv")

    sample_chunks: list[pd.DataFrame] = []
    frac = TARGET_SAMPLE_SIZE / EXPRESSO_APPROX_ROWS

    log.info(
        "[Ingestion] Chunk-sampling expresso.csv  (chunksize=%d, frac=%.4f) …",
        SAMPLE_CHUNKSIZE, frac
    )
    for chunk in pd.read_csv(
        EXPRESSO_FILE,
        chunksize=SAMPLE_CHUNKSIZE,
        dtype=EXPRESSO_DTYPES,
        low_memory=False,
    ):
        sample_chunks.append(
            chunk.sample(frac=frac, random_state=RANDOM_STATE)
        )

    sample = pd.concat(sample_chunks)

    # Ensure exactly TARGET_SAMPLE_SIZE rows
    if len(sample) >= TARGET_SAMPLE_SIZE:
        sample = sample.sample(n=TARGET_SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        log.warning(
            "[Ingestion] Sample has only %d rows (target %d); using all.",
            len(sample), TARGET_SAMPLE_SIZE
        )

    sample = sample.reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = str(EXPRESSO_SAMPLE_100K_FILE)
    sample.to_csv(out, index=False)
    size_mb = os.path.getsize(out) / 1_048_576
    elapsed = time.time() - t0
    log.info(
        "[Ingestion] ✓ expresso_sample_100k saved → %s  (%d rows, %.1f MB) [%.1fs]",
        out, len(sample), size_mb, elapsed,
    )
    return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4.  GADM GEOPACKAGE DOWNLOAD (lazy, cached)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def download_gadm_geopackage(retries: int = 3) -> str:
    """
    Download the GADM v4.1 Senegal GeoPackage if it is not already cached.

    Parameters
    ----------
    retries : int
        Number of download attempts before giving up.

    Returns
    -------
    str
        Absolute path to the GeoPackage file.
    """
    dest = Path(GADM_GPKG_FILE)
    if dest.exists():
        size_mb = dest.stat().st_size / 1_048_576
        log.info("[Ingestion] GADM GeoPackage already cached (%.1f MB), skipping download.", size_mb)
        return str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("[Ingestion] Downloading GADM GeoPackage from %s …", GADM_URL)

    for attempt in range(1, retries + 1):
        try:
            log.info("[Ingestion] Download attempt %d/%d …", attempt, retries)
            resp = requests.get(GADM_URL, stream=True, timeout=300)
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=256 * 1024):
                    fh.write(chunk)
            size_mb = dest.stat().st_size / 1_048_576
            log.info("[Ingestion] ✓ GADM GeoPackage saved (%.1f MB)", size_mb)
            return str(dest)
        except Exception as exc:                # noqa: BLE001
            log.warning("[Ingestion] Attempt %d failed: %s", attempt, exc)
            time.sleep(5)

    raise RuntimeError(
        f"[Ingestion] Failed to download GADM GeoPackage after {retries} attempts.\n"
        f"Download manually from: {GADM_URL}\n"
        f"and place it at: {dest}"
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.  NOTIFICATION                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _send_email_notification(subject: str, body: str) -> None:
    """Send a plain-text email via the configured SMTP server."""
    if not NOTIFICATION_SMTP_HOST or not NOTIFICATION_EMAIL_TO:
        log.debug("[Notification] SMTP not configured — skipping email.")
        return
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"]    = NOTIFICATION_EMAIL_FROM
        msg["To"]      = NOTIFICATION_EMAIL_TO

        with smtplib.SMTP(NOTIFICATION_SMTP_HOST, NOTIFICATION_SMTP_PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            if NOTIFICATION_SMTP_USER:
                smtp.login(NOTIFICATION_SMTP_USER, NOTIFICATION_SMTP_PASSWORD)
            smtp.sendmail(NOTIFICATION_EMAIL_FROM, [NOTIFICATION_EMAIL_TO], msg.as_string())

        log.info("[Notification] Email sent to %s", NOTIFICATION_EMAIL_TO)
    except Exception as exc:                # noqa: BLE001
        log.warning("[Notification] Email failed: %s", exc)


def _send_slack_notification(text: str) -> None:
    """Post a message to a Slack channel via an incoming webhook URL."""
    if not NOTIFICATION_SLACK_WEBHOOK:
        log.debug("[Notification] Slack webhook not configured — skipping.")
        return
    try:
        resp = requests.post(
            NOTIFICATION_SLACK_WEBHOOK,
            json={"text": text},
            timeout=10,
        )
        resp.raise_for_status()
        log.info("[Notification] Slack message sent.")
    except Exception as exc:                # noqa: BLE001
        log.warning("[Notification] Slack notification failed: %s", exc)


def send_notification_email(
    subject: str = "Telecom Pipeline: Data Ingestion Succeeded ✓",
    body: str | None = None,
) -> None:
    """
    Airflow-callable notification function.  Sends both an email and an
    optional Slack message to inform the team that new data has been
    successfully ingested.

    Parameters
    ----------
    subject : str
        Email / notification subject line.
    body    : str | None
        Message body.  A sensible default is generated if *None*.
    """
    if body is None:
        body = (
            "The Telecom AI-Enhanced Data Pipeline has successfully ingested new data.\n\n"
            f"  OpenCellID 90-day snapshot : {OPENCELLID_90D_FILE}\n"
            f"  Expresso 100k sample       : {EXPRESSO_SAMPLE_100K_FILE}\n\n"
            "The pipeline will now proceed to the processing stage.\n"
        )
    log.info("[Notification] Sending ingestion-success notification …")
    _send_email_notification(subject, body)
    _send_slack_notification(f"*{subject}*\n{body}")
    log.info("[Notification] Notification dispatched.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.  AIRFLOW ENTRYPOINTS                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def ingest_data(**kwargs) -> dict:
    """
    Airflow ``PythonOperator`` callable — orchestrates the full ingestion
    stage:

    1. Validate raw input files.
    2. Ingest OpenCellID towers → ``opencellid_senegal_90d.csv``.
    3. Ingest Expresso subscribers → ``expresso_sample_100k.csv``.
    4. Download GADM GeoPackage if not cached.
    5. Send a success notification.

    Returns
    -------
    dict
        Paths to produced files (pushed to XCom by Airflow automatically).
    """
    log.info("[Ingestion] ══════ INGESTION STAGE STARTED ══════")

    validate_raw_inputs()

    opencellid_path = ingest_opencellid()
    expresso_path   = ingest_expresso_sample()
    gadm_path       = download_gadm_geopackage()

    send_notification_email()

    result = {
        "opencellid_90d_path":  opencellid_path,
        "expresso_100k_path":   expresso_path,
        "gadm_gpkg_path":       gadm_path,
    }
    log.info("[Ingestion] ══════ INGESTION STAGE COMPLETE ══════")
    log.info("[Ingestion] Outputs: %s", result)
    return result


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7.  STANDALONE EXECUTION (development / debugging only)                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import pprint
    pprint.pprint(ingest_data())
