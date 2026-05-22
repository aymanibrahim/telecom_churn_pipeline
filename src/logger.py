"""
logger.py
---------
Centralised logging setup for the Telecom AI-Enhanced Data Pipeline.
Every script imports get_logger() so that all stages write to a common
rotating log file as well as stdout/stderr (captured by Airflow).
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str, log_dir: str | Path = "/opt/airflow/logs") -> logging.Logger:
    """
    Return a named logger that writes to both the console and a rotating
    file at ``log_dir/telecom_pipeline.log``.

    Parameters
    ----------
    name    : Module / script name (use ``__name__``).
    log_dir : Directory where the log file is created.  Defaults to the
              Airflow log directory but is overridable for local runs.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called more than once
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (visible in Airflow task logs) ─────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Rotating file handler ───────────────────────────────────────────────
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path / "telecom_pipeline.log",
            maxBytes=10 * 1024 * 1024,   # 10 MB
            backupCount=5,
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as exc:          # noqa: BLE001
        # Gracefully degrade — file logging is optional in constrained envs
        logger.warning("Could not create file handler: %s", exc)

    return logger
