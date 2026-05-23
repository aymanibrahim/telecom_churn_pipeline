"""Path resolution for the Telecom Churn pipeline.

All paths are derived from a single base directory, configurable via the
``CHURN_BASE_DIR`` environment variable. This lets the same code run on:

* Windows local dev:  ``set CHURN_BASE_DIR=C:\\Users\\Lenovo\\Downloads\\04_requirements``
* Docker Airflow:     ``CHURN_BASE_DIR=/opt/airflow/project`` (set in docker-compose)
* CI:                 ``CHURN_BASE_DIR=$GITHUB_WORKSPACE``

If the env var is unset, BASE_DIR falls back to the project root inferred from
this file's location (``src/paths.py`` → parent of ``src/`` is the project root).
"""
from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Base directory
# ---------------------------------------------------------------------------
def _resolve_base_dir() -> Path:
    env_dir = os.environ.get("CHURN_BASE_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    # Fallback: parent of the directory containing this file.
    return Path(__file__).resolve().parents[1]


BASE_DIR: Path = _resolve_base_dir()

# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------
DATASETS_DIR: Path = BASE_DIR / "datasets"
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
CONFIG_DIR: Path = BASE_DIR / "config"
DASHBOARDS_DIR: Path = BASE_DIR / "dashboards"
# Throw-away sidecar outputs from tasks that produce multiple files but
# only want one (e.g. the dual `build_telecom_churn_*` tasks). Gitignored.
SCRATCH_DIR: Path = DATASETS_DIR / "_scratch"

# ---------------------------------------------------------------------------
# Sub-output directories
# ---------------------------------------------------------------------------
EDA_DIR: Path = OUTPUTS_DIR / "eda"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"

# ---------------------------------------------------------------------------
# Specific input files
# ---------------------------------------------------------------------------
EXPRESSO_RAW: Path = DATASETS_DIR / "expresso" / "expresso.csv"
EXPRESSO_SAMPLE: Path = DATASETS_DIR / "expresso" / "expresso_sample_100k.csv"

OPENCELLID_RAW: Path = DATASETS_DIR / "opencellid" / "Africa_towers.csv"
OPENCELLID_SENEGAL: Path = DATASETS_DIR / "opencellid" / "opencellid_senegal_90d.csv"

TELECOM_CHURN_FULL: Path = DATASETS_DIR / "telecom_churn" / "telecom_churn.csv"
TELECOM_CHURN_100K: Path = DATASETS_DIR / "telecom_churn" / "telecom_churn_100k.csv"

GADM_GPKG: Path = DATASETS_DIR / "telecom_churn" / "gadm41_SEN.gpkg"

# ---------------------------------------------------------------------------
# Specific output files
# ---------------------------------------------------------------------------
PREDICTIONS_CSV: Path = PREDICTIONS_DIR / "churn_predictions.csv"
EVAL_METRICS_CSV: Path = METRICS_DIR / "evaluation_metrics.csv"
FEATURE_IMPORTANCES_CSV: Path = METRICS_DIR / "feature_importances.csv"
SHAP_SUMMARY_PNG: Path = METRICS_DIR / "shap_summary.png"
EDA_SUMMARY_JSON: Path = EDA_DIR / "eda_summary.json"

CHURN_MODEL_JOBLIB: Path = MODELS_DIR / "churn_model.joblib"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """Create ``path`` (a directory or the parent of a file) if needed."""
    target = path if path.suffix == "" else path.parent
    target.mkdir(parents=True, exist_ok=True)
    return path


def all_dirs() -> list[Path]:
    """Return every directory the pipeline writes to. Used for bootstrap."""
    return [
        DATASETS_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        EDA_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
    ]


def bootstrap() -> None:
    """Create every output directory the pipeline needs."""
    for d in all_dirs():
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    bootstrap()
    print(f"BASE_DIR: {BASE_DIR}")
    for name, p in sorted(globals().items()):
        if isinstance(p, Path) and name.isupper():
            print(f"  {name}: {p}")
