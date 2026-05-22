"""
serving.py
----------
PART 1 – STAGE 4: Model Serving & Output

Responsibilities
~~~~~~~~~~~~~~~~
* Load the churn predictions CSV written by the modeling stage.
* Load the versioned calibrated model payload to extract metadata.
* Enrich the output with human-readable labels and business context.
* Validate output schema so the Streamlit dashboard and Flask API receive
  a guaranteed contract.
* Export the final predictions CSV to ``OUTPUTS_DIR/churn_predictions.csv``.
* Export a model-info summary JSON consumed by the Flask REST API.
* Produce a concise executive summary for logging / reporting.
* Expose all logic as importable functions for the Airflow PythonOperator.

Output contract (columns guaranteed in churn_predictions.csv)
--------------------------------------------------------------
  churn_probability   float   Model-calibrated churn probability.
  churn_prediction    int     Binary label at the tuned threshold.
  risk_segment        str     High / Medium / Low.
  action_segment      str     Priority Retention / Low-cost Campaign /
                              Loyalty Program / Monitor.
  value_segment       str     High Value / Low Value (if revenue available).
  actual_churn        int     Ground-truth label (test-set rows only).

  + all original feature columns inherited from the scored DataFrame.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import (
    EVAL_METRICS_FILE,
    LATEST_MODEL_SYMLINK,
    MODELS_DIR,
    OUTPUTS_DIR,
    PREDICTIONS_FILE,
    RISK_TIERS,
)
from src.logger import get_logger

log = get_logger(__name__)

# ── Additional serving-specific output paths ─────────────────────────────────
MODEL_INFO_JSON     = OUTPUTS_DIR / "model_info.json"
EXECUTIVE_RPT_FILE  = OUTPUTS_DIR / "executive_summary.txt"
HIGH_RISK_FILE      = OUTPUTS_DIR / "high_risk_customers.csv"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1.  LOAD PREDICTIONS                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_predictions(path: str | Path = PREDICTIONS_FILE) -> pd.DataFrame:
    """
    Load the churn predictions CSV produced by the modeling stage.

    Parameters
    ----------
    path : Path to the predictions CSV.

    Returns
    -------
    pd.DataFrame
    """
    log.info("[Serving] Loading predictions from %s …", path)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"[Serving] Predictions file not found: {path}\n"
            "Ensure the modeling stage completed successfully."
        )
    df = pd.read_csv(str(path))
    log.info("[Serving] Loaded %d prediction rows × %d columns", len(df), df.shape[1])
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2.  LOAD MODEL METADATA                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_model_metadata(model_path: str | Path = LATEST_MODEL_SYMLINK) -> dict:
    """
    Load the joblib model payload and extract the metadata dictionary.

    Parameters
    ----------
    model_path : Path to the ``.joblib`` file (symlink or versioned file).

    Returns
    -------
    dict — model metadata (version, threshold, metrics, feature names …).
    """
    log.info("[Serving] Loading model metadata from %s …", model_path)
    payload  = joblib.load(str(model_path))
    metadata = payload.get("metadata", {})
    log.info("[Serving] Model version: %s | Threshold: %.2f | ROC-AUC: %.4f",
             metadata.get("version", "unknown"),
             metadata.get("threshold", 0.5),
             metadata.get("roc_auc", float("nan")))
    return metadata


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3.  OUTPUT VALIDATION                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Minimum columns the Streamlit dashboard and Flask API require
REQUIRED_OUTPUT_COLS = [
    "churn_probability",
    "churn_prediction",
    "risk_segment",
    "action_segment",
]


def validate_predictions(df: pd.DataFrame) -> None:
    """
    Assert that all required output columns are present and contain valid
    values.  Raises ``ValueError`` on any contract violation.
    """
    log.info("[Serving] Validating prediction output schema …")

    missing = [c for c in REQUIRED_OUTPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[Serving] Prediction output is missing required columns: {missing}"
        )

    # churn_probability must be in [0, 1]
    prob_col = df["churn_probability"]
    if not ((prob_col >= 0) & (prob_col <= 1)).all():
        raise ValueError(
            "[Serving] 'churn_probability' contains values outside [0, 1]."
        )

    # churn_prediction must be binary
    invalid_pred = ~df["churn_prediction"].isin([0, 1])
    if invalid_pred.any():
        raise ValueError(
            f"[Serving] 'churn_prediction' contains non-binary values: "
            f"{df.loc[invalid_pred, 'churn_prediction'].unique()}"
        )

    # risk_segment must use known tier labels
    known_tiers = set(RISK_TIERS.keys())
    bad_tiers   = set(df["risk_segment"].unique()) - known_tiers
    if bad_tiers:
        raise ValueError(
            f"[Serving] Unknown risk_segment values: {bad_tiers}. "
            f"Expected subset of {known_tiers}."
        )

    # No rows should be entirely null
    all_null_rows = df.isnull().all(axis=1).sum()
    if all_null_rows > 0:
        log.warning("[Serving] %d rows are entirely null in predictions.", all_null_rows)

    log.info("[Serving] ✓ Output schema validation passed (%d rows)", len(df))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4.  OUTPUT ENRICHMENT                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def enrich_predictions(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Add derived convenience columns to the predictions DataFrame.

    Columns added (idempotent — skipped if already present)
    -------------------------------------------------------
    ``churn_prob_pct``   : churn_probability formatted as percentage string.
    ``model_version``    : version tag from model metadata.
    ``recommended_action``: plain-language recommendation string.

    Parameters
    ----------
    df       : Predictions DataFrame from the modeling stage.
    metadata : Model metadata dictionary.

    Returns
    -------
    Enriched pd.DataFrame.
    """
    log.info("[Serving] Enriching predictions …")
    df = df.copy()

    if "churn_prob_pct" not in df.columns:
        df["churn_prob_pct"] = (df["churn_probability"] * 100).round(1).astype(str) + "%"

    if "model_version" not in df.columns:
        df["model_version"] = metadata.get("version", "unknown")

    # Human-readable retention recommendation
    if "recommended_action" not in df.columns:
        action_map = {
            "Priority Retention": "Immediate personalised retention offer",
            "Low-cost Campaign":  "Targeted SMS / push notification campaign",
            "Loyalty Program":    "Enrol in loyalty rewards programme",
            "Monitor":            "Routine monitoring — no immediate action",
        }
        df["recommended_action"] = df["action_segment"].map(action_map).fillna("Monitor")

    log.info("[Serving] Enrichment complete. Final columns: %d", len(df.columns))
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.  SAVE PREDICTION OUTPUTS                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_predictions(df: pd.DataFrame, path: str | Path = PREDICTIONS_FILE) -> str:
    """
    Persist the enriched predictions DataFrame to CSV.

    Parameters
    ----------
    df   : Enriched predictions DataFrame.
    path : Destination path (default: ``OUTPUTS_DIR/churn_predictions.csv``).

    Returns
    -------
    str — absolute path to the saved file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path), index=False)
    size_mb = os.path.getsize(str(path)) / 1_048_576
    log.info("[Serving] ✓ Predictions saved → %s  (%d rows, %.1f MB)",
             path, len(df), size_mb)
    return str(path)


def save_high_risk_customers(df: pd.DataFrame) -> str:
    """
    Export only High-risk customers to a separate file for immediate
    downstream CRM integration.

    Returns
    -------
    str — absolute path to the saved file.
    """
    high_risk = df[df["risk_segment"] == "High"].copy()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    high_risk.to_csv(str(HIGH_RISK_FILE), index=False)
    size_mb = os.path.getsize(str(HIGH_RISK_FILE)) / 1_048_576
    log.info(
        "[Serving] ✓ High-risk customers saved → %s  (%d rows, %.1f MB)",
        HIGH_RISK_FILE, len(high_risk), size_mb
    )
    return str(HIGH_RISK_FILE)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.  MODEL INFO JSON (consumed by Flask REST API)                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def export_model_info(metadata: dict, metrics: dict | None = None) -> str:
    """
    Write a model-info JSON file that the Flask API reads on startup to
    report its current model version and performance.

    Parameters
    ----------
    metadata : Metadata dict from the joblib payload.
    metrics  : Optional evaluation metrics dict.  If ``None``, the metrics
               CSV is read from disk.

    Returns
    -------
    str — absolute path to the JSON file.
    """
    log.info("[Serving] Exporting model info JSON …")

    if metrics is None and EVAL_METRICS_FILE.exists():
        metrics_df = pd.read_csv(str(EVAL_METRICS_FILE))
        metrics    = dict(zip(metrics_df["metric"], metrics_df["value"]))
    else:
        metrics = metrics or {}

    info = {
        "model_version": metadata.get("version", "unknown"),
        "model_name":    metadata.get("model_name", "lightgbm"),
        "threshold":     metadata.get("threshold", 0.5),
        "calibrated":    metadata.get("calibrated", True),
        "smote":         metadata.get("smote", True),
        "performance":   {k: round(float(v), 4) for k, v in metrics.items()
                         if isinstance(v, (int, float))},
        "feature_cols": {
            "numerical":   metadata.get("num_cols", []),
            "categorical": metadata.get("cat_cols", []),
        },
        "risk_tiers": {
            tier: {"low": bounds[0], "high": bounds[1]}
            for tier, bounds in RISK_TIERS.items()
        },
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(MODEL_INFO_JSON), "w") as fh:
        json.dump(info, fh, indent=2)

    log.info("[Serving] ✓ Model info JSON saved → %s", MODEL_INFO_JSON)
    return str(MODEL_INFO_JSON)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7.  EXECUTIVE SUMMARY                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def produce_executive_summary(df: pd.DataFrame, metadata: dict) -> str:
    """
    Generate a plain-text executive summary of the current pipeline run
    and save it to disk.  Also logged at INFO level.

    Returns
    -------
    str — summary text.
    """
    total      = len(df)
    n_high     = (df["risk_segment"] == "High").sum()
    n_medium   = (df["risk_segment"] == "Medium").sum()
    n_low      = (df["risk_segment"] == "Low").sum()
    avg_prob   = df["churn_probability"].mean()
    n_churners = df["churn_prediction"].sum() if "churn_prediction" in df.columns else "N/A"

    lines = [
        "=" * 60,
        "  TELECOM AI PIPELINE — EXECUTIVE SUMMARY",
        "=" * 60,
        f"  Model version : {metadata.get('version', 'unknown')}",
        f"  Decision threshold : {metadata.get('threshold', 0.5):.2f}",
        f"  ROC-AUC       : {metadata.get('roc_auc', float('nan')):.4f}",
        f"  F1 Score      : {metadata.get('f1', float('nan')):.4f}",
        "-" * 60,
        f"  Total customers scored : {total:,}",
        f"  Predicted churners     : {n_churners:,}",
        f"  Avg churn probability  : {avg_prob:.2%}",
        "-" * 60,
        f"  High Risk    : {n_high:,}   ({n_high / total:.1%})",
        f"  Medium Risk  : {n_medium:,}   ({n_medium / total:.1%})",
        f"  Low Risk     : {n_low:,}   ({n_low / total:.1%})",
        "=" * 60,
    ]
    summary = "\n".join(lines)
    log.info("\n%s", summary)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(EXECUTIVE_RPT_FILE), "w") as fh:
        fh.write(summary + "\n")
    log.info("[Serving] ✓ Executive summary saved → %s", EXECUTIVE_RPT_FILE)
    return summary


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8.  AIRFLOW ENTRYPOINT                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def serve_predictions(**kwargs) -> dict:
    """
    Airflow ``PythonOperator`` callable — full serving stage:

    1. Load predictions CSV written by the modeling stage.
    2. Load model metadata from the stable symlink.
    3. Validate output schema.
    4. Enrich predictions with derived columns.
    5. Save final predictions CSV (overwrites the intermediate file).
    6. Export high-risk customer subset for CRM integration.
    7. Export model-info JSON for the Flask REST API.
    8. Produce and save the executive summary report.

    Returns
    -------
    dict with output file paths (pushed to XCom automatically by Airflow).
    """
    log.info("[Serving] ══════ SERVING STAGE STARTED ══════")
    t0 = time.time()

    # ── 1-2. Load ────────────────────────────────────────────────────────────
    df       = load_predictions(PREDICTIONS_FILE)
    metadata = load_model_metadata(LATEST_MODEL_SYMLINK)

    # ── 3. Validate ──────────────────────────────────────────────────────────
    validate_predictions(df)

    # ── 4. Enrich ────────────────────────────────────────────────────────────
    df = enrich_predictions(df, metadata)

    # ── 5-6. Save predictions ────────────────────────────────────────────────
    pred_path      = save_predictions(df, PREDICTIONS_FILE)
    high_risk_path = save_high_risk_customers(df)

    # ── 7. Model info JSON ───────────────────────────────────────────────────
    model_info_path = export_model_info(metadata)

    # ── 8. Executive summary ─────────────────────────────────────────────────
    produce_executive_summary(df, metadata)

    elapsed = time.time() - t0
    result = {
        "predictions_path":  pred_path,
        "high_risk_path":    high_risk_path,
        "model_info_path":   model_info_path,
        "executive_rpt_path":str(EXECUTIVE_RPT_FILE),
    }
    log.info("[Serving] ══════ SERVING STAGE COMPLETE (%.1fs) ══════", elapsed)
    log.info("[Serving] Outputs: %s", result)
    return result


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9.  STANDALONE EXECUTION                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import pprint
    pprint.pprint(serve_predictions())
