"""Batch inference and risk segmentation.

Source notebook: notebooks/Telecom Customer Churn - Training, Inference
& Evaluation - Project Team A.ipynb (segmentation cells).

Output schema (consumed by dashboards/churn_dashboard_app.py):

* user_id, churn_probability, churn_prediction, risk_segment,
  actual_churn (if present in input), revenue (if present)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config, data_io, feature_engineering
from .logging_setup import get_logger

log = get_logger(__name__)


def classify_risk(prob: float) -> str:
    """Return ``High`` / ``Medium`` / ``Low`` from a churn probability."""
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def predict(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    threshold: float | None = None,
) -> Path:
    """Score every row of ``input_path`` with the saved model.

    The output CSV includes the columns the Streamlit dashboard expects.
    """
    model_path = Path(model_path)
    input_path = Path(input_path)
    output_path = Path(output_path)

    payload = data_io.load_joblib(model_path)
    model = payload["model"] if isinstance(payload, dict) else payload
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    threshold = float(
        threshold if threshold is not None
        else metadata.get("threshold", config.DEFAULT_THRESHOLD)
    )
    log.info("model=%s  threshold=%.2f", model_path.name, threshold)

    df_raw = data_io.read_csv(input_path)
    df = feature_engineering.prepare_features(df_raw)

    num_cols = metadata.get("num_cols")
    cat_cols = metadata.get("cat_cols")
    if num_cols and cat_cols:
        cols = [c for c in num_cols + cat_cols if c in df.columns]
        X = df[cols]
    else:
        # Fallback if metadata is missing.
        n, c = feature_engineering.split_feature_columns(df)
        X = df[n + c]

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    segments = pd.Series(proba, index=df.index).map(classify_risk)

    out = pd.DataFrame({
        "churn_probability": proba,
        "churn_prediction":  preds,
        "risk_segment":      segments,
    })

    # Carry forward useful columns the dashboard tabs need.
    if "user_id" in df_raw.columns:
        out.insert(0, "user_id", df_raw["user_id"].values)
    if config.TARGET in df_raw.columns:
        out["actual_churn"] = df_raw[config.TARGET].astype(int).values
    if "revenue" in df_raw.columns:
        out["revenue"] = df_raw["revenue"].values
    if "region" in df_raw.columns:
        out["region"] = df_raw["region"].astype(str).str.strip().str.upper().values

    return data_io.write_csv(out, output_path)
