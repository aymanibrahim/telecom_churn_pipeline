"""
app.py
------
Flask REST API — Telecom Customer Churn Prediction

Endpoints
~~~~~~~~~
GET  /health          — Liveness probe; returns model version + status.
POST /predict         — Single-customer churn probability inference.
POST /predict/batch   — Batch inference (list of customer records).
GET  /model/info      — Returns model metadata and performance metrics.
GET  /predictions     — Returns the latest batch predictions summary.

Authentication
~~~~~~~~~~~~~~
Set the ``API_KEY`` environment variable.  Requests must include the
header ``X-API-Key: <key>``.  Disabled when ``API_KEY`` is empty (dev).

Usage
~~~~~
    POST /predict
    Content-Type: application/json

    {
      "revenue":    5000.0,
      "regularity": 25,
      "frequence":  12.0,
      "data_volume": 1500.0
    }

    Response 200:
    {
      "churn_probability": 0.23,
      "churn_prediction":  0,
      "risk_segment":      "Low",
      "recommended_action":"Routine monitoring — no immediate action",
      "model_version":     "20260411_174945",
      "threshold":         0.42
    }
"""

from __future__ import annotations

import json
import os
from functools import wraps
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# ── Paths (injected via env vars in Docker) ─────────────────────────────────
MODELS_DIR  = Path(os.environ.get("MODELS_DIR",  "/app/models"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "/app/outputs"))

MODEL_PATH     = MODELS_DIR  / "churn_model.joblib"
MODEL_INFO_PATH = OUTPUTS_DIR / "model_info.json"
PREDICTIONS_PATH = OUTPUTS_DIR / "churn_predictions.csv"

API_KEY = os.environ.get("API_KEY", "")    # Empty = auth disabled (dev only)

# ── Risk tier thresholds (must match config.py) ──────────────────────────────
RISK_TIERS = {
    "High":   (0.70, 1.01),
    "Medium": (0.40, 0.70),
    "Low":    (0.00, 0.40),
}
ACTION_MAP = {
    "High":   "Immediate personalised retention offer",
    "Medium": "Targeted SMS / push notification campaign",
    "Low":    "Routine monitoring — no immediate action",
}

app = Flask(__name__)

# ── Module-level model cache ──────────────────────────────────────────────────
_model    = None
_metadata = None


def _load_model() -> tuple:
    """Load the model and metadata from disk (cached in module globals)."""
    global _model, _metadata          # noqa: PLW0603
    if _model is None:
        payload   = joblib.load(str(MODEL_PATH))
        _model    = payload["model"]
        _metadata = payload.get("metadata", {})
        app.logger.info(
            "Model loaded: version=%s  threshold=%.2f  ROC-AUC=%.4f",
            _metadata.get("version", "?"),
            _metadata.get("threshold", 0.5),
            _metadata.get("roc_auc", float("nan")),
        )
    return _model, _metadata


def _reload_model() -> None:
    """Force a model reload (called when the DAG signals a model update)."""
    global _model, _metadata          # noqa: PLW0603
    _model    = None
    _metadata = None
    _load_model()
    app.logger.info("Model reloaded successfully.")


def _classify_risk(prob: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "Low"


# ── Authentication decorator ──────────────────────────────────────────────────
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if API_KEY and request.headers.get("X-API-Key") != API_KEY:
            return jsonify({"error": "Unauthorized — invalid or missing API key."}), 401
        return f(*args, **kwargs)
    return decorated


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENDPOINTS                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.get("/health")
def health():
    """Liveness probe."""
    try:
        _, meta = _load_model()
        return jsonify({
            "status":        "ok",
            "model_version": meta.get("version", "unknown"),
            "model_path":    str(MODEL_PATH),
        }), 200
    except Exception as exc:          # noqa: BLE001
        return jsonify({"status": "error", "detail": str(exc)}), 503


@app.post("/predict")
@require_api_key
def predict():
    """
    Single-customer churn probability prediction.

    Body fields (all optional — missing values are imputed by the pipeline)
    -----------------------------------------------------------------------
    revenue       float   Monthly revenue.
    regularity    float   Activity regularity (0-90 day window).
    frequence     float   Number of transactions.
    data_volume   float   Mobile data volume (MB).
    montant       float   Top-up amount.
    on_net        float   On-network calls.
    orange        float   Calls to Orange network.
    tigo          float   Calls to Tigo network.
    region        str     Senegal region name.
    top_pack      str     Most active subscription pack.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        model, meta = _load_model()
        threshold   = meta.get("threshold", 0.5)

        input_df  = pd.DataFrame([data])
        prob      = float(model.predict_proba(input_df)[:, 1][0])
        pred      = int(prob >= threshold)
        risk      = _classify_risk(prob)

        return jsonify({
            "churn_probability":  round(prob, 4),
            "churn_prediction":   pred,
            "risk_segment":       risk,
            "recommended_action": ACTION_MAP.get(risk, "Monitor"),
            "model_version":      meta.get("version", "unknown"),
            "threshold":          threshold,
        }), 200

    except Exception as exc:          # noqa: BLE001
        app.logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


@app.post("/predict/batch")
@require_api_key
def predict_batch():
    """
    Batch inference for a list of customer records.

    Body: ``{"records": [ {customer_1}, {customer_2}, ... ]}``

    Returns a list of prediction objects in the same order.
    """
    data = request.get_json(silent=True)
    if not data or "records" not in data:
        return jsonify({"error": "Body must contain a 'records' list."}), 400

    records = data["records"]
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "'records' must be a non-empty list."}), 400

    if len(records) > 10_000:
        return jsonify({"error": "Batch size exceeds 10,000 records."}), 400

    try:
        model, meta = _load_model()
        threshold   = meta.get("threshold", 0.5)

        input_df = pd.DataFrame(records)
        probs    = model.predict_proba(input_df)[:, 1]
        preds    = (probs >= threshold).astype(int)

        results = []
        for prob, pred in zip(probs, preds):
            prob = float(prob)
            risk = _classify_risk(prob)
            results.append({
                "churn_probability":  round(prob, 4),
                "churn_prediction":   int(pred),
                "risk_segment":       risk,
                "recommended_action": ACTION_MAP.get(risk, "Monitor"),
            })

        return jsonify({
            "count":         len(results),
            "model_version": meta.get("version", "unknown"),
            "threshold":     threshold,
            "predictions":   results,
        }), 200

    except Exception as exc:          # noqa: BLE001
        app.logger.exception("Batch prediction error")
        return jsonify({"error": str(exc)}), 500


@app.get("/model/info")
@require_api_key
def model_info():
    """Return model metadata and performance metrics."""
    if MODEL_INFO_PATH.exists():
        with open(str(MODEL_INFO_PATH)) as fh:
            info = json.load(fh)
        return jsonify(info), 200

    try:
        _, meta = _load_model()
        return jsonify(meta), 200
    except Exception as exc:          # noqa: BLE001
        return jsonify({"error": str(exc)}), 503


@app.post("/model/reload")
@require_api_key
def reload_model():
    """
    Force a model reload.  Called by the Airflow HttpOperator at the end of
    each DAG run so the API always serves the latest trained model.
    """
    try:
        _reload_model()
        _, meta = _load_model()
        return jsonify({
            "status":        "reloaded",
            "model_version": meta.get("version", "unknown"),
        }), 200
    except Exception as exc:          # noqa: BLE001
        app.logger.exception("Model reload error")
        return jsonify({"error": str(exc)}), 500


@app.get("/predictions")
@require_api_key
def predictions_summary():
    """Return a summary of the latest batch predictions."""
    if not PREDICTIONS_PATH.exists():
        return jsonify({"error": "Predictions file not found. Run the pipeline first."}), 404

    try:
        df = pd.read_csv(str(PREDICTIONS_PATH))
        summary = {
            "total_customers": len(df),
            "predicted_churn": int(df["churn_prediction"].sum()) if "churn_prediction" in df.columns else None,
            "avg_churn_probability": round(float(df["churn_probability"].mean()), 4),
            "risk_distribution": df["risk_segment"].value_counts().to_dict() if "risk_segment" in df.columns else {},
            "action_distribution": df["action_segment"].value_counts().to_dict() if "action_segment" in df.columns else {},
        }
        return jsonify(summary), 200
    except Exception as exc:          # noqa: BLE001
        app.logger.exception("Predictions summary error")
        return jsonify({"error": str(exc)}), 500


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STARTUP                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Pre-load model on startup so the first request is not slow
with app.app_context():
    try:
        _load_model()
    except Exception as exc:          # noqa: BLE001
        app.logger.warning("Model not loaded on startup: %s", exc)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
