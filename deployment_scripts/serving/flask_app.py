"""Flask REST API for the Telecom Customer Churn pipeline.

Endpoints
---------
GET  /              Swagger-UI auto-generated from the OpenAPI spec.
GET  /health        Liveness probe. Returns {"status":"ok","model_loaded":bool}.
GET  /model-info    Metadata from the joblib payload (version, threshold, metrics).
POST /predict       Score a single customer dict, return prob + segment.
POST /predict_batch Score an array of customer dicts.

Model loading
-------------
On startup the app loads ``churn_model.joblib`` from one of:

1. ``CHURN_MODEL_PATH`` env var (explicit local path), OR
2. ``AZURE_BLOB_URL`` env var (downloads with ``requests``), OR
3. ``/app/models/churn_model.joblib`` (baked into the image), OR
4. ``<project_root>/models/churn_model.joblib`` (local dev).

The joblib payload must be the dict produced by ``src.modeling.train``:
``{"model": <CalibratedClassifierCV>, "metadata": {"threshold": float, ...}}``.
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)-15s | %(message)s",
)
log = logging.getLogger("flask_app")


# ---------------------------------------------------------------------------
# Model loading (runs once at import time, cached in process memory)
# ---------------------------------------------------------------------------
def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    if env := os.environ.get("CHURN_MODEL_PATH"):
        candidates.append(Path(env))
    candidates.append(Path("/app/models/churn_model.joblib"))
    candidates.append(Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib")
    return candidates


def _download_from_blob(url: str) -> bytes:
    import requests
    log.info("Downloading model from %s", url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    log.info("Downloaded %.1f KB", len(r.content) / 1024)
    return r.content


def _load_model() -> tuple[Any, dict]:
    blob_url = os.environ.get("AZURE_BLOB_URL")
    if blob_url:
        try:
            data = _download_from_blob(blob_url)
            payload = joblib.load(io.BytesIO(data))
            log.info("Loaded model from blob: %s", blob_url)
            return payload["model"], payload.get("metadata", {})
        except Exception as exc:
            log.warning("Blob load failed (%s) - falling back to local paths", exc)

    for path in _candidate_paths():
        if path.exists():
            payload = joblib.load(path)
            log.info("Loaded model from %s", path)
            return payload["model"], payload.get("metadata", {})

    raise FileNotFoundError(
        "No churn_model.joblib found. Set CHURN_MODEL_PATH or AZURE_BLOB_URL, "
        f"or place the file at one of: {[str(p) for p in _candidate_paths()]}"
    )


try:
    MODEL, METADATA = _load_model()
    MODEL_LOADED = True
except Exception as exc:
    log.error("Model load failed at startup: %s", exc)
    MODEL, METADATA = None, {}
    MODEL_LOADED = False


THRESHOLD = float(METADATA.get("threshold", 0.42))


def _risk_segment(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Feature handling
# ---------------------------------------------------------------------------
NUM_COLS = METADATA.get("num_cols", [])
CAT_COLS = METADATA.get("cat_cols", [])


def _make_frame(records: list[dict]) -> pd.DataFrame:
    """Coerce request payloads to the DataFrame the pipeline expects.

    Missing columns are filled with NaN; the in-pipeline SimpleImputer handles them.
    """
    df = pd.DataFrame(records)
    expected = NUM_COLS + CAT_COLS
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected]
    # Numeric coercion for resilience to JSON strings.
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _predict_rows(records: list[dict]) -> list[dict]:
    df = _make_frame(records)
    probs = MODEL.predict_proba(df)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)
    return [
        {
            "churn_probability": float(p),
            "churn_prediction": int(pred),
            "risk_segment": _risk_segment(float(p)),
            "threshold": THRESHOLD,
        }
        for p, pred in zip(probs, preds, strict=True)
    ]


# ---------------------------------------------------------------------------
# OpenAPI spec (served at /openapi.json) — drives Swagger-UI at /
# ---------------------------------------------------------------------------
OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "Telecom Customer Churn — Prediction API",
        "version": "1.0.0",
        "description": (
            "Score individual or batched telecom customer records for churn "
            "risk. Loads a calibrated LightGBM pipeline trained on the "
            "Expresso + OpenCellID joined dataset."
        ),
    },
    "paths": {
        "/health": {
            "get": {
                "summary": "Liveness probe",
                "responses": {"200": {"description": "OK"}},
            }
        },
        "/model-info": {
            "get": {
                "summary": "Model metadata (version, threshold, metrics)",
                "responses": {"200": {"description": "OK"}},
            }
        },
        "/predict": {
            "post": {
                "summary": "Score one customer",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "example": {
                                    "revenue": 50,
                                    "regularity": 15,
                                    "frequence": 10,
                                    "data_volume": 1000,
                                    "region": "DAKAR",
                                },
                            }
                        }
                    },
                },
                "responses": {"200": {"description": "Prediction"}},
            }
        },
        "/predict_batch": {
            "post": {
                "summary": "Score a list of customers",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "array",
                                "items": {"type": "object"},
                                "example": [
                                    {"revenue": 50, "regularity": 15},
                                    {"revenue": 5000, "regularity": 60},
                                ],
                            }
                        }
                    },
                },
                "responses": {"200": {"description": "Array of predictions"}},
            }
        },
    },
}


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_LOADED})


@app.route("/model-info", methods=["GET"])
def model_info():
    if not MODEL_LOADED:
        return jsonify({"error": "model not loaded"}), 503
    safe_meta = {
        k: v for k, v in METADATA.items()
        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }
    return jsonify({
        "model_loaded": True,
        "threshold": THRESHOLD,
        "metadata": safe_meta,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "model not loaded"}), 503
    try:
        record = request.get_json(force=True) or {}
        if not isinstance(record, dict):
            return jsonify({"error": "expected a JSON object, got list — use /predict_batch"}), 400
        result = _predict_rows([record])[0]
        return jsonify(result)
    except Exception as exc:
        log.exception("predict failed")
        return jsonify({"error": str(exc)}), 400


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if not MODEL_LOADED:
        return jsonify({"error": "model not loaded"}), 503
    try:
        records = request.get_json(force=True)
        if not isinstance(records, list):
            return jsonify({"error": "expected a JSON array"}), 400
        if not records:
            return jsonify([])
        return jsonify(_predict_rows(records))
    except Exception as exc:
        log.exception("predict_batch failed")
        return jsonify({"error": str(exc)}), 400


@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    return jsonify(OPENAPI_SPEC)


@app.route("/", methods=["GET"])
def swagger_ui():
    return """<!DOCTYPE html>
<html><head><title>Telecom Churn API</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head><body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
SwaggerUIBundle({ url: "/openapi.json", dom_id: "#swagger-ui" });
</script>
</body></html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
