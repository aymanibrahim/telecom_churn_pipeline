# Telecom Customer Churn — REST API Reference

> **Audience**: reviewers who want to query the live prediction service or
> reproduce results against their local environment.

## Live endpoints

| Service | URL | Notes |
|---|---|---|
| **Flask REST API** (Azure) | https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io | Cold-start ≤ 30 s when idle |
| **Swagger UI** (interactive) | https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/ | Auto-generated from OpenAPI 3.0 |
| **OpenAPI spec** | https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/openapi.json | Machine-readable |
| **Streamlit dashboard** (Azure) | https://telechurn-streamlit.thankfulsand-f5821563.eastus.azurecontainerapps.io | Read-only KPIs |
| **Streamlit dashboard** (Cloud) | https://telecom-churn-aerqyhwebuhgi3dc527hvv.streamlit.app | Same data, redundant deploy |
| **GitHub source** | https://github.com/Mohamedhassanofficial/Telecom-Churn | Public, MIT |

## Authentication

The demo API is **open** — no token required. For production, wrap each route
with `@require_api_key` and serve the key via an Azure App Configuration secret.

## Endpoints

### `GET /health`

Liveness / readiness probe.

| | |
|---|---|
| **Method** | `GET` |
| **Body** | — |
| **Response** | `application/json` |
| **200** | `{"status":"ok","model_loaded":true}` |
| **503** | model failed to load at startup |

```bash
curl -s https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/health
# {"model_loaded":true,"status":"ok"}
```

```python
import requests
r = requests.get(
    "https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/health"
)
assert r.json()["model_loaded"] is True
```

---

### `GET /model-info`

Metadata of the currently-loaded model: version, threshold, training metrics,
feature columns.

```bash
curl -s https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/model-info | jq
```

Sample response:

```json
{
  "model_loaded": true,
  "threshold": 0.35,
  "metadata": {
    "version": "20260518_202422",
    "model_name": "lightgbm",
    "roc_auc": 0.9297,
    "f1": 0.6917,
    "smote": true,
    "calibrated": true,
    "num_cols": ["montant", "frequence_rech", "revenue", "..."],
    "cat_cols": ["region", "top_pack"],
    "trained_at": "2026-05-18T20:24:22Z"
  }
}
```

---

### `POST /predict`

Score a single customer.

| | |
|---|---|
| **Method** | `POST` |
| **Headers** | `Content-Type: application/json` |
| **Body** | JSON object — any subset of the trained-on features (missing → imputed) |
| **Response** | JSON object with `churn_probability`, `churn_prediction`, `risk_segment`, `threshold` |
| **400** | malformed payload |
| **503** | model not loaded |

```bash
curl -X POST \
  https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/predict \
  -H "Content-Type: application/json" \
  -d '{
    "revenue":     50,
    "regularity":  15,
    "frequence":   10,
    "data_volume": 1000,
    "region":      "DAKAR",
    "montant":     100,
    "frequence_rech": 5,
    "top_pack":    "Pack 5"
  }'
```

Response:

```json
{
  "churn_probability": 0.1765,
  "churn_prediction":  0,
  "risk_segment":      "Low",
  "threshold":         0.29
}
```

```python
import requests

payload = {
    "revenue": 50, "regularity": 15, "frequence": 10,
    "data_volume": 1000, "region": "DAKAR",
    "montant": 100, "frequence_rech": 5, "top_pack": "Pack 5",
}
r = requests.post(
    "https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/predict",
    json=payload,
    timeout=30,
)
print(r.json())
# {'churn_probability': 0.1765, 'churn_prediction': 0,
#  'risk_segment': 'Low', 'threshold': 0.29}
```

---

### `POST /predict_batch`

Same shape as `/predict`, but the body is a JSON **array** and the response
is an array of predictions in the same order.

```bash
curl -X POST \
  https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/predict_batch \
  -H "Content-Type: application/json" \
  -d '[
    {"revenue":50, "regularity":15, "region":"DAKAR"},
    {"revenue":5000, "regularity":60, "region":"DIOURBEL"}
  ]'
```

Response:

```json
[
  {"churn_probability": 0.18, "churn_prediction": 0,
   "risk_segment": "Low",    "threshold": 0.29},
  {"churn_probability": 0.04, "churn_prediction": 0,
   "risk_segment": "Low",    "threshold": 0.29}
]
```

---

### `GET /openapi.json`

The OpenAPI 3.0 spec for the service. Drop it into Postman / Insomnia /
Swagger Editor.

---

### `GET /`

Interactive Swagger-UI rendered from the OpenAPI spec. Open it in a browser.

---

## Field reference (request payload)

All fields are **optional**; missing fields are imputed in-pipeline. Provide
as many as you have for the best probability estimate.

| Field | Type | Range / values | Description |
|---|---|---|---|
| `region` | string | `DAKAR`, `THIES`, `SAINT-LOUIS`, `DIOURBEL`, `KAOLACK`, `LOUGA`, `FATICK`, `KOLDA`, `MATAM`, `TAMBACOUNDA`, `KAFFRINE`, `SEDHIOU`, `KEDOUGOU`, `ZIGUINCHOR` | Senegal administrative region (ADM1) |
| `tenure` | string | `D 3-6 month`, `E 6-9 month`, `F 9-12 month`, `G 12-15 month`, `H 15-18 month`, `I 18-21 month`, `J 21-24 month`, `K > 24 month` | Customer tenure bucket |
| `top_pack` | string | e.g. `Pack 5`, `Data:1000` | Most-frequently-purchased product |
| `montant` | number ≥ 0 | float32 | Total recharge amount (XOF) |
| `frequence_rech` | number ≥ 0 | float32 | Recharge frequency in window |
| `revenue` | number ≥ 0 | float32 | Total revenue (XOF) |
| `arpu_segment` | number | float32 | Average revenue per user (XOF) |
| `frequence` | number ≥ 0 | float32 | Activity frequency |
| `data_volume` | number ≥ 0 | float32 | Data consumed (MB) |
| `on_net` | number ≥ 0 | float32 | On-network calls count |
| `orange` | number ≥ 0 | float32 | Calls to Orange |
| `tigo` | number ≥ 0 | float32 | Calls to Tigo |
| `regularity` | int ≥ 0, ≤ 62 | int16 | Days active in observation window |
| `freq_top_pack` | number ≥ 0 | float32 | Top-pack usage frequency |
| `region_tower_count` | int ≥ 0 | derived | OpenCellID towers in customer's region |
| `region_avg_range` | number | derived | Mean tower range in region (m) |
| `region_coverage_index` | number | derived | tower_count × avg_range |
| `region_network_quality_score` | number | derived | composite quality score |
| `arr_coverage_index`, `arr_network_quality_score` | number | derived | Arrondissement-level KPIs |
| `department_coverage_index`, ... | number | derived | Department-level KPIs |

Drop the `_index` / KPI columns when calling from a thin client — the API
will impute defaults. Region alone gives a meaningful prediction because
the trained pipeline encodes regional churn rates via `TargetEncoder`.

## Cold start

The Azure Container App scales to zero when idle. The first request after
a quiet period may take **15-30 s** to return as the container boots and
loads the joblib. Subsequent requests are sub-second.

To warm a container ahead of a demo:

```bash
for i in $(seq 1 3); do
  curl -s https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/health
done
```

## How to reproduce a prediction locally

```bash
git clone https://github.com/Mohamedhassanofficial/Telecom-Churn.git
cd Telecom-Churn
pip install -r requirements.txt

python - <<'EOF'
import joblib, pandas as pd
from src.inference import classify_risk

payload = joblib.load("models/churn_model.joblib")
model, meta = payload["model"], payload["metadata"]
threshold = float(meta["threshold"])

row = pd.DataFrame([{
    "revenue": 50, "regularity": 15, "frequence": 10,
    "data_volume": 1000, "region": "DAKAR",
    "montant": 100, "frequence_rech": 5, "top_pack": "Pack 5",
}])
# Fill any missing trained-on cols with NaN (in-pipeline imputer handles them)
for c in meta["num_cols"] + meta["cat_cols"]:
    if c not in row.columns:
        row[c] = pd.NA
row = row[meta["num_cols"] + meta["cat_cols"]]

p = model.predict_proba(row)[:, 1][0]
print({
    "churn_probability": float(p),
    "churn_prediction":  int(p >= threshold),
    "risk_segment":      classify_risk(float(p)),
    "threshold":         threshold,
})
EOF
```

The probability the API returns and the probability this script returns
should be identical to ~8 decimal places (same model, same input).

## Verify the canonical run

```bash
python scripts/compare_dag_vs_notebook.py \
  --dag-csv     outputs/predictions/churn_predictions.csv \
  --baseline    outputs/predictions/churn_predictions_notebook.csv \
  --output-json outputs/dag_vs_notebook_comparison.json
```

`OVERALL: PASS` confirms the local run matches the canonical 100 k baseline
shipped in the repo.

## Error handling

| HTTP | Cause | Action |
|---|---|---|
| 200 | success | — |
| 400 | malformed JSON / wrong shape | check curl `Content-Type: application/json` and the body |
| 503 | model failed to load (rare) | retry; if persistent, check `/model-info` |
| 504 (gateway) | container cold-start in progress | retry after 30 s |
