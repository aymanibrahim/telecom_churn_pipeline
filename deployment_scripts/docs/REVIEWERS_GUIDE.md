# Reviewer's Guide — Telecom Customer Churn

> **Goal**: a colleague can sanity-check the live service, reproduce a
> prediction locally, and confirm output parity with the canonical run —
> all in under 10 minutes.

## 0. Live URLs

| Surface | URL |
|---|---|
| GitHub repo (public, MIT) | https://github.com/Mohamedhassanofficial/Telecom-Churn |
| Streamlit dashboard (Cloud) | https://telecom-churn-aerqyhwebuhgi3dc527hvv.streamlit.app |
| Streamlit dashboard (Azure backup) | https://telechurn-streamlit.thankfulsand-f5821563.eastus.azurecontainerapps.io |
| Flask REST API | https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io |
| Swagger UI | https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/ |
| Airflow UI (local only) | http://localhost:8080 — `docker compose up` first |

## 1. Sanity check the API (60 seconds)

```bash
# 1. Liveness
curl -s https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/health
# Expected: {"model_loaded":true,"status":"ok"}

# 2. Model metadata
curl -s https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/model-info \
  | python -m json.tool | head -20

# 3. One prediction
curl -s -X POST https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/predict \
  -H "Content-Type: application/json" \
  -d '{"revenue":50,"regularity":15,"frequence":10,"data_volume":1000,"region":"DAKAR","montant":100,"frequence_rech":5,"top_pack":"Pack 5"}'
# Expected: {"churn_probability":0.18,"churn_prediction":0,"risk_segment":"Low","threshold":0.29}
```

If you hit a cold-start, the first call returns after ≈ 15-30 s. The
following calls are sub-second.

## 2. Reproduce a prediction locally (5 minutes)

```bash
# 1. Clone (public repo, no token needed)
git clone https://github.com/Mohamedhassanofficial/Telecom-Churn.git
cd Telecom-Churn

# 2. Install only what the dashboard / model need
pip install -r requirements.txt

# 3. Score the same row the curl above sends
python - <<'EOF'
import joblib, pandas as pd
from src.inference import classify_risk

payload = joblib.load("models/churn_model.joblib")
model, meta = payload["model"], payload["metadata"]
t = float(meta["threshold"])

row = pd.DataFrame([{
    "revenue": 50, "regularity": 15, "frequence": 10,
    "data_volume": 1000, "region": "DAKAR",
    "montant": 100, "frequence_rech": 5, "top_pack": "Pack 5",
}])
for c in meta["num_cols"] + meta["cat_cols"]:
    if c not in row.columns:
        row[c] = pd.NA
row = row[meta["num_cols"] + meta["cat_cols"]]

prob = float(model.predict_proba(row)[:, 1][0])
print({
    "churn_probability": prob,
    "churn_prediction":  int(prob >= t),
    "risk_segment":      classify_risk(prob),
    "threshold":         t,
})
EOF
```

The local print and the API response should be identical to ~8 decimal
places. If they're not, you're on a stale repo — `git pull` and retry.

## 3. Verify the full predictions CSV matches (2 minutes)

The repo ships a canonical baseline `outputs/predictions/churn_predictions_notebook.csv`
(100 000 rows). When you re-run the pipeline you should get a bit-identical
file.

```bash
python scripts/compare_dag_vs_notebook.py \
  --dag-csv     outputs/predictions/churn_predictions.csv \
  --baseline    outputs/predictions/churn_predictions_notebook.csv
```

Expected:

```
DAG CSV:   outputs/predictions/churn_predictions.csv     (md5 d8abfb0c)
Baseline:  outputs/predictions/churn_predictions_notebook.csv (md5 d8abfb0c)
Rows:      dag=100000  baseline=100000
-----------------------------------------------------------
  churn_prediction_agreement: 1.0
  churn_probability_mean_abs_delta: 0.0
  risk_segment_max_delta: 0.0
-----------------------------------------------------------
OVERALL: PASS
```

The report is written to `outputs/dag_vs_notebook_comparison.json` (also
on GitHub).

## 4. Pull the predictions CSV directly

Don't want to clone? Grab the file straight off the repo:

```bash
curl -O https://raw.githubusercontent.com/Mohamedhassanofficial/Telecom-Churn/main/outputs/predictions/churn_predictions_notebook.csv
```

```python
import pandas as pd
url = ("https://raw.githubusercontent.com/Mohamedhassanofficial/"
       "Telecom-Churn/main/outputs/predictions/churn_predictions_notebook.csv")
df = pd.read_csv(url)
print(df.head())
print(df["risk_segment"].value_counts())
```

## 5. Reproduce the entire pipeline end-to-end (~10 min on a laptop)

If you want to retrain and rescore from scratch:

```bash
git clone https://github.com/Mohamedhassanofficial/Telecom-Churn.git
cd Telecom-Churn
pip install -r requirements-pipeline.txt

# Stage the two large CSVs (see scripts/00_download_raw.py for sources):
#   - datasets/expresso/expresso.csv          (~248 MB, Kaggle)
#   - datasets/opencellid/Africa_towers.csv   (~259 MB, Kaggle)

python scripts/01_generate_expresso_sample.py
python scripts/02_build_opencellid_senegal.py
python scripts/03_build_telecom_churn.py
python scripts/04_run_eda.py
python scripts/05_train_model.py --no-shap
python scripts/06_predict.py
streamlit run dashboards/churn_dashboard_app.py
```

The same six steps run inside the Airflow DAG (`dags/churn_pipeline.py`).

## 6. Where to file findings

- **Bug / unexpected output**: open an Issue on GitHub
  https://github.com/Mohamedhassanofficial/Telecom-Churn/issues
- **Reproducibility question**: comment on the relevant commit
- **Model behaviour concerns**: include the request payload + the API
  response + the corresponding row from `churn_predictions_notebook.csv`

## 7. Common questions

**Q: The Streamlit Cloud URL shows "Error installing requirements".**
A: The build is still running. First-time builds take 5-7 min. Check
   "Manage app → Logs" or just use the Azure URL above.

**Q: The Azure Flask `/predict` returns 504 / takes 30 s.**
A: That's a cold start (Consumption tier scales to zero). The 2nd call
   is fast. To warm it: hit `/health` three times then make your call.

**Q: My local prediction differs from the API prediction.**
A: Your local repo is stale, OR your `requirements.txt` install pulled
   a different `scikit-learn` / `lightgbm` version. `pip freeze | diff`
   against `requirements.txt` to spot drift.

**Q: How do I run the Airflow DAG locally?**
A: `docker compose up -d` then open http://localhost:8080 (login
   `airflow / airflow`). Unpause `telecom_churn_production_pipeline`
   and click "Trigger DAG".
