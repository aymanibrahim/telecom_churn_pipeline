# 📈 Telecom Customer Churn — End-to-End Pipeline

[![CI](https://github.com/Mohamedhassanofficial/Telecom-Churn/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohamedhassanofficial/Telecom-Churn/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Airflow](https://img.shields.io/badge/airflow-2.10.4-017CEE.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A production-grade machine learning pipeline that predicts customer churn for the Senegalese telecom operator **Expresso**, fusing customer behaviour data with **OpenCellID** network-tower coverage. Orchestrated with **Apache Airflow**, served via a **Streamlit** dashboard with an interactive Senegal map.

---

## 🌐 Live deployments (for reviewers)

| Surface | URL | Status |
|---|---|---|
| **Streamlit dashboard** (Streamlit Cloud) | https://telecom-churn-aerqyhwebuhgi3dc527hvv.streamlit.app | ✅ |
| **Streamlit dashboard** (Azure backup) | https://telechurn-streamlit.thankfulsand-f5821563.eastus.azurecontainerapps.io | ✅ |
| **Flask REST API** (Azure) | https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io | ✅ |
| **Swagger UI** (interactive API) | https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/ | ✅ |
| **GitHub repo** (public, MIT) | https://github.com/Mohamedhassanofficial/Telecom-Churn | ✅ |

### Documentation for reviewers / colleagues
- **[`docs/API.md`](docs/API.md)** — full REST API reference (endpoints, schemas, curl + Python examples, field reference)
- **[`docs/REVIEWERS_GUIDE.md`](docs/REVIEWERS_GUIDE.md)** — 60-second / 5-minute / 2-minute reproduction recipes
- **[`docs/PIPELINE_OUTPUTS.md`](docs/PIPELINE_OUTPUTS.md)** — stage-by-stage outputs of the data pipeline (every script's artefact, with charts)
- **[`docs/SCREENSHOTS.md`](docs/SCREENSHOTS.md)** — dashboard + ML evaluation visuals

### One-liner sanity check

```bash
curl -X POST https://telechurn-flask.thankfulsand-f5821563.eastus.azurecontainerapps.io/predict \
  -H "Content-Type: application/json" \
  -d '{"revenue":50,"regularity":15,"frequence":10,"data_volume":1000,"region":"DAKAR","montant":100,"frequence_rech":5,"top_pack":"Pack 5"}'
# {"churn_probability":0.1765,"churn_prediction":0,"risk_segment":"Low","threshold":0.29}
```

---

## 🧭 Pipeline overview

```
sample_expresso ─┐
                 ├─→ build_telecom_churn_100k ─┐
build_opencellid ┤                              ├─→ collect_telecom_paths ─→ run_eda ─→ train_model ─→ predict
                 └─→ build_telecom_churn_full ─┘
```

The DAG produces **both scales** in one run: `build_telecom_churn_100k` feeds the 100k Expresso sample → `telecom_churn_100k.csv` (~100k rows); `build_telecom_churn_full` feeds the raw 2 M Expresso → `telecom_churn.csv` (~2.15 M rows). Downstream tasks pick the scale via the `USE_FULL_DATASET` Airflow Variable.

| # | Stage | Source notebook → Python module |
|---|---|---|
| 1 | Sample Expresso (2M → 100k stratified) | `notebooks/generate_expresso_sample_100k.ipynb` → `src/sampling.py` |
| 2 | Build OpenCellID Senegal (90-day window, MCC=608, MNC=3) | `notebooks/build_opencellid_senegal_90d_dataset.ipynb` → `src/geo.py` |
| 3 | Spatial join + KPI aggregation (region / department / arrondissement) | `notebooks/build_telecom_churn_dataset.ipynb` → `src/geo.py` + `src/network_kpis.py` |
| 4 | Headless EDA report (PNGs + JSON summary) | `notebooks/Telecom Customer Churn - EDA - Project Team A.ipynb` → `src/eda.py` |
| 5 | Train LightGBM (Calibrated + SMOTE + SHAP) | `notebooks/Telecom Customer Churn - Training, Inference & Evaluation - Project Team A.ipynb` → `src/modeling.py` |
| 6 | Batch inference + risk segmentation | same notebook → `src/inference.py` |

---

## 🚀 Quickstart

### Docker (recommended)

```bash
git clone https://github.com/Mohamedhassanofficial/Telecom-Churn.git
cd Telecom-Churn
cp .env.example .env

# Download raw data (Kaggle CLI required for expresso.csv + Africa_towers.csv)
python scripts/00_download_raw.py

docker compose build
docker compose up airflow-init
docker compose up
```

Open:
- **Airflow UI**: http://localhost:8080  (login: `airflow` / `airflow`)
- **Streamlit dashboard**: http://localhost:8501

Trigger the DAG `telecom_churn_production_pipeline` and watch the 6 tasks succeed; the dashboard auto-refreshes when `outputs/predictions/churn_predictions.csv` is rewritten.

### Local Windows / WSL

```bash
pip install -r requirements-pipeline.txt
pip install -r dashboards/requirements-dashboards.txt

set CHURN_BASE_DIR=C:\path\to\Telecom-Churn          # Windows
# export CHURN_BASE_DIR=$(pwd)                        # macOS / Linux

python scripts/05_train_model.py --no-shap
python scripts/06_predict.py
streamlit run dashboards/churn_dashboard_app.py
```

---

## 🗂 Project layout

```
Telecom-Churn/
├── dags/churn_pipeline.py            # Airflow TaskFlow DAG
├── scripts/                          # CLI entrypoints (00–06)
├── src/                              # Shared library (14 modules)
│   ├── paths.py                      # CHURN_BASE_DIR-driven paths
│   ├── config.py                     # YAML-loaded constants
│   ├── logging_setup.py              # Centralised logging
│   ├── validation.py                 # Pandera-style schema checks
│   ├── idempotency.py                # @skip_if_fresh decorator
│   ├── sampling.py / geo.py / ...
│   └── modeling.py / inference.py
├── tests/                            # Pytest (29 unit tests)
├── dashboards/churn_dashboard_app.py # Streamlit + Senegal map
├── notebooks/                        # Original 15 reference notebooks
├── config/config.yaml                # Hyperparameters + thresholds
├── docker-compose.yaml               # Airflow + Postgres + Streamlit
├── Dockerfile.airflow
├── pyproject.toml                    # ruff + mypy + pytest config
├── Makefile                          # make smoke / test / docker-up
├── requirements-pipeline.txt
└── .github/workflows/ci.yml          # GitHub Actions
```

---

## ✨ Engineering features

| Feature | Why it matters |
|---|---|
| **TaskFlow API DAG** | Type-hinted Python, automatic XCom, fewer footguns than PythonOperator |
| **Schema validation between stages** | Catches data drift the moment it appears |
| **Idempotent task runs** | `@skip_if_fresh` makes daily runs cheap; bypass with `CHURN_FORCE_REBUILD=1` |
| **Calibrated LightGBM + SMOTE + Target encoding** | Pipeline avoids leakage and produces well-calibrated probabilities |
| **SHAP explainability** | Per-feature contribution plot saved with every training run |
| **MLflow tracking** | Auto-enables when `MLFLOW_TRACKING_URI` is set; otherwise no-op |
| **Streamlit Senegal map** | Choropleth of churn risk per ADM1 region |
| **Pytest suite + GitHub Actions CI** | ruff + mypy + 29 unit tests on every push |
| **Docker Compose for one-command stand-up** | Postgres + Airflow scheduler/webserver + Streamlit sidecar |

---

## 📊 Model performance

| Metric | Value |
|---|---|
| Accuracy | 0.87 |
| Precision | 0.63 |
| Recall | 0.78 |
| F1 | 0.69 |
| ROC-AUC | 0.85 |

Trained on the 100k stratified sample with a probability threshold of **0.42** (auto-tuned on the validation set for max F1).

---

## 🔧 Configuration

All non-path constants live in `config/config.yaml`:

```yaml
opencellid:
  mcc: 608          # Senegal
  mnc: 3            # Expresso
  window_days: 90

model:
  default_threshold: 0.42
  calibration_method: isotonic
  smote_k_neighbors: 5
```

Paths are resolved from `CHURN_BASE_DIR` (env var) by `src/paths.py` — same code runs on Windows, Linux, and inside the Docker Airflow container.

---

## 🧪 Verification

```bash
make smoke       # compile + import smoke + lightweight unit tests
make test        # full pytest suite
make docker-up   # spin up Airflow + Streamlit
```

CI runs the same checks plus a DAG parse on every push:

```yaml
# .github/workflows/ci.yml
- ruff check src scripts dags tests
- mypy src --ignore-missing-imports
- python -m compileall -q src scripts dags tests
- pytest --cov=src --cov-report=term-missing
- airflow dags list-import-errors
```

---

## 📚 Documentation

- **`README_PIPELINE.md`** — extended setup notes
- **`guides/`** — original brief, dataset definitions, ML methodology PDFs
- **`notebooks/`** — original Jupyter notebooks kept as reference

---

## 📝 License

MIT — see `LICENSE` (TODO: add).

---

> Built with ❤️ from notebooks → modular Python → Airflow.
> Source notebooks are in `notebooks/`; the converted modules in `src/`.
