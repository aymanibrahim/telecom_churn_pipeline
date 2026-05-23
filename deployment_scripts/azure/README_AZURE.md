# Azure Deployment — Telecom Customer Churn

Deploys the Flask REST API and Streamlit dashboard to **Azure Container Apps**.

## Architecture

```
                ┌─────────────────────────────┐
   user/browser │   *.azurecontainerapps.io   │
       │        │      (TLS terminated)       │
       │        └──────────────┬──────────────┘
       │                       │
       ▼                       ▼
  ┌─────────────────┐   ┌──────────────────┐
  │ Container App:  │   │ Container App:   │
  │   streamlit     │   │     flask        │
  │   :8501         │   │     :5000        │
  │   0.5 CPU/1Gi   │   │   0.25 CPU/0.5Gi │
  └────────┬────────┘   └────────┬─────────┘
           │                     │
           │     pulls image     │
           └──────────┬──────────┘
                      ▼
              ┌──────────────┐         ┌────────────────────┐
              │     ACR      │         │  Storage Account   │
              │ (Basic SKU)  │         │  blob "models/"    │
              │ telechurnacr │         │ churn_model.joblib │
              └──────────────┘         └────────────────────┘
```

All resources live in one resource group: `rg-telecom-churn` (East US by default).

## Cost expectations

| Resource | Idle | Light traffic |
|---|---|---|
| ACR Basic | $5/mo | $5/mo |
| Container Apps (scale to 0) | $0 | $5–15/mo |
| Storage Account (Standard LRS, <1 GB) | <$1/mo | <$1/mo |
| Log Analytics (free 5 GB/mo) | $0 | $0 |
| **Total** | **~$5/mo** | **~$15–25/mo** |

Teardown: `az group delete -n rg-telecom-churn --yes --no-wait` (nukes everything, stops the bill).

## Prerequisites

1. **Azure subscription** (free tier or pay-as-you-go).
2. **Docker** running locally — we run the Azure CLI from a Docker container so nothing needs to be installed natively.
3. **Bash** (Git Bash on Windows works; macOS/Linux native).
4. **The local `models/churn_model.joblib`** exists (run `python scripts/05_train_model.py` first if not).

## Quickstart

```bash
# 1. Log in to Azure (device-code flow — opens a browser)
docker run --rm -v "$HOME/.azure:/root/.azure" mcr.microsoft.com/azure-cli \
    az login --use-device-code

# 2. Deploy everything (provisions resources + builds images + creates apps)
bash azure/deploy.sh

# 3. The script prints the public FQDNs at the end:
#    Flask:     https://telechurn-flask-...azurecontainerapps.io/
#    Streamlit: https://telechurn-streamlit-...azurecontainerapps.io/
```

Re-running `deploy.sh` is safe (Bicep is idempotent + `az acr build` overwrites).

### Windows note

If you use Git Bash on Windows, replace `$HOME` with `$USERPROFILE` and quote the bind-mount path so MSYS doesn't mangle it:

```bash
docker run --rm -v "C:\Users\<you>\.azure:/root/.azure" mcr.microsoft.com/azure-cli \
    az login --use-device-code
```

The `deploy.sh` script does this automatically.

## What gets deployed

| Resource | Bicep name | Notes |
|---|---|---|
| Resource Group | `rg-telecom-churn` | Created by `deploy.sh`, not Bicep |
| Log Analytics workspace | `telechurn-logs` | Required by Container Apps Env |
| ACR | `telechurnacr<hash>` | Globally-unique 8-char suffix |
| Storage Account | `telechurnst<hash>` | Public Blob container `models` |
| Container Apps Env | `telechurn-env` | Consumption-tier |
| Container App: Flask | `telechurn-flask` | Ingress 5000 → external |
| Container App: Streamlit | `telechurn-streamlit` | Ingress 8501 → external |

## Verify the deployment

```bash
# Replace <fqdn> with the value printed by deploy.sh

# Flask
curl https://<flask-fqdn>/health
# {"status":"ok","model_loaded":true}

curl https://<flask-fqdn>/model-info
# {"model_loaded":true,"threshold":0.35,"metadata":{...}}

curl -X POST https://<flask-fqdn>/predict \
  -H "Content-Type: application/json" \
  -d '{"revenue":50,"regularity":15,"frequence":10,"data_volume":1000,"region":"DAKAR"}'
# {"churn_probability":0.18,"churn_prediction":0,"risk_segment":"Low","threshold":0.35}

# Swagger UI:
open https://<flask-fqdn>/                  # macOS
start https://<flask-fqdn>/                 # Windows
xdg-open https://<flask-fqdn>/              # Linux

# Streamlit
curl https://<streamlit-fqdn>/_stcore/health
# ok
```

## Streaming logs

```bash
# Real-time logs from Flask
az containerapp logs show -n telechurn-flask -g rg-telecom-churn --follow

# Or via the same Docker wrapper used in deploy.sh
docker run --rm -v "$HOME/.azure:/root/.azure" mcr.microsoft.com/azure-cli \
    az containerapp logs show -n telechurn-flask -g rg-telecom-churn --follow
```

## REST API reference

### `GET /health`
Liveness probe. Returns:
```json
{"status": "ok", "model_loaded": true}
```

### `GET /model-info`
```json
{
  "model_loaded": true,
  "threshold": 0.35,
  "metadata": {
    "version": "20260506_053659",
    "model_name": "lightgbm",
    "roc_auc": 0.9299,
    "f1": 0.6899,
    "num_cols": [...],
    "cat_cols": [...],
    "smote": true,
    "calibrated": true,
    "trained_at": "..."
  }
}
```

### `POST /predict`
Body: any subset of the trained-on feature columns. Missing columns are imputed in-pipeline.

```json
{
  "revenue": 50,
  "regularity": 15,
  "frequence": 10,
  "data_volume": 1000,
  "region": "DAKAR",
  "top_pack": "Pack 5",
  "montant": 100,
  "frequence_rech": 5
}
```

Returns:
```json
{
  "churn_probability": 0.18,
  "churn_prediction": 0,
  "risk_segment": "Low",
  "threshold": 0.35
}
```

### `POST /predict_batch`
Same as `/predict` but accepts a JSON array; returns an array of predictions in the same order.

### `GET /openapi.json`
The full OpenAPI 3.0 spec for the API.

### `GET /`
Swagger UI — interactive API explorer.

## Updating the deployed model

```bash
# 1. Re-train locally
python scripts/05_train_model.py

# 2. Re-upload to Blob
docker run --rm \
  -v "$HOME/.azure:/root/.azure" \
  -v "$PWD:/work" \
  -w /work \
  mcr.microsoft.com/azure-cli \
  az storage blob upload --account-name <storage-name> \
    --container-name models --name churn_model.joblib \
    --file models/churn_model.joblib --overwrite --auth-mode key

# 3. Restart the Flask app so it picks up the new model
az containerapp revision restart -n telechurn-flask -g rg-telecom-churn
```

(Or simply re-run `bash azure/deploy.sh` to rebuild + redeploy everything.)

## Roadmap — Layer 3: Airflow on Azure (not in this round)

Adds the full DAG orchestration:

| New resource | SKU | Cost |
|---|---|---|
| Azure Database for PostgreSQL Flexible Server | Burstable B1ms | ~$12/mo |
| Azure Files (premium share for DAGs + outputs) | 100 GB | ~$10/mo |
| Container App: airflow-scheduler | 0.5 CPU/1Gi | always-on |
| Container App: airflow-webserver | 0.5 CPU/1Gi | min 0 |

Estimated additional: **~$30-60/mo**.

If you want this, run `bash azure/deploy.sh --with-airflow` (TBD — currently disabled).

## Teardown

```bash
# Delete everything in one shot (irreversible)
docker run --rm -v "$HOME/.azure:/root/.azure" mcr.microsoft.com/azure-cli \
  az group delete -n rg-telecom-churn --yes --no-wait
```

This stops all billing.
