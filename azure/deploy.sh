#!/usr/bin/env bash
# =============================================================================
# Telecom Customer Churn — one-shot Azure deployer.
#
# Usage:   bash azure/deploy.sh [RESOURCE_GROUP] [LOCATION]
# Default: RESOURCE_GROUP=rg-telecom-churn  LOCATION=eastus
#
# Requires `az` (we call it via docker for portability) and `docker buildx`.
# Idempotent — safe to re-run.
# =============================================================================
set -euo pipefail

RG="${1:-rg-telecom-churn}"
LOC="${2:-eastus}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Use the local az binary if installed, else docker.
if command -v az >/dev/null 2>&1; then
  AZ=(az)
else
  AZ=(docker run --rm
      -v "${HOME:-$USERPROFILE}/.azure:/root/.azure"
      -v "${PROJECT_DIR}:/work"
      -w /work
      mcr.microsoft.com/azure-cli az)
fi

echo "==> Azure CLI: ${AZ[*]}"
"${AZ[@]}" account show --query name -o tsv

# -----------------------------------------------------------------------------
# Step 1 — Resource group
# -----------------------------------------------------------------------------
echo "==> Resource group: $RG ($LOC)"
"${AZ[@]}" group create -n "$RG" -l "$LOC" -o none

# -----------------------------------------------------------------------------
# Step 2 — Bootstrap deployment (ACR + Storage + Env, no apps yet)
# -----------------------------------------------------------------------------
echo "==> Bootstrap infra (no apps yet)…"
"${AZ[@]}" deployment group create \
  -g "$RG" \
  -n bootstrap-$(date +%s) \
  --template-file azure/main.bicep \
  --parameters azure/parameters.json \
  --parameters flaskImage='' streamlitImage='' \
  -o none

ACR_NAME=$("${AZ[@]}" acr list -g "$RG" --query "[0].name" -o tsv | tr -d '\r')
ACR_LOGIN=$("${AZ[@]}" acr show -n "$ACR_NAME" --query loginServer -o tsv | tr -d '\r')
STORAGE_NAME=$("${AZ[@]}" storage account list -g "$RG" --query "[0].name" -o tsv | tr -d '\r')
echo "==> ACR:     $ACR_LOGIN"
echo "==> Storage: $STORAGE_NAME"

# -----------------------------------------------------------------------------
# Step 3 — Upload the churn_model.joblib to Blob
# -----------------------------------------------------------------------------
if [[ -f models/churn_model.joblib ]]; then
  echo "==> Uploading churn_model.joblib to blob…"
  "${AZ[@]}" storage blob upload \
    --account-name "$STORAGE_NAME" \
    --container-name models \
    --name churn_model.joblib \
    --file models/churn_model.joblib \
    --overwrite \
    --auth-mode key \
    -o none
fi

# -----------------------------------------------------------------------------
# Step 4 — Build images on ACR (no local docker push required)
# -----------------------------------------------------------------------------
echo "==> Building flask image on ACR…"
"${AZ[@]}" acr build -r "$ACR_NAME" -t flask:latest -f serving/Dockerfile.flask . -o none

echo "==> Building streamlit image on ACR…"
"${AZ[@]}" acr build -r "$ACR_NAME" -t streamlit:latest -f serving/Dockerfile.streamlit . -o none

# -----------------------------------------------------------------------------
# Step 5 — Re-deploy Bicep with the image tags so Container Apps come up
# -----------------------------------------------------------------------------
echo "==> Deploying Container Apps…"
"${AZ[@]}" deployment group create \
  -g "$RG" \
  -n apps-$(date +%s) \
  --template-file azure/main.bicep \
  --parameters azure/parameters.json \
  --parameters flaskImage="$ACR_LOGIN/flask:latest" streamlitImage="$ACR_LOGIN/streamlit:latest" \
  -o none

# -----------------------------------------------------------------------------
# Step 6 — Print FQDNs
# -----------------------------------------------------------------------------
FLASK_FQDN=$("${AZ[@]}" containerapp show -n telechurn-flask -g "$RG" --query "properties.configuration.ingress.fqdn" -o tsv | tr -d '\r')
ST_FQDN=$("${AZ[@]}" containerapp show -n telechurn-streamlit -g "$RG" --query "properties.configuration.ingress.fqdn" -o tsv | tr -d '\r')

cat <<EOF

============================================================
Deployment complete.

  Flask REST API:        https://${FLASK_FQDN}/
  Flask health check:    https://${FLASK_FQDN}/health
  Streamlit dashboard:   https://${ST_FQDN}/

Teardown:
  az group delete -n ${RG} --yes --no-wait
============================================================
EOF
