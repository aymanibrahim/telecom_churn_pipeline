"""
config.py
---------
Central configuration for the Telecom AI-Enhanced Data Pipeline.
All file paths, schema definitions, model hyperparameters, and
notification settings are managed here so that individual scripts
remain environment-agnostic.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Base directory (project root)
# ─────────────────────────────────────────────
BASE_DIR = Path(os.environ.get("TELECOM_BASE_DIR", "/opt/airflow"))

# ─────────────────────────────────────────────
# Raw data paths
# ─────────────────────────────────────────────
RAW_DIR            = BASE_DIR / "data" / "raw"
PROCESSED_DIR      = BASE_DIR / "data" / "processed"
MODELS_DIR         = BASE_DIR / "models"
OUTPUTS_DIR        = BASE_DIR / "outputs"
LOGS_DIR           = BASE_DIR / "logs"

# Source files (must be placed in RAW_DIR before pipeline runs)
AFRICA_TOWERS_FILE = RAW_DIR / "Africa_towers.csv"
EXPRESSO_FILE      = RAW_DIR / "expresso.csv"
GADM_GPKG_FILE     = RAW_DIR / "gadm41_SEN.gpkg"

# Intermediate processed files
OPENCELLID_90D_FILE       = PROCESSED_DIR / "opencellid_senegal_90d.csv"
EXPRESSO_SAMPLE_100K_FILE = PROCESSED_DIR / "expresso_sample_100k.csv"
TELECOM_CHURN_FILE        = PROCESSED_DIR / "telecom_churn.csv"
TELECOM_CHURN_100K_FILE   = PROCESSED_DIR / "telecom_churn_100k.csv"

# Output files produced by the model stage
PREDICTIONS_FILE    = OUTPUTS_DIR / "churn_predictions.csv"
EVAL_METRICS_FILE   = OUTPUTS_DIR / "evaluation_metrics.csv"
MODEL_FILE_TEMPLATE = "lgb_churn_pipeline_{version}.joblib"   # versioned filename
LATEST_MODEL_SYMLINK = MODELS_DIR / "churn_model.joblib"       # Streamlit looks here

# GADM GeoPackage download URL (cached locally after first run)
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_SEN.gpkg"

# ─────────────────────────────────────────────
# OpenCellID / telecoms filter constants
# ─────────────────────────────────────────────
SENEGAL_MCC = 608          # Mobile Country Code for Senegal
EXPRESSO_MNC = 3           # Mobile Network Code for Expresso

# 90-day observation window for tower data (align with Expresso churn definition)
TOWER_WINDOW_START = "2019-04-01"
TOWER_WINDOW_END   = "2019-07-01"

# OpenCellID column dtypes (saves memory during Dask load)
OPENCELLID_DTYPES = {
    "radio":         "category",
    "MCC":           "int16",
    "MNC":           "int8",
    "TAC":           "int32",
    "CID":           "int64",
    "unit":          "int16",
    "LON":           "float32",
    "LAT":           "float32",
    "RANGE":         "float32",
    "SAM":           "int16",
    "changeable":    "int8",
    "created":       "int64",
    "updated":       "int64",
    "averageSignal": "float32",
    "Country":       "category",
    "Network":       "category",
    "Continent":     "category",
}

# ─────────────────────────────────────────────
# Expresso dataset constants
# ─────────────────────────────────────────────
EXPRESSO_DTYPES = {
    "user_id":       "object",
    "REGION":        "category",
    "TENURE":        "category",
    "MONTANT":       "float32",
    "FREQUENCE_RECH":"float32",
    "REVENUE":       "float32",
    "ARPU_SEGMENT":  "float32",
    "FREQUENCE":     "float32",
    "DATA_VOLUME":   "float32",
    "ON_NET":        "float32",
    "ORANGE":        "float32",
    "TIGO":          "float32",
    "ZONE1":         "float32",
    "ZONE2":         "float32",
    "MRG":           "category",
    "REGULARITY":    "int16",
    "TOP_PACK":      "category",
    "FREQ_TOP_PACK": "float32",
    "CHURN":         "int8",
}

EXPRESSO_CAT_COLS = ["REGION", "TENURE", "MRG", "TOP_PACK"]

# Canonical column order for telecom_churn integrated dataset
TELECOM_CHURN_FINAL_COLS = [
    "user_id", "region",
    "tenure", "montant", "frequence_rech", "revenue", "arpu_segment",
    "frequence", "data_volume", "on_net", "orange", "tigo",
    "zone1", "zone2", "mrg", "regularity", "top_pack", "freq_top_pack",
    "region_tower_count", "region_avg_range", "region_avg_samples", "region_avg_signal",
    "region_coverage_index", "region_signal_strength_index", "region_network_quality_score",
    "department_signal_strength_index", "department_network_quality_score",
    "department_coverage_index",
    "arr_signal_strength_index", "arr_network_quality_score", "arr_coverage_index",
    "churn",
]

# ─────────────────────────────────────────────
# Sampling constants
# ─────────────────────────────────────────────
EXPRESSO_APPROX_ROWS  = 2_000_000
SAMPLE_CHUNKSIZE      = 200_000
TARGET_SAMPLE_SIZE    = 100_000
RANDOM_STATE          = 42

# ─────────────────────────────────────────────
# GADM layer names and column mappings
# ─────────────────────────────────────────────
GADM_LAYERS = {
    "ADM1": "ADM_ADM_1",
    "ADM2": "ADM_ADM_2",
    "ADM3": "ADM_ADM_3",
}
REGION_COL         = "NAME_1"
DEPARTMENT_COL     = "NAME_2"
ARRONDISSEMENT_COL = "NAME_3"

# ─────────────────────────────────────────────
# ML model settings
# ─────────────────────────────────────────────
SEED = 42

# Columns to drop before modelling (from EDA findings)
ZERO_VAR_COLS = [
    "mrg",
    "region_avg_signal",
    "region_signal_strength_index",
    "department_signal_strength_index",
    "arr_signal_strength_index",
]
COLLINEAR_COLS = [
    "arr_coverage_index",
    "department_coverage_index",
    "department_network_quality_score",
]
SPARSE_COLS = ["zone1", "zone2"]
ID_COLS     = ["user_id"]

TARGET_COL   = "churn"
LOYAL_VALUE  = "K > 24 month"   # Tenure binarisation sentinel

# Log-transform candidates
LOG_TRANSFORM_COLS = [
    "montant", "revenue", "data_volume", "on_net", "orange", "tigo",
    "freq_top_pack", "avg_recharge_amount", "avg_revenue_per_tx",
]

# Engagement score weights
ENGAGEMENT_WEIGHTS = {"regularity": 0.5, "revenue": 0.3, "frequence": 0.2}

# Hyperparameter grid for LightGBM RandomizedSearchCV
LGB_PARAM_GRID = {
    "clf__n_estimators":     [200, 400, 600],
    "clf__max_depth":        [5, 7, 10, -1],
    "clf__num_leaves":       [30, 60, 120],
    "clf__learning_rate":    [0.03, 0.05, 0.1],
    "clf__subsample":        [0.7, 0.8, 1.0],
    "clf__colsample_bytree": [0.7, 0.8, 1.0],
}

# Train / Val / Test split ratios
TEST_SIZE = 0.15
VAL_SIZE  = 0.15 / 0.85          # fraction of temp set

# Risk tier thresholds (churn probability)
RISK_TIERS = {
    "High":   (0.70, 1.01),
    "Medium": (0.40, 0.70),
    "Low":    (0.00, 0.40),
}

# Business KPI 
COST_PER_CAMPAIGN = 10    # USD per retention outreach
CLV               = 200   # USD customer lifetime value per successful retention

# ─────────────────────────────────────────────
# Notification settings
# ─────────────────────────────────────────────
NOTIFICATION_EMAIL_TO      = os.environ.get("NOTIFY_EMAIL_TO",    "team@example.com")
NOTIFICATION_EMAIL_FROM    = os.environ.get("NOTIFY_EMAIL_FROM",  "airflow@example.com")
NOTIFICATION_SMTP_HOST     = os.environ.get("SMTP_HOST",          "smtp.example.com")
NOTIFICATION_SMTP_PORT     = int(os.environ.get("SMTP_PORT",      "587"))
NOTIFICATION_SMTP_USER     = os.environ.get("SMTP_USER",          "")
NOTIFICATION_SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD",      "")

# ─────────────────────────────────────────────
# Airflow DAG settings
# ─────────────────────────────────────────────
DAG_OWNER        = "data_team"
DAG_EMAIL        = [NOTIFICATION_EMAIL_TO]
DAG_START_DATE_STR = "2024-01-01"
DAG_SCHEDULE     = "@daily"               # or a cron expression
DAG_RETRIES      = 2
DAG_RETRY_DELAY_MINUTES = 5
