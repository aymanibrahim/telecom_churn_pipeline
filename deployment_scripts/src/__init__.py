"""Shared library for the Telecom Customer Churn pipeline.

Modules
-------
paths               : Resolves all input/output file paths from CHURN_BASE_DIR env var.
config              : Loads non-path constants (MCC, MNC, hyperparameters, threshold).
logging_setup       : Centralised logger configuration.
validation          : Lightweight Pandera-style schema validation.
idempotency         : ``skip_if_fresh`` decorator for incremental DAG runs.
data_io             : CSV / joblib helpers with directory creation.
sampling            : Stratified sampling for the Expresso 100k subset.
geo                 : OpenCellID Senegal filtering + GADM spatial joins.
network_kpis        : Region / department / arrondissement KPI aggregations.
feature_engineering : Cleaning + derived features used by training and inference.
eda                 : Headless EDA report generator (matplotlib Agg backend).
modeling            : LightGBM pipeline builder, training, calibration, threshold tuning.
inference           : Batch prediction with risk segmentation.
"""

__all__ = [
    "config",
    "data_io",
    "eda",
    "feature_engineering",
    "geo",
    "idempotency",
    "inference",
    "logging_setup",
    "modeling",
    "network_kpis",
    "paths",
    "sampling",
    "validation",
]
