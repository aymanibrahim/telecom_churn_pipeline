"""Feature engineering shared between training and inference.

This module replaces the inline feature_engineering_task that lives in
the old ``dags/churn_pipeline.py``. The same transformations must run
during training **and** during batch inference, otherwise the model
would see a different schema in production. Keep this file the single
source of truth for the feature space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants from the EDA notebook
# ---------------------------------------------------------------------------
LOG_COLS: list[str] = [
    "montant", "revenue", "data_volume", "on_net", "orange", "tigo",
    "freq_top_pack", "avg_recharge_amount", "avg_revenue_per_tx",
]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove zero-variance, sparse, collinear and ID columns.

    Lists come from ``config.CONFIG['drop_columns']`` so they can be
    overridden via ``config/config.yaml``.
    """
    drop_cfg = config.CONFIG["drop_columns"]
    cols_to_drop: list[str] = (
        drop_cfg["zero_variance"]
        + drop_cfg["collinear"]
        + drop_cfg["sparse"]
        + drop_cfg["id"]
    )
    present = [c for c in cols_to_drop if c in df.columns]
    if present:
        log.info("Dropping %s columns: %s", len(present), present)
    return df.drop(columns=present)


def impute_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute numerical columns and 'Unknown'-impute key categoricals."""
    df = df.copy()
    for col in ("montant", "revenue"):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if "data_volume" in df.columns:
        df["data_volume"] = df["data_volume"].fillna(0)
    if "top_pack" in df.columns:
        df["top_pack"] = df["top_pack"].astype("object").fillna("Unknown")
    return df


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features used by the LightGBM pipeline."""
    df = df.copy()

    # Boolean / count flags
    if "data_volume" in df.columns:
        df["is_data_user"] = (df["data_volume"].fillna(0) > 0).astype(int)
        df["no_data_flag"] = df["data_volume"].isna().astype(int)

    # Per-transaction averages (avoid div-by-0).
    if {"montant", "frequence_rech"}.issubset(df.columns):
        df["avg_recharge_amount"] = df["montant"] / df["frequence_rech"].replace(0, np.nan)
    if {"revenue", "frequence"}.issubset(df.columns):
        df["avg_revenue_per_tx"] = df["revenue"] / df["frequence"].replace(0, np.nan)

    # Region-vs-arrondissement quality delta.
    if {"region_network_quality_score", "arr_network_quality_score"}.issubset(df.columns):
        df["network_quality_delta"] = (
            df["region_network_quality_score"] - df["arr_network_quality_score"]
        )

    # Log transforms on heavy-tailed columns.
    for col in LOG_COLS:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].fillna(0).clip(lower=0))

    # Engagement composite.
    weights = {"regularity": 0.5, "revenue": 0.3, "frequence": 0.2}
    score = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for col, w in weights.items():
        if col in df.columns:
            x = df[col].astype(float)
            denom = (x.max() - x.min()) + 1e-9
            norm = (x - x.min()) / denom
            score = score + w * norm.fillna(0.0)
            total_w += w
    if total_w > 0:
        df["engagement_score"] = score / total_w

    # Tenure → loyal flag (drop the original).
    if "tenure" in df.columns:
        df["is_loyal"] = (df["tenure"].astype(str) == "K > 24 month").astype(int)
        df = df.drop(columns=["tenure"])

    return df


# ---------------------------------------------------------------------------
# End-to-end transform
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full transform: drop → impute → derive."""
    return add_derived_features(impute_basic(drop_irrelevant_columns(df)))


def split_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numerical, categorical) feature column lists.

    The target (``churn``) is excluded from both. ``user_id`` is excluded
    too in case it's still around.
    """
    target = config.TARGET
    cat_cols = [c for c in ("region", "top_pack") if c in df.columns]
    num_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c != target and c not in cat_cols
    ]
    return num_cols, cat_cols
