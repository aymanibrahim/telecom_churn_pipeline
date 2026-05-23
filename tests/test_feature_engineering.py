"""Tests for src.feature_engineering."""
from __future__ import annotations

import pandas as pd

from src import feature_engineering as fe


def test_drop_irrelevant_removes_zero_var_and_id(tiny_telecom_churn: pd.DataFrame) -> None:
    out = fe.drop_irrelevant_columns(tiny_telecom_churn)
    for col in ("user_id", "mrg", "zone1", "zone2"):
        assert col not in out.columns, col
    assert len(out) == len(tiny_telecom_churn)


def test_impute_fills_known_columns(tiny_telecom_churn: pd.DataFrame) -> None:
    df = tiny_telecom_churn.copy()
    df.loc[:5, "montant"] = None
    df.loc[:5, "data_volume"] = None
    out = fe.impute_basic(df)
    assert out["montant"].isna().sum() == 0
    assert out["data_volume"].isna().sum() == 0


def test_add_derived_features_adds_expected_columns(
    tiny_telecom_churn: pd.DataFrame,
) -> None:
    df = fe.add_derived_features(tiny_telecom_churn)
    for col in (
        "is_data_user", "no_data_flag", "avg_recharge_amount",
        "avg_revenue_per_tx", "engagement_score", "is_loyal",
        "log_revenue", "log_data_volume",
    ):
        assert col in df.columns, f"missing {col}"
    # tenure → is_loyal
    assert "tenure" not in df.columns


def test_prepare_features_full_pipeline(tiny_telecom_churn: pd.DataFrame) -> None:
    out = fe.prepare_features(tiny_telecom_churn)
    assert "churn" in out.columns
    assert "user_id" not in out.columns
    num_cols, cat_cols = fe.split_feature_columns(out)
    assert "churn" not in num_cols
    assert "churn" not in cat_cols
    assert len(num_cols) > 0
    # 'region' and 'top_pack' are the only declared categoricals.
    assert set(cat_cols).issubset({"region", "top_pack"})
