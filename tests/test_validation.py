"""Schema validation tests."""
from __future__ import annotations

import pandas as pd
import pytest

from src import validation


def test_passes_when_schema_matches(tiny_telecom_churn: pd.DataFrame) -> None:
    schema = validation.Schema(
        name="tiny",
        min_rows=100,
        columns=[
            validation.Column("user_id",  "str", nullable=False),
            validation.Column("revenue",  "number", min=0),
            validation.Column("churn",    "number", min=0, max=1, nullable=False),
        ],
    )
    validation.validate_dataframe(tiny_telecom_churn, schema)


def test_fails_on_missing_required_column(tiny_telecom_churn: pd.DataFrame) -> None:
    schema = validation.Schema(
        name="tiny",
        columns=[validation.Column("does_not_exist", "number")],
    )
    with pytest.raises(validation.SchemaError, match="missing required column"):
        validation.validate_dataframe(tiny_telecom_churn, schema)


def test_fails_on_value_out_of_range(tiny_telecom_churn: pd.DataFrame) -> None:
    schema = validation.Schema(
        name="bad-range",
        columns=[validation.Column("regularity", "number", min=0, max=10)],
    )
    with pytest.raises(validation.SchemaError, match="values > max"):
        validation.validate_dataframe(tiny_telecom_churn, schema)


def test_fails_on_too_few_rows(tiny_telecom_churn: pd.DataFrame) -> None:
    schema = validation.Schema(
        name="big",
        min_rows=999_999,
        columns=[validation.Column("churn", "number")],
    )
    with pytest.raises(validation.SchemaError, match="row count"):
        validation.validate_dataframe(tiny_telecom_churn, schema)


def test_fails_on_disallowed_value(tiny_telecom_churn: pd.DataFrame) -> None:
    df = tiny_telecom_churn.copy()
    df["churn"] = 99   # invalid value
    schema = validation.Schema(
        name="enum",
        columns=[validation.Column("churn", "number", in_set={0, 1})],
    )
    with pytest.raises(validation.SchemaError, match="outside allowed set"):
        validation.validate_dataframe(df, schema)


def test_telecom_churn_schema_accepts_fixture(tiny_telecom_churn: pd.DataFrame) -> None:
    validation.validate_dataframe(tiny_telecom_churn, validation.TELECOM_CHURN)
