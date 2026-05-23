"""Lightweight schema + data quality validation between pipeline stages.

We don't pull in Pandera to keep the dependency surface small. Instead
each stage's expected schema is declared as a :class:`Schema` (column
name → dtype family + nullable + optional value bounds) and validated
with one helper.

Usage
-----
::

    from src import validation, schemas
    validation.validate_dataframe(df, schemas.OPENCELLID_SENEGAL)

Failures raise :class:`SchemaError`. Soft warnings (unexpected extra
columns, missing-rate above threshold) go through the logger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
DTYPE_FAMILIES: dict[str, tuple[type, ...]] = {
    "int":      (np.integer,),
    "float":    (np.floating,),
    "number":   (np.integer, np.floating),
    "str":      (np.object_, pd.StringDtype, str),
    "category": (pd.CategoricalDtype,),
    "bool":     (np.bool_,),
}


@dataclass
class Column:
    name: str
    dtype: str = "number"           # one of DTYPE_FAMILIES keys
    nullable: bool = True
    min: float | None = None
    max: float | None = None
    in_set: set[Any] | None = None
    required: bool = True


@dataclass
class Schema:
    name: str
    columns: list[Column]
    min_rows: int = 1
    allow_extra_columns: bool = True
    max_missing_rate: float = 0.5    # warn-level only, per column


class SchemaError(ValueError):
    """Raised when a DataFrame violates its declared schema."""


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
def _is_dtype(series: pd.Series, family: str) -> bool:
    if family not in DTYPE_FAMILIES:
        return True  # unknown family — skip type check
    if family == "str":
        return series.dtype == object or pd.api.types.is_string_dtype(series)
    if family == "category":
        return isinstance(series.dtype, pd.CategoricalDtype)
    if family == "bool":
        return pd.api.types.is_bool_dtype(series)
    if family == "int":
        return pd.api.types.is_integer_dtype(series)
    if family == "float":
        return pd.api.types.is_float_dtype(series)
    if family == "number":
        return pd.api.types.is_numeric_dtype(series)
    return True


def validate_dataframe(df: pd.DataFrame, schema: Schema) -> None:
    """Validate ``df`` against ``schema``. Raises :class:`SchemaError` on hard failure."""
    errors: list[str] = []

    if len(df) < schema.min_rows:
        errors.append(
            f"row count {len(df)} < min_rows {schema.min_rows}"
        )

    declared = {c.name for c in schema.columns}
    actual = set(df.columns)

    # Missing required columns
    for col in schema.columns:
        if col.required and col.name not in actual:
            errors.append(f"missing required column: {col.name!r}")

    # Extra columns (warning only by default)
    extras = actual - declared
    if extras and not schema.allow_extra_columns:
        errors.append(f"unexpected columns: {sorted(extras)}")
    elif extras:
        log.debug("[%s] extra columns (allowed): %s", schema.name, sorted(extras))

    # Per-column checks
    for col in schema.columns:
        if col.name not in actual:
            continue
        s = df[col.name]

        if not _is_dtype(s, col.dtype):
            errors.append(
                f"{col.name!r}: dtype {s.dtype} is not in family {col.dtype!r}"
            )

        if not col.nullable and s.isna().any():
            errors.append(
                f"{col.name!r}: nulls found ({s.isna().sum():,}) but nullable=False"
            )

        if col.min is not None and pd.api.types.is_numeric_dtype(s):
            below = (s < col.min).sum()
            if below:
                errors.append(f"{col.name!r}: {below:,} values < min={col.min}")
        if col.max is not None and pd.api.types.is_numeric_dtype(s):
            above = (s > col.max).sum()
            if above:
                errors.append(f"{col.name!r}: {above:,} values > max={col.max}")

        if col.in_set is not None:
            bad = ~s.dropna().isin(col.in_set)
            if bad.any():
                errors.append(
                    f"{col.name!r}: {bad.sum():,} values outside allowed set"
                )

        # Soft warning: high missingness
        miss_rate = float(s.isna().mean())
        if miss_rate > schema.max_missing_rate:
            log.warning(
                "[%s] %s: missing rate %.1f%% exceeds %.1f%% threshold",
                schema.name, col.name, miss_rate * 100, schema.max_missing_rate * 100,
            )

    if errors:
        msg = f"Schema validation failed for '{schema.name}':\n  - " + "\n  - ".join(errors)
        log.error(msg)
        raise SchemaError(msg)

    log.info("[%s] OK - %s rows x %s cols", schema.name, len(df), len(df.columns))


def validate_csv(path: Any, schema: Schema, **read_kwargs: Any) -> pd.DataFrame:
    """Read ``path`` and validate. Returns the DataFrame."""
    df = pd.read_csv(path, **read_kwargs)
    validate_dataframe(df, schema)
    return df


# ---------------------------------------------------------------------------
# Concrete schemas for the pipeline outputs
# ---------------------------------------------------------------------------
EXPRESSO_SAMPLE = Schema(
    name="expresso_sample_100k",
    min_rows=10_000,
    columns=[
        Column("user_id",         "str",    nullable=False),
        Column("REGION",          "str"),
        Column("TENURE",          "str"),
        Column("MONTANT",         "number", min=0),
        Column("FREQUENCE_RECH",  "number", min=0),
        Column("REVENUE",         "number", min=0),
        Column("ARPU_SEGMENT",    "number"),
        Column("FREQUENCE",       "number", min=0),
        Column("DATA_VOLUME",     "number", min=0),
        Column("REGULARITY",      "number", min=0, max=62),
        Column("CHURN",           "number", min=0, max=1, nullable=False),
    ],
)

OPENCELLID_SENEGAL = Schema(
    name="opencellid_senegal_90d",
    min_rows=100,
    columns=[
        Column("MCC",   "number", in_set={608}, nullable=False),
        Column("MNC",   "number", in_set={3}, nullable=False),
        Column("LON",   "number", min=-20, max=-10, nullable=False),
        Column("LAT",   "number", min=10, max=20, nullable=False),
        Column("RANGE", "number", min=0, nullable=True),
        Column("SAM",   "number", min=0, nullable=True),
        Column("CID",   "number", nullable=False),
    ],
)

TELECOM_CHURN = Schema(
    name="telecom_churn",
    min_rows=100,
    columns=[
        Column("region",                          "str"),
        Column("revenue",                         "number", min=0),
        Column("regularity",                      "number", min=0, max=62),
        Column("churn",                           "number", min=0, max=1,
               nullable=False),
        Column("region_tower_count",              "number", required=False),
        Column("region_network_quality_score",    "number", required=False),
    ],
)

CHURN_PREDICTIONS = Schema(
    name="churn_predictions",
    min_rows=1,
    columns=[
        Column("churn_probability", "number", min=0, max=1, nullable=False),
        Column("churn_prediction",  "number", min=0, max=1, nullable=False),
        Column("risk_segment",      "str",    nullable=False,
               in_set={"Low", "Medium", "High"}),
    ],
)
