"""Network KPI aggregation by administrative level.

Source notebook: notebooks/build_telecom_churn_dataset.ipynb (KPI section).

For each admin polygon we compute:

* ``tower_count``           : number of cells inside the polygon
* ``avg_range``             : mean reported RANGE (m)
* ``avg_samples``           : mean SAM (sample count per cell)
* ``avg_signal``            : mean ``averageSignal``
* ``coverage_index``        : tower_count x avg_range
* ``signal_strength_index`` : avg_signal / avg_samples (avoids /0)
* ``network_quality_score`` : 0.4·signal_strength + 0.3·tower_count + 0.3·avg_samples
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def compute_kpis(
    cells: pd.DataFrame,
    group_cols: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    """Aggregate cell-tower metrics into a KPI table for ``group_cols``.

    The cells dataframe must contain columns: ``CID``, ``RANGE``, ``SAM``,
    ``averageSignal`` and the columns named in ``group_cols``.
    """
    agg = cells.groupby(list(group_cols)).agg(
        **{f"{prefix}_tower_count": ("CID", "count")},
        **{f"{prefix}_avg_range":   ("RANGE", "mean")},
        **{f"{prefix}_avg_samples": ("SAM", "mean")},
        **{f"{prefix}_avg_signal":  ("averageSignal", "mean")},
    ).reset_index()

    tower_count = agg[f"{prefix}_tower_count"]
    avg_range = agg[f"{prefix}_avg_range"]
    avg_samples = agg[f"{prefix}_avg_samples"].replace(0, np.nan)
    avg_signal = agg[f"{prefix}_avg_signal"]

    agg[f"{prefix}_coverage_index"] = tower_count * avg_range
    agg[f"{prefix}_signal_strength_index"] = avg_signal / avg_samples
    agg[f"{prefix}_network_quality_score"] = (
        0.4 * agg[f"{prefix}_signal_strength_index"]
        + 0.3 * tower_count
        + 0.3 * avg_samples
    )
    return agg


def compute_region_kpis(cells_with_region: pd.DataFrame) -> pd.DataFrame:
    """Region-level (ADM1) KPIs. Expects a 'REGION' column on ``cells_with_region``."""
    return compute_kpis(cells_with_region, ["REGION"], "region")


def compute_department_kpis_per_region(
    cells_with_region_dept: pd.DataFrame,
) -> pd.DataFrame:
    """Department-level (ADM2) KPIs collapsed to one row per REGION.

    The Expresso table has REGION (ADM1) only — to attach department-level
    insights we aggregate them up by summing coverage and averaging the
    quality / signal indices.
    """
    dept = compute_kpis(
        cells_with_region_dept,
        ["REGION", "department"],
        "department",
    )
    return (
        dept.groupby("REGION", observed=True)
        .agg(
            department_coverage_index=(
                "department_coverage_index", "sum"),
            department_signal_strength_index=(
                "department_signal_strength_index", "mean"),
            department_network_quality_score=(
                "department_network_quality_score", "mean"),
        )
        .reset_index()
    )


def compute_arrondissement_kpis_per_region(
    cells_with_region_arr: pd.DataFrame,
) -> pd.DataFrame:
    """Arrondissement-level (ADM3) KPIs collapsed to one row per REGION."""
    arr = compute_kpis(
        cells_with_region_arr,
        ["REGION", "arrondissement"],
        "arrondissement",
    )
    return (
        arr.groupby("REGION", observed=True)
        .agg(
            arr_coverage_index=(
                "arrondissement_coverage_index", "sum"),
            arr_signal_strength_index=(
                "arrondissement_signal_strength_index", "mean"),
            arr_network_quality_score=(
                "arrondissement_network_quality_score", "mean"),
        )
        .reset_index()
    )
