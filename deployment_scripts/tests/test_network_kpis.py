"""Tests for src.network_kpis."""
from __future__ import annotations

import pandas as pd

from src import network_kpis


def test_compute_kpis_basic(tiny_cells: pd.DataFrame) -> None:
    df = tiny_cells.copy()
    df["REGION"] = ["DAKAR"] * 15 + ["THIES"] * 15

    out = network_kpis.compute_kpis(df, ["REGION"], "region")

    expected = {
        "REGION",
        "region_tower_count",
        "region_avg_range",
        "region_avg_samples",
        "region_avg_signal",
        "region_coverage_index",
        "region_signal_strength_index",
        "region_network_quality_score",
    }
    assert set(out.columns) == expected
    assert len(out) == 2
    assert (out["region_tower_count"] == 15).all()
    # coverage = tower_count * avg_range  → must be > 0
    assert (out["region_coverage_index"] > 0).all()


def test_region_kpis_helper(tiny_cells: pd.DataFrame) -> None:
    df = tiny_cells.copy()
    df["REGION"] = ["DAKAR"] * 30
    out = network_kpis.compute_region_kpis(df)
    assert "region_network_quality_score" in out.columns
    assert len(out) == 1
