"""Shared pytest fixtures for the Telecom Churn pipeline tests.

Adds the project root to ``sys.path`` so ``from src.X import Y`` works
when tests run from any directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def project_root() -> Path:
    return _ROOT


@pytest.fixture
def tiny_telecom_churn() -> pd.DataFrame:
    """A 200-row DataFrame mirroring the schema of telecom_churn_100k.csv."""
    rng = np.random.default_rng(42)
    n = 200
    regions = ["DAKAR", "THIES", "SAINT-LOUIS", "DIOURBEL"]
    return pd.DataFrame({
        "user_id":                   [f"u{i:05d}" for i in range(n)],
        "region":                    rng.choice(regions, size=n),
        "tenure":                    rng.choice(["I 18-21 month", "K > 24 month"], size=n),
        "montant":                   rng.uniform(0, 50_000, size=n),
        "frequence_rech":            rng.uniform(0, 30, size=n),
        "revenue":                   rng.uniform(0, 50_000, size=n),
        "arpu_segment":              rng.uniform(0, 17_000, size=n),
        "frequence":                 rng.uniform(0, 100, size=n),
        "data_volume":               rng.uniform(0, 10_000, size=n),
        "on_net":                    rng.uniform(0, 5_000, size=n),
        "orange":                    rng.uniform(0, 1_000, size=n),
        "tigo":                      rng.uniform(0, 1_000, size=n),
        "zone1":                     rng.uniform(0, 100, size=n),
        "zone2":                     rng.uniform(0, 100, size=n),
        "mrg":                       ["NO"] * n,
        "regularity":                rng.integers(0, 62, size=n),
        "top_pack":                  rng.choice(["A", "B", "C", None], size=n),
        "freq_top_pack":             rng.uniform(0, 50, size=n),
        "region_tower_count":        rng.integers(1, 50, size=n),
        "region_avg_range":          rng.uniform(100, 5_000, size=n),
        "region_avg_samples":        rng.uniform(1, 100, size=n),
        "region_avg_signal":         rng.uniform(-110, -50, size=n),
        "region_coverage_index":     rng.uniform(100, 100_000, size=n),
        "region_signal_strength_index":   rng.uniform(-10, 10, size=n),
        "region_network_quality_score":   rng.uniform(0, 100, size=n),
        "department_signal_strength_index": rng.uniform(-10, 10, size=n),
        "department_network_quality_score": rng.uniform(0, 100, size=n),
        "department_coverage_index": rng.uniform(100, 100_000, size=n),
        "arr_signal_strength_index": rng.uniform(-10, 10, size=n),
        "arr_network_quality_score": rng.uniform(0, 100, size=n),
        "arr_coverage_index":        rng.uniform(100, 100_000, size=n),
        "churn":                     rng.integers(0, 2, size=n),
    })


@pytest.fixture
def tiny_cells() -> pd.DataFrame:
    """30-row OpenCellID-shaped fixture, all in Senegal/Expresso."""
    rng = np.random.default_rng(7)
    n = 30
    return pd.DataFrame({
        "radio":         ["GSM"] * n,
        "MCC":           [608] * n,
        "MNC":           [3] * n,
        "TAC":           rng.integers(1, 99999, size=n),
        "CID":           rng.integers(1, 99999, size=n),
        "unit":          [0] * n,
        "LON":           rng.uniform(-17.5, -11.4, size=n),  # Senegal bbox
        "LAT":           rng.uniform(12.3, 16.7, size=n),
        "RANGE":         rng.uniform(500, 5000, size=n),
        "SAM":           rng.integers(1, 30, size=n),
        "changeable":    [1] * n,
        "created":       [1551398400] * n,   # 2019-03-01
        "updated":       [1561939200] * n,   # 2019-07-01
        "averageSignal": rng.uniform(-110, -50, size=n),
        "Country":       ["Senegal"] * n,
        "Network":       ["Expresso"] * n,
        "Continent":     ["Africa"] * n,
    })
