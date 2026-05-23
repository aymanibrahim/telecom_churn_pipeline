"""Tests for src.inference helpers (the predict() function needs a model fixture
so it's covered by the end-to-end test instead)."""
from __future__ import annotations

import pytest

from src import inference


@pytest.mark.parametrize(
    "prob, expected",
    [
        (0.05, "Low"),
        (0.39, "Low"),
        (0.40, "Medium"),
        (0.65, "Medium"),
        (0.70, "High"),
        (0.99, "High"),
    ],
)
def test_classify_risk_buckets(prob: float, expected: str) -> None:
    assert inference.classify_risk(prob) == expected
