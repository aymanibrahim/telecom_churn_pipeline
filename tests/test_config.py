"""Sanity-check the YAML config loader and merged defaults."""
from __future__ import annotations

from pathlib import Path

from src import config


def test_config_has_expected_top_level_keys() -> None:
    cfg = config.load_config()
    for key in ("opencellid", "sample", "model", "business", "drop_columns"):
        assert key in cfg, f"missing key: {key}"


def test_opencellid_constants() -> None:
    cfg = config.load_config()
    assert cfg["opencellid"]["mcc"] == 608, "Senegal MCC must be 608"
    assert cfg["opencellid"]["mnc"] == 3, "Expresso MNC must be 3"
    assert cfg["opencellid"]["window_days"] == 90


def test_default_threshold_is_in_range() -> None:
    assert 0.0 < config.DEFAULT_THRESHOLD < 1.0


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    yml = tmp_path / "config.yaml"
    yml.write_text("model:\n  default_threshold: 0.6\n", encoding="utf-8")
    cfg = config.load_config(yml)
    assert cfg["model"]["default_threshold"] == 0.6
    # Untouched defaults still present
    assert cfg["opencellid"]["mcc"] == 608
