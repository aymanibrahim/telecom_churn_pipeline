"""Non-path configuration constants for the Telecom Churn pipeline.

Loads ``config/config.yaml`` if present; otherwise falls back to the
hardcoded defaults below. Anything that is *not* a file path lives here:

* OpenCellID filter constants (Senegal MCC, Expresso MNC, observation window)
* Sampling parameters (stratified sample size, seed)
* Modeling hyperparameters (LightGBM grid, calibration method, threshold)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import paths

# ---------------------------------------------------------------------------
# Default configuration (used if config/config.yaml is missing)
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "opencellid": {
        "mcc": 608,             # Senegal mobile country code
        "mnc": 3,               # Expresso mobile network code
        "window_days": 90,
        "ref_window_start": "2019-04-01",
        "ref_window_end": "2019-07-01",
    },
    "sample": {
        "n": 100_000,
        "stratify_col": "CHURN",
        "seed": 42,
        "input_chunksize": 200_000,
        "input_estimated_rows": 2_000_000,
    },
    "model": {
        "target": "churn",
        "test_size": 0.15,
        "val_size": 0.15,         # of the remaining 85% → ≈ 15% overall
        "default_threshold": 0.42,
        "calibration_method": "isotonic",
        "smote_k_neighbors": 5,
        "lgbm_param_grid": {
            "clf__n_estimators":     [200, 400, 600],
            "clf__max_depth":        [5, 7, 10, -1],
            "clf__num_leaves":       [30, 60, 120],
            "clf__learning_rate":    [0.03, 0.05, 0.1],
            "clf__subsample":        [0.7, 0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 1.0],
        },
        "n_iter_search": 20,
        "cv_folds": 5,
        "shap_sample_size": 1_000,
    },
    "business": {
        "cost_per_campaign": 10.0,   # $
        "customer_lifetime_value": 200.0,  # $
        "risk_tiers": {
            "High":   [0.70, 1.01],
            "Medium": [0.40, 0.70],
            "Low":    [0.00, 0.40],
        },
    },
    "drop_columns": {
        "zero_variance": [
            "mrg",
            "region_avg_signal",
            "region_signal_strength_index",
            "department_signal_strength_index",
            "arr_signal_strength_index",
        ],
        "collinear": [
            "arr_coverage_index",
            "department_coverage_index",
            "department_network_quality_score",
        ],
        "sparse": ["zone1", "zone2"],
        "id": ["user_id"],
    },
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge ``overrides`` into a copy of ``base`` (nested dicts)."""
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(yaml_path: Path | None = None) -> dict[str, Any]:
    """Load YAML config (if present) and merge over the defaults.

    Parameters
    ----------
    yaml_path
        Optional override. Defaults to ``<BASE_DIR>/config/config.yaml``.
    """
    yaml_path = yaml_path or (paths.CONFIG_DIR / "config.yaml")
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        return _deep_merge(_DEFAULTS, user_cfg)
    return _DEFAULTS


# Convenience accessors (loaded once at import time).
CONFIG: dict[str, Any] = load_config()
SEED: int = CONFIG["sample"]["seed"]
TARGET: str = CONFIG["model"]["target"]
DEFAULT_THRESHOLD: float = CONFIG["model"]["default_threshold"]


if __name__ == "__main__":
    import json
    print(json.dumps(CONFIG, indent=2, default=str))
