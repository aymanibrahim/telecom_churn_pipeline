"""Headless EDA report generator.

Source notebook: notebooks/Telecom Customer Churn - Exploratory Data
Analysis (EDA) - Project Team A.ipynb

Designed to run inside an Airflow worker (no display server). Uses the
matplotlib Agg backend, writes PNGs to ``output_dir``, and emits a
machine-readable summary as ``eda_summary.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .logging_setup import get_logger

log = get_logger(__name__)
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.name)
    return path


def run_eda(input_path: Path, output_dir: Path) -> Path:
    """Run EDA and write plots + JSON summary into ``output_dir``."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    target = config.TARGET
    log.info("%s: shape=%s churn_rate=%.1f%%",
             input_path.name, df.shape, df[target].mean() * 100)

    # ---------------------------------------------------------- Quality
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct":   df.isna().mean() * 100.0,
    }).sort_values("missing_pct", ascending=False)
    missing_pos = missing[missing["missing_count"] > 0]

    duplicates = int(df.duplicated().sum())
    zero_var_obj = [c for c in df.select_dtypes("object")
                    if df[c].nunique(dropna=False) == 1]

    # Missing-value heatmap — per-row imshow allocates an N×K bitmap, which
    # OOM-blows on the 2 M dataset. Fall back to a per-column bar chart when
    # the input is large.
    HEATMAP_ROW_LIMIT = 200_000
    if not missing_pos.empty:
        if len(df) <= HEATMAP_ROW_LIMIT:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.isna(), cbar=False, yticklabels=False, ax=ax)
            ax.set_title("Missing-value pattern (per-row)")
            _save(fig, output_dir / "missing_pattern.png")
        else:
            log.info("Skipping per-row missing heatmap (n=%s > %s); "
                     "writing missing_pct_bar.png instead",
                     f"{len(df):,}", f"{HEATMAP_ROW_LIMIT:,}")
            fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(missing_pos))))
            missing_pos["missing_pct"].iloc[::-1].plot(
                kind="barh", ax=ax, color="C3", alpha=0.7)
            ax.set_xlabel("Missing rate (%)")
            ax.set_title(f"Missing-value percentage by column (n={len(df):,})")
            _save(fig, output_dir / "missing_pattern.png")

    # ---------------------------------------------------------- Target
    fig, ax = plt.subplots(figsize=(6, 4))
    df[target].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title(f"Target distribution ({target})")
    ax.set_xlabel(target)
    ax.set_ylabel("count")
    _save(fig, output_dir / "target_distribution.png")

    # ---------------------------------------------------------- Numerical distributions
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]

    if num_cols:
        n = len(num_cols)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, num_cols, strict=False):
            df[col].dropna().plot(kind="hist", bins=40, ax=ax)
            ax.set_title(col, fontsize=9)
        for ax in axes[len(num_cols):]:
            ax.set_visible(False)
        fig.suptitle("Numerical feature distributions", y=1.01)
        fig.tight_layout()
        _save(fig, output_dir / "numerical_distributions.png")

    # ---------------------------------------------------------- Correlation
    corr = df[[*num_cols, target]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Correlation matrix")
    _save(fig, output_dir / "correlation_matrix.png")

    churn_corr = (
        corr[target].drop(target)
        .reindex(corr[target].drop(target).abs().sort_values(ascending=False).index)
    )
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(churn_corr))))
    churn_corr.plot(kind="barh", ax=ax)
    ax.set_title(f"Feature correlations with {target}")
    ax.invert_yaxis()
    _save(fig, output_dir / "churn_correlations.png")

    # ---------------------------------------------------------- Bivariate
    if "regularity" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=target, y="regularity", data=df, ax=ax)
        ax.set_title(f"regularity by {target}")
        _save(fig, output_dir / "regularity_vs_churn.png")

    # ---------------------------------------------------------- Summary JSON
    summary = {
        "input": str(input_path),
        "shape": list(df.shape),
        "churn_rate": float(df[target].mean()),
        "duplicates": duplicates,
        "zero_variance_object_cols": zero_var_obj,
        "missing_top": (
            missing_pos.head(15).reset_index().to_dict(orient="records")
        ),
        "top_correlations_with_churn": churn_corr.head(15).round(4).to_dict(),
    }

    json_path = output_dir / "eda_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("wrote summary -> %s", json_path)
    return json_path
