"""Generate stage-by-stage outputs for the data-pipeline review catalogue.

For each step of the pipeline (sample → opencellid → build_telecom_churn →
EDA → train → predict), this script writes:

* ``outputs/pipeline_outputs/<NN>_<stage>_summary.txt`` — terse table:
  shape, head, describe, null counts.
* ``outputs/pipeline_outputs/<NN>_<stage>_<chart>.png`` (where applicable)
  — one or two stage-relevant matplotlib figures.

All artefacts are small (text + < 200 KB PNGs) and intended to be committed
into the repo so reviewers can audit each pipeline link without re-running
the heavy code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

import _bootstrap                # noqa: F401, E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[pipe] wrote {path}")


def _write_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[pipe] wrote {path}")


def _summarize(df: pd.DataFrame, sample_rows: int = 5) -> list[str]:
    lines = []
    lines.append(f"Shape:    {df.shape}")
    lines.append(f"Columns:  {df.shape[1]}  ({', '.join(df.columns[:6])}{'...' if df.shape[1] > 6 else ''})")
    lines.append(f"Dtypes:   {df.dtypes.value_counts().to_dict()}")
    lines.append("")
    lines.append(f"Head({sample_rows}):")
    with pd.option_context("display.max_columns", 6, "display.width", 100):
        lines.append(str(df.head(sample_rows)))
    lines.append("")
    lines.append("Null counts (top 10):")
    nulls = df.isna().sum()
    lines.append(str(nulls[nulls > 0].sort_values(ascending=False).head(10)))
    lines.append("")
    lines.append("Numeric describe (top 6 columns):")
    with pd.option_context("display.max_columns", 6, "display.width", 100):
        lines.append(str(df.select_dtypes(include=np.number).describe().T.head(6).round(2)))
    return lines


# ---------------------------------------------------------------------------
# Per-stage tasks
# ---------------------------------------------------------------------------
OUT_DIR = Path("outputs/pipeline_outputs")


def stage_01_sample() -> None:
    """Expresso 100k sample."""
    path = Path("datasets/expresso/expresso_sample_100k.csv")
    if not path.exists():
        print(f"[pipe] skip stage 01 — {path} not found")
        return
    df = pd.read_csv(path)
    _write_summary(
        OUT_DIR / "01_sample_summary.txt",
        ["STAGE 01 — sample_expresso → datasets/expresso/expresso_sample_100k.csv",
         "Source notebook: notebooks/generate_expresso_sample_100k.ipynb",
         "Source script:   scripts/01_generate_expresso_sample.py",
         "", *_summarize(df)],
    )


def stage_02_opencellid() -> None:
    """OpenCellID 90-day Senegal slice."""
    path = Path("datasets/opencellid/opencellid_senegal_90d.csv")
    if not path.exists():
        print(f"[pipe] skip stage 02 — {path} not found")
        return
    df = pd.read_csv(path)
    _write_summary(
        OUT_DIR / "02_opencellid_summary.txt",
        ["STAGE 02 — build_opencellid → datasets/opencellid/opencellid_senegal_90d.csv",
         "Source notebook: notebooks/build_opencellid_senegal_90d_dataset.ipynb",
         "Source script:   scripts/02_build_opencellid_senegal.py",
         f"MCC=608 (Senegal) / MNC=3 (Expresso) / 90-day window",
         "", *_summarize(df)],
    )

    # Scatter of tower locations
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(df["LON"], df["LAT"], c=df["RANGE"], s=20, alpha=0.6,
                    cmap="viridis")
    ax.set_title(f"OpenCellID — Senegal/Expresso cell towers ({len(df):,} cells)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="RANGE (m)")
    _save(fig, OUT_DIR / "02_opencellid_towers_map.png")


def stage_03_telecom_churn() -> None:
    """telecom_churn join (100k variant for documentation)."""
    path = Path("datasets/telecom_churn/telecom_churn_100k.csv")
    if not path.exists():
        print(f"[pipe] skip stage 03 — {path} not found")
        return
    df = pd.read_csv(path)
    _write_summary(
        OUT_DIR / "03_telecom_churn_summary.txt",
        ["STAGE 03 — build_telecom_churn → datasets/telecom_churn/telecom_churn_100k.csv",
         "Source notebook: notebooks/build_telecom_churn_dataset.ipynb",
         "Source script:   scripts/03_build_telecom_churn.py",
         "Operation:       spatial-join GADM admin polygons + Expresso REGION + cell-tower KPIs",
         f"Churn rate:      {df['churn'].mean():.1%}",
         "", *_summarize(df)],
    )

    # Churn rate by region
    by_region = (df.groupby("region")
                   .agg(churn_rate=("churn", "mean"),
                        n=("churn", "size"))
                   .sort_values("churn_rate"))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(by_region))))
    ax.barh(by_region.index, by_region["churn_rate"], color="C3", alpha=0.7)
    ax.set_xlabel("Churn rate")
    ax.set_title(f"Churn rate by region (n={len(df):,})")
    for i, (region, row) in enumerate(by_region.iterrows()):
        ax.text(row["churn_rate"] + 0.002, i, f"n={row['n']:,}",
                va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, OUT_DIR / "03_telecom_churn_by_region.png")

    # Network quality vs churn (boxplot)
    if "region_network_quality_score" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        df.boxplot(column="region_network_quality_score", by="churn", ax=ax)
        ax.set_xticklabels(["No churn", "Churn"])
        ax.set_xlabel("churn")
        ax.set_ylabel("region_network_quality_score")
        ax.set_title("Network quality score vs churn")
        plt.suptitle("")
        _save(fig, OUT_DIR / "03_telecom_churn_network_quality.png")


def stage_04_eda_recap() -> None:
    summary_path = Path("outputs/eda/eda_summary.json")
    if not summary_path.exists():
        print("[pipe] skip stage 04 recap — outputs/eda/eda_summary.json missing")
        return
    _write_summary(
        OUT_DIR / "04_eda_summary_recap.txt",
        ["STAGE 04 — run_eda → outputs/eda/*.png + outputs/eda/eda_summary.json",
         "Source notebook: notebooks/Telecom Customer Churn - EDA - Project Team A.ipynb",
         "Source script:   scripts/04_run_eda.py",
         "",
         "Charts (already in outputs/eda/, embedded in docs/SCREENSHOTS.md):",
         "  - missing_pattern.png",
         "  - target_distribution.png",
         "  - numerical_distributions.png",
         "  - correlation_matrix.png",
         "  - churn_correlations.png",
         "  - regularity_vs_churn.png",
         "",
         "Full JSON summary:",
         summary_path.read_text(encoding="utf-8")],
    )


def stage_05_model() -> None:
    metrics_path = Path("outputs/metrics/evaluation_metrics.csv")
    if not metrics_path.exists():
        print("[pipe] skip stage 05 — metrics CSV missing")
        return
    metrics = pd.read_csv(metrics_path)
    lines = [
        "STAGE 05 — train_model → models/churn_model.joblib + outputs/metrics/*",
        "Source notebook: notebooks/Telecom Customer Churn - Training, Inference & Evaluation - Project Team A.ipynb",
        "Source script:   scripts/05_train_model.py",
        "",
        "Test-set metrics (Full 2M training run):",
    ]
    for _, row in metrics.iterrows():
        lines.append(f"  {row['metric']:<15s} = {float(row['value']):.4f}")
    lines.append("")
    fi_path = Path("outputs/metrics/feature_importances.csv")
    if fi_path.exists():
        fi = pd.read_csv(fi_path).head(15)
        lines.append("Top-15 features by importance:")
        for _, row in fi.iterrows():
            lines.append(f"  {row['feature']:<45s} {float(row['importance']):.2f}")
    lines.append("")
    lines.append("Charts (already in outputs/metrics/, embedded in docs/SCREENSHOTS.md):")
    lines.append("  - confusion_matrix.png, roc_curve.png, precision_recall_curve.png")
    lines.append("  - calibration_plot.png, feature_importance_top20.png")
    lines.append("  - shap_summary.png, prediction_distribution.png, risk_segment_pie.png")
    _write_summary(OUT_DIR / "05_model_summary.txt", lines)


def stage_06_predict() -> None:
    path = Path("outputs/predictions/churn_predictions.csv")
    if not path.exists():
        # Fall back to the committed baseline.
        path = Path("outputs/predictions/churn_predictions_notebook.csv")
    if not path.exists():
        print("[pipe] skip stage 06 — no predictions CSV available")
        return
    df = pd.read_csv(path)
    lines = [
        "STAGE 06 — predict → outputs/predictions/churn_predictions.csv",
        "Source notebook: notebooks/Telecom Customer Churn - Training, Inference & Evaluation - Project Team A.ipynb",
        "Source script:   scripts/06_predict.py",
        f"Source CSV:      {path}",
        "",
        f"Predictions shape: {df.shape}",
        f"Columns:           {sorted(df.columns.tolist())}",
        "",
        "Risk segment distribution:",
        str(df["risk_segment"].value_counts(normalize=True).round(4)),
        "",
        "churn_prediction distribution:",
        str(df["churn_prediction"].value_counts(normalize=True).round(4)),
        "",
        f"Mean churn_probability: {df['churn_probability'].mean():.4f}",
    ]
    if "actual_churn" in df.columns:
        from sklearn.metrics import (accuracy_score, f1_score,
                                     precision_score, recall_score,
                                     roc_auc_score)
        lines.append("")
        lines.append("Live test-set metrics (recomputed from the CSV):")
        y = df["actual_churn"]; p = df["churn_probability"]; pr = df["churn_prediction"]
        lines.append(f"  accuracy = {accuracy_score(y, pr):.4f}")
        lines.append(f"  precision= {precision_score(y, pr, zero_division=0):.4f}")
        lines.append(f"  recall   = {recall_score(y, pr, zero_division=0):.4f}")
        lines.append(f"  f1       = {f1_score(y, pr, zero_division=0):.4f}")
        lines.append(f"  roc_auc  = {roc_auc_score(y, p):.4f}")
    _write_summary(OUT_DIR / "06_predict_summary.txt", lines)


def stage_dag_overview() -> None:
    """Schematic Matplotlib drawing of the DAG (no Mermaid runtime needed)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    nodes = {
        "sample_expresso":     (1, 4),
        "build_opencellid":    (1, 1.5),
        "build_telecom_churn": (4, 2.75),
        "run_eda":             (6, 4),
        "train_model":         (7.5, 2.75),
        "predict":             (9, 2.75),
    }
    for name, (x, y) in nodes.items():
        ax.add_patch(plt.Rectangle((x - 0.9, y - 0.35), 1.8, 0.7,
                                   fc="#e8f0fe", ec="#1a73e8", lw=1.8))
        ax.text(x, y, name, ha="center", va="center", fontsize=9,
                fontweight="bold")

    def arrow(a, b):
        ax.annotate("", xy=(nodes[b][0] - 0.9, nodes[b][1]),
                    xytext=(nodes[a][0] + 0.9, nodes[a][1]),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#5f6368"))

    arrow("sample_expresso", "build_telecom_churn")
    arrow("build_opencellid", "build_telecom_churn")
    arrow("build_telecom_churn", "run_eda")
    arrow("build_telecom_churn", "train_model")
    arrow("run_eda", "train_model")
    arrow("train_model", "predict")

    ax.set_title("telecom_churn_production_pipeline — Airflow DAG", fontsize=12)
    _save(fig, OUT_DIR / "dag_overview.png")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_01_sample()
    stage_02_opencellid()
    stage_03_telecom_churn()
    stage_04_eda_recap()
    stage_05_model()
    stage_06_predict()
    stage_dag_overview()
    print("[pipe] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
