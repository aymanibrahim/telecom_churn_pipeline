"""Generate ML-evaluation PNGs from the canonical predictions + model.

Produces, under ``outputs/metrics/``:

* ``confusion_matrix.png``        — heatmap of TP/FP/FN/TN
* ``roc_curve.png``               — ROC with AUC annotation
* ``precision_recall_curve.png``  — PR curve with AP annotation
* ``calibration_plot.png``        — reliability diagram (10 bins)
* ``feature_importance_top20.png``— top-20 LightGBM feature importances
* ``prediction_distribution.png`` — churn-probability histogram by risk segment
* ``risk_segment_pie.png``        — pie chart of Low / Medium / High counts

The script is idempotent and overwrites existing PNGs. Run it after
``scripts/05_train_model.py`` + ``scripts/06_predict.py`` have refreshed
the model + predictions on disk.

Usage
-----
    python scripts/generate_review_plots.py
    python scripts/generate_review_plots.py \
        --predictions outputs/predictions/churn_predictions.csv \
        --model       models/churn_model.joblib \
        --output-dir  outputs/metrics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import joblib                    # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

import _bootstrap                # noqa: F401, E402


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {path}")


def plot_confusion_matrix(df: pd.DataFrame, out: Path) -> None:
    from sklearn.metrics import confusion_matrix
    if "actual_churn" not in df.columns:
        print("[plots] skip confusion_matrix: no actual_churn column")
        return
    cm = confusion_matrix(df["actual_churn"], df["churn_prediction"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No churn", "Churn"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No churn", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (n={len(df):,})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    _save(fig, out)


def plot_roc(df: pd.DataFrame, out: Path) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve
    if "actual_churn" not in df.columns:
        return
    fpr, tpr, _ = roc_curve(df["actual_churn"], df["churn_probability"])
    auc = roc_auc_score(df["actual_churn"], df["churn_probability"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="C0", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    _save(fig, out)


def plot_pr(df: pd.DataFrame, out: Path) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve
    if "actual_churn" not in df.columns:
        return
    p, r, _ = precision_recall_curve(df["actual_churn"], df["churn_probability"])
    ap = average_precision_score(df["actual_churn"], df["churn_probability"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r, p, color="C2", lw=2, label=f"AP = {ap:.4f}")
    base = df["actual_churn"].mean()
    ax.axhline(base, ls="--", color="gray", alpha=0.5,
               label=f"chance = {base:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    _save(fig, out)


def plot_calibration(df: pd.DataFrame, out: Path, n_bins: int = 10) -> None:
    if "actual_churn" not in df.columns:
        return
    bins = np.linspace(0, 1, n_bins + 1)
    df = df.copy()
    df["bin"] = pd.cut(df["churn_probability"], bins, include_lowest=True)
    agg = df.groupby("bin", observed=True).agg(
        pred_mean=("churn_probability", "mean"),
        actual_mean=("actual_churn", "mean"),
        count=("actual_churn", "size"),
    ).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="perfect")
    ax.plot(agg["pred_mean"], agg["actual_mean"], "o-", color="C3",
            label="model")
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Observed churn rate")
    ax.set_title(f"Calibration Plot ({n_bins} bins)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out)


def plot_feature_importance(model_path: Path, out: Path, top_n: int = 20) -> None:
    payload = joblib.load(model_path)
    model = payload["model"] if isinstance(payload, dict) else payload
    # CalibratedClassifierCV → calibrated_classifiers_[0].estimator
    base = None
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, "estimator"):
        base = model.estimator
    elif hasattr(model, "named_steps"):
        base = model
    if base is None:
        print("[plots] skip feature_importance: could not unwrap model")
        return
    pre = base.named_steps.get("preprocessor")
    clf = base.named_steps.get("clf")
    if pre is None or clf is None or not hasattr(clf, "feature_importances_"):
        print("[plots] skip feature_importance: model shape not supported")
        return
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(len(clf.feature_importances_))]
    imp = (pd.Series(clf.feature_importances_, index=names)
             .sort_values(ascending=False).head(top_n).iloc[::-1])
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(imp.index, imp.values, color="C0")
    ax.set_xlabel("LightGBM importance (MDI)")
    ax.set_title(f"Top {top_n} feature importances")
    _save(fig, out)


def plot_prediction_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = {"Low": "C2", "Medium": "C1", "High": "C3"}
    for seg in ["Low", "Medium", "High"]:
        sub = df[df["risk_segment"] == seg]["churn_probability"]
        if len(sub):
            ax.hist(sub, bins=40, alpha=0.6, label=seg,
                    color=colours.get(seg, "C0"))
    ax.set_xlabel("Churn probability")
    ax.set_ylabel("Customer count")
    ax.set_title(f"Predicted churn-probability distribution (n={len(df):,})")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out)


def plot_risk_segment_pie(df: pd.DataFrame, out: Path) -> None:
    counts = df["risk_segment"].value_counts().reindex(
        ["Low", "Medium", "High"]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6, 6))
    colours = ["#2ca02c", "#ff7f0e", "#d62728"]
    ax.pie(counts.values, labels=[f"{k}\n{v:,}" for k, v in counts.items()],
           autopct="%1.1f%%", colors=colours, startangle=90)
    ax.set_title(f"Risk segment distribution (n={counts.sum():,})")
    _save(fig, out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path,
                   default=Path("outputs/predictions/churn_predictions.csv"))
    p.add_argument("--model", type=Path,
                   default=Path("models/churn_model.joblib"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/metrics"))
    args = p.parse_args()

    if not args.predictions.exists():
        print(f"ERROR: predictions CSV not found: {args.predictions}",
              file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"ERROR: model joblib not found: {args.model}",
              file=sys.stderr)
        return 1

    print(f"[plots] loading predictions: {args.predictions}")
    df = pd.read_csv(args.predictions)
    print(f"[plots] shape: {df.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(df,        args.output_dir / "confusion_matrix.png")
    plot_roc(df,                      args.output_dir / "roc_curve.png")
    plot_pr(df,                       args.output_dir / "precision_recall_curve.png")
    plot_calibration(df,              args.output_dir / "calibration_plot.png")
    plot_feature_importance(args.model, args.output_dir / "feature_importance_top20.png")
    plot_prediction_distribution(df,  args.output_dir / "prediction_distribution.png")
    plot_risk_segment_pie(df,         args.output_dir / "risk_segment_pie.png")

    print("[plots] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
