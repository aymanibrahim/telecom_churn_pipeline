"""CLI entrypoint: train + calibrate the LightGBM churn pipeline.

Wraps :func:`src.modeling.train`.

Usage
-----
Single-scale (default):

    python scripts/05_train_model.py                          # uses TELECOM_CHURN_100K
    python scripts/05_train_model.py --input <path/to/csv>    # custom input

Dual-scale (runs both 100k and the full 2M dataset, saves suffixed
artefacts, and writes a side-by-side comparison CSV):

    python scripts/05_train_model.py --all-scales --no-shap

After ``--all-scales`` you get:

* ``models/churn_model_100k.joblib`` + ``models/churn_model_full.joblib``
* ``outputs/metrics/evaluation_metrics_{100k,full}.csv``
* ``outputs/metrics/feature_importances_{100k,full}.csv``
* ``outputs/metrics/scales_comparison.csv``  (one row per scale)
* The canonical ``models/churn_model.joblib`` is left at whichever scale
  ran last (so the dashboard / Flask continue to work without changes).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src import modeling, paths


def _copy_canonical_to_suffixed(model_dir: Path, metrics_dir: Path, tag: str) -> None:
    """After a train() call, copy the latest churn_model.joblib + metric CSVs
    to suffixed filenames so the next iteration doesn't overwrite them."""
    mapping = [
        (model_dir / "churn_model.joblib", model_dir / f"churn_model_{tag}.joblib"),
        (metrics_dir / "evaluation_metrics.csv",
         metrics_dir / f"evaluation_metrics_{tag}.csv"),
        (metrics_dir / "feature_importances.csv",
         metrics_dir / f"feature_importances_{tag}.csv"),
    ]
    for src, dst in mapping:
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"[train] snapshot → {dst}")


def _write_comparison(results: list[tuple[str, dict]], out: Path) -> None:
    """Render a one-CSV-per-scale comparison report."""
    import csv
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc",
            "avg_precision", "threshold"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scale", *keys, "model_path"])
        for tag, result in results:
            metrics = result.get("metrics", {})
            w.writerow([
                tag,
                *[f"{float(metrics.get(k, 0.0)):.4f}" for k in keys],
                result.get("model_path", ""),
            ])
    print(f"\n[train] wrote comparison → {out}")
    # Echo to stdout
    print("\n  Side-by-side comparison")
    print("  " + "-" * 80)
    print(f"  {'scale':<8}" + "".join(f"{k:>14}" for k in keys))
    for tag, result in results:
        metrics = result.get("metrics", {})
        print(f"  {tag:<8}" + "".join(
            f"{float(metrics.get(k, 0.0)):>14.4f}" for k in keys))
    print("  " + "-" * 80)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=paths.TELECOM_CHURN_100K,
                   help="Training dataset CSV (single-scale mode)")
    p.add_argument("--all-scales", action="store_true",
                   help=("Train on BOTH telecom_churn_100k AND telecom_churn "
                         "(full 2M). Saves suffixed artefacts + a comparison CSV. "
                         "Overrides --input."))
    p.add_argument("--model-dir", type=Path, default=paths.MODELS_DIR)
    p.add_argument("--metrics-dir", type=Path, default=paths.METRICS_DIR)
    p.add_argument("--no-shap", action="store_true",
                   help="Skip the SHAP explainability step (faster).")
    args = p.parse_args()

    if args.all_scales:
        scales: list[tuple[str, Path]] = [
            ("100k", paths.TELECOM_CHURN_100K),
            ("full", paths.TELECOM_CHURN_FULL),
        ]
        results: list[tuple[str, dict]] = []
        for tag, csv_path in scales:
            if not csv_path.exists():
                print(f"[train] skip {tag} — {csv_path} not found", file=sys.stderr)
                continue
            print(f"\n{'=' * 78}\n[train] === Scale: {tag}  →  {csv_path}\n{'=' * 78}\n")
            result = modeling.train(
                input_path=csv_path,
                model_dir=args.model_dir,
                metrics_dir=args.metrics_dir,
                run_shap=not args.no_shap,
            )
            _copy_canonical_to_suffixed(args.model_dir, args.metrics_dir, tag)
            results.append((tag, result))

        if len(results) >= 2:
            _write_comparison(
                results,
                args.metrics_dir / "scales_comparison.csv",
            )
        else:
            print("\n[train] WARNING: --all-scales found fewer than 2 usable inputs; "
                  "no comparison written.", file=sys.stderr)
    else:
        modeling.train(
            input_path=args.input,
            model_dir=args.model_dir,
            metrics_dir=args.metrics_dir,
            run_shap=not args.no_shap,
        )


if __name__ == "__main__":
    main()
