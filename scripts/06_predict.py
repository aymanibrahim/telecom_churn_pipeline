"""CLI entrypoint: batch-score a CSV with the saved churn model.

Wraps :func:`src.inference.predict`.

Usage
-----
Single-scale (default):

    python scripts/06_predict.py                              # default 100k input
    python scripts/06_predict.py --input <path/to/csv>        # custom input

Dual-scale (uses the suffixed joblibs that ``05_train_model.py
--all-scales`` produced and writes suffixed prediction CSVs):

    python scripts/06_predict.py --all-scales

After ``--all-scales`` you get:

* ``outputs/predictions/churn_predictions_100k.csv``
* ``outputs/predictions/churn_predictions_full.csv``

Each uses the model trained on its matching scale (so the 100k joblib
scores the 100k dataset, and the 2M joblib scores the 2M dataset).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src import inference, paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=paths.CHURN_MODEL_JOBLIB,
                   help="Path to churn_model.joblib (single-scale mode)")
    p.add_argument("--input", type=Path, default=paths.TELECOM_CHURN_100K,
                   help="CSV to score (single-scale mode)")
    p.add_argument("--output", type=Path, default=paths.PREDICTIONS_CSV,
                   help="Destination CSV (read by Streamlit dashboard)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override the threshold from model metadata")
    p.add_argument("--all-scales", action="store_true",
                   help=("Run inference twice: once with churn_model_100k.joblib "
                         "on telecom_churn_100k.csv, once with churn_model_full.joblib "
                         "on telecom_churn.csv. Writes suffixed prediction CSVs."))
    args = p.parse_args()

    if args.all_scales:
        scales = [
            ("100k", paths.MODELS_DIR / "churn_model_100k.joblib",
             paths.TELECOM_CHURN_100K,
             paths.PREDICTIONS_DIR / "churn_predictions_100k.csv"),
            ("full", paths.MODELS_DIR / "churn_model_full.joblib",
             paths.TELECOM_CHURN_FULL,
             paths.PREDICTIONS_DIR / "churn_predictions_full.csv"),
        ]
        for tag, model_path, input_path, output_path in scales:
            if not model_path.exists():
                print(f"[predict] skip {tag} — {model_path} not found "
                      f"(run scripts/05_train_model.py --all-scales first)",
                      file=sys.stderr)
                continue
            if not input_path.exists():
                print(f"[predict] skip {tag} — {input_path} not found",
                      file=sys.stderr)
                continue
            print(f"\n{'=' * 78}\n[predict] === Scale: {tag}\n"
                  f"  model:  {model_path}\n  input:  {input_path}\n"
                  f"  output: {output_path}\n{'=' * 78}\n")
            inference.predict(
                model_path=model_path,
                input_path=input_path,
                output_path=output_path,
                threshold=args.threshold,
            )
    else:
        inference.predict(
            model_path=args.model,
            input_path=args.input,
            output_path=args.output,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
