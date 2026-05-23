"""CLI entrypoint: run headless EDA and write plots + JSON summary.

Wraps :func:`src.eda.run_eda`.

Usage
-----
Single-scale (default — writes to ``outputs/eda/``):

    python scripts/04_run_eda.py                              # 100k input
    python scripts/04_run_eda.py --input <path/to/csv>
    python scripts/04_run_eda.py --output-dir <path/to/dir>

Dual-scale (runs EDA on both 100k AND full 2M, writes into
``outputs/eda/100k/`` + ``outputs/eda/full/`` so the top-level
``outputs/eda/*.png`` references stay intact):

    python scripts/04_run_eda.py --all-scales

After ``--all-scales`` you get:

* ``outputs/eda/100k/`` and ``outputs/eda/full/`` with 6 PNGs + ``eda_summary.json``
* ``outputs/eda/eda_scales_comparison.csv`` — side-by-side stats per scale
  (row count, churn rate, missing-rate top-10, top-15 correlations with churn,
  delta-from-notebook flag)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src import eda, paths


# Canonical "notebook" insights — used to flag drift in the comparison
# CSV. Sourced from the original 100k EDA notebook run, kept here so the
# CSV can be diffed without re-running the notebook.
NOTEBOOK_BASELINE = {
    "churn_rate":             0.188,  # ~18.8%
    "top_missing_columns":    ["zone2", "zone1", "mrg"],
    "top_correlate_feature":  "regularity",
    "top_correlate_value":    -0.48,  # approximate
}


def _load_summary(json_path: Path) -> dict:
    if not json_path.exists():
        return {}
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


def _format_top_corr(top_corr: dict) -> str:
    """Render the top-15 correlations as `feature=±0.123` joined by spaces."""
    items = list(top_corr.items())[:15]
    return " ".join(f"{k}={float(v):+.3f}" for k, v in items)


def _write_comparison(scale_to_summary: dict[str, dict], out: Path) -> None:
    """Render outputs/eda/eda_scales_comparison.csv from the per-scale summaries."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "scale",
        "input",
        "n_rows",
        "n_cols",
        "churn_rate",
        "duplicates",
        "zero_var_object_cols",
        "missing_top_3",
        "top_correlates_with_churn",
        "delta_vs_notebook",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for tag, s in scale_to_summary.items():
            rows = s.get("shape", [0, 0])
            churn_rate = float(s.get("churn_rate", 0))
            miss_top = s.get("missing_top", [])
            miss_top3 = "; ".join(
                f"{r['index']}={r['missing_pct']:.1f}%" for r in miss_top[:3]
            )
            top_corr = s.get("top_correlations_with_churn", {})
            delta = abs(churn_rate - NOTEBOOK_BASELINE["churn_rate"])
            delta_note = (f"churn_rate Δ {delta:+.3f} vs notebook "
                          f"({NOTEBOOK_BASELINE['churn_rate']:.3f})")
            w.writerow([
                tag,
                s.get("input", ""),
                rows[0] if rows else "",
                rows[1] if rows else "",
                f"{churn_rate:.4f}",
                s.get("duplicates", ""),
                "|".join(s.get("zero_variance_object_cols", []) or []),
                miss_top3,
                _format_top_corr(top_corr),
                delta_note,
            ])
    print(f"\n[eda] wrote comparison → {out}")

    # Echo
    print("\n  EDA side-by-side comparison")
    print("  " + "-" * 100)
    for tag, s in scale_to_summary.items():
        churn = s.get("churn_rate", 0)
        shape = s.get("shape", [0, 0])
        top_corr = s.get("top_correlations_with_churn", {})
        first_corr = next(iter(top_corr.items())) if top_corr else ("?", 0)
        print(f"  {tag:<6}  n={shape[0]:>10,}  churn={churn:>6.3%}  "
              f"top corr: {first_corr[0]} ({float(first_corr[1]):+.3f})")
    print("  " + "-" * 100)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=paths.TELECOM_CHURN_100K,
                   help="Path to a telecom_churn CSV (single-scale mode)")
    p.add_argument("--output-dir", type=Path, default=paths.EDA_DIR,
                   help="Directory to write EDA artefacts into (single-scale mode)")
    p.add_argument("--all-scales", action="store_true",
                   help=("Run EDA on BOTH telecom_churn_100k AND telecom_churn "
                         "(full 2M). Writes to outputs/eda/100k/ + "
                         "outputs/eda/full/ + an eda_scales_comparison.csv."))
    args = p.parse_args()

    if args.all_scales:
        scales: list[tuple[str, Path]] = [
            ("100k", paths.TELECOM_CHURN_100K),
            ("full", paths.TELECOM_CHURN_FULL),
        ]
        summaries: dict[str, dict] = {}
        for tag, csv_path in scales:
            if not csv_path.exists():
                print(f"[eda] skip {tag} — {csv_path} not found", file=sys.stderr)
                continue
            out_dir = paths.EDA_DIR / tag
            print(f"\n{'=' * 78}\n[eda] === Scale: {tag}  →  {csv_path}\n"
                  f"  output: {out_dir}\n{'=' * 78}\n")
            eda.run_eda(input_path=csv_path, output_dir=out_dir)
            summaries[tag] = _load_summary(out_dir / "eda_summary.json")

        if len(summaries) >= 2:
            _write_comparison(summaries, paths.EDA_DIR / "eda_scales_comparison.csv")
        else:
            print("\n[eda] WARNING: --all-scales found fewer than 2 usable inputs; "
                  "no comparison written.", file=sys.stderr)
    else:
        eda.run_eda(input_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
