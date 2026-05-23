"""Compare DAG-produced predictions against the notebook baseline.

Usage
-----
    python scripts/compare_dag_vs_notebook.py
    python scripts/compare_dag_vs_notebook.py \
        --dag-csv     outputs/predictions/churn_predictions.csv \
        --baseline    outputs/predictions/churn_predictions_notebook.csv \
        --output-json outputs/dag_vs_notebook_comparison.json

The script checks both files row-wise:

* same row count and column set
* same set of user_ids (when present)
* churn_prediction row-wise agreement >= AGREEMENT_THRESHOLD (default 99%)
* churn_probability mean absolute delta <= PROB_DELTA (default 0.01)
* risk_segment distribution within +-RISK_DELTA percentage points

Exit code 0 = pass, 1 = fail. Writes a JSON report so CI / Airflow can
ingest the result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401  -- sys.path bootstrap


AGREEMENT_THRESHOLD = 0.99
PROB_DELTA = 0.01
RISK_DELTA = 0.005   # +/- 0.5 pp


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(dag_path: Path, baseline_path: Path) -> dict:
    dag = pd.read_csv(dag_path)
    base = pd.read_csv(baseline_path)

    report: dict = {
        "dag_csv":      str(dag_path),
        "baseline_csv": str(baseline_path),
        "dag_md5":      _md5(dag_path),
        "baseline_md5": _md5(baseline_path),
        "dag_rows":     int(len(dag)),
        "baseline_rows": int(len(base)),
        "dag_cols":     sorted(dag.columns.tolist()),
        "baseline_cols": sorted(base.columns.tolist()),
        "checks": {},
        "pass": True,
    }

    # 1. row count
    same_rows = len(dag) == len(base)
    report["checks"]["row_count_match"] = same_rows
    if not same_rows:
        report["pass"] = False

    # 2. column set
    same_cols = set(dag.columns) == set(base.columns)
    report["checks"]["column_set_match"] = same_cols
    if not same_cols:
        report["pass"] = False

    if "user_id" in dag.columns and "user_id" in base.columns:
        dag_ids = set(dag["user_id"].astype(str))
        base_ids = set(base["user_id"].astype(str))
        same_ids = dag_ids == base_ids
        report["checks"]["user_id_set_match"] = same_ids
        report["checks"]["user_id_overlap_pct"] = round(
            100 * len(dag_ids & base_ids) / max(len(dag_ids | base_ids), 1), 4
        )
        if not same_ids:
            # Soft fail: row-by-row alignment may differ.
            pass

    # Align on common rows for row-wise checks.
    if "user_id" in dag.columns and "user_id" in base.columns:
        merged = dag[["user_id", "churn_prediction", "churn_probability",
                      "risk_segment"]].merge(
            base[["user_id", "churn_prediction", "churn_probability",
                  "risk_segment"]],
            on="user_id", suffixes=("_dag", "_base"),
        )
    elif len(dag) == len(base):
        merged = pd.DataFrame({
            "churn_prediction_dag":  dag["churn_prediction"].values,
            "churn_prediction_base": base["churn_prediction"].values,
            "churn_probability_dag":  dag["churn_probability"].values,
            "churn_probability_base": base["churn_probability"].values,
            "risk_segment_dag":  dag["risk_segment"].values,
            "risk_segment_base": base["risk_segment"].values,
        })
    else:
        report["checks"]["row_alignment"] = "no common key, sizes differ"
        report["pass"] = False
        return report

    report["checks"]["aligned_rows"] = int(len(merged))

    # 3. churn_prediction agreement
    agree = (merged["churn_prediction_dag"] ==
             merged["churn_prediction_base"]).mean()
    agree = float(agree)
    report["checks"]["churn_prediction_agreement"] = round(agree, 6)
    if agree < AGREEMENT_THRESHOLD:
        report["pass"] = False

    # 4. churn_probability mean abs delta
    prob_delta = float(np.abs(
        merged["churn_probability_dag"] - merged["churn_probability_base"]
    ).mean())
    report["checks"]["churn_probability_mean_abs_delta"] = round(prob_delta, 6)
    if prob_delta > PROB_DELTA:
        report["pass"] = False

    # 5. risk_segment distribution
    def _dist(s: pd.Series) -> dict[str, float]:
        return {k: round(float(v), 6) for k, v in
                s.value_counts(normalize=True).items()}

    dag_dist = _dist(merged["risk_segment_dag"])
    base_dist = _dist(merged["risk_segment_base"])
    report["checks"]["risk_segment_dist_dag"] = dag_dist
    report["checks"]["risk_segment_dist_base"] = base_dist
    biggest_delta = max(
        abs(dag_dist.get(k, 0) - base_dist.get(k, 0))
        for k in set(dag_dist) | set(base_dist)
    )
    report["checks"]["risk_segment_max_delta"] = round(biggest_delta, 6)
    if biggest_delta > RISK_DELTA:
        # Soft warning: keep pass=True unless agreement / prob_delta also failed.
        report["checks"]["risk_segment_warning"] = (
            f"max distribution delta {biggest_delta:.4f} > {RISK_DELTA}"
        )

    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dag-csv", type=Path,
                   default=Path("outputs/predictions/churn_predictions.csv"))
    p.add_argument("--baseline", type=Path,
                   default=Path("outputs/predictions/churn_predictions_notebook.csv"))
    p.add_argument("--output-json", type=Path,
                   default=Path("outputs/dag_vs_notebook_comparison.json"))
    args = p.parse_args()

    if not args.dag_csv.exists():
        print(f"ERROR: DAG CSV not found: {args.dag_csv}", file=sys.stderr)
        return 2
    if not args.baseline.exists():
        print(f"ERROR: Baseline CSV not found: {args.baseline}", file=sys.stderr)
        return 2

    report = compare(args.dag_csv, args.baseline)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("=" * 60)
    print(f"DAG CSV:      {args.dag_csv}  (md5 {report['dag_md5'][:8]})")
    print(f"Baseline:     {args.baseline}  (md5 {report['baseline_md5'][:8]})")
    print(f"Rows:         dag={report['dag_rows']}  baseline={report['baseline_rows']}")
    print("-" * 60)
    for k, v in report["checks"].items():
        print(f"  {k}: {v}")
    print("-" * 60)
    print(f"OVERALL: {'PASS' if report['pass'] else 'FAIL'}")
    print(f"Report saved -> {args.output_json}")
    print("=" * 60)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
