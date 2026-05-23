"""Optional script: download the three raw inputs the pipeline expects.

This is **not** wired into the Airflow DAG. Run it manually on a fresh
checkout if your ``datasets/`` folder is empty.

Notes
-----
* ``expresso.csv`` and ``Africa_towers.csv`` come from Kaggle. The
  Kaggle CLI must be configured (``~/.kaggle/kaggle.json``).
* ``gadm41_SEN.gpkg`` is fetched directly from geodata.ucdavis.edu.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401 — sys.path bootstrap

from src import geo, paths


def _kaggle_download(dataset: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] kaggle datasets download -d {dataset} -p {dest_dir}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest_dir), "--unzip"],
        check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-kaggle", action="store_true",
                   help="Only download the GADM file (assumes Kaggle CSVs are already present).")
    p.add_argument("--expresso-kaggle", default="aymericguibert/expresso-churn",
                   help="Kaggle dataset slug for Expresso.")
    p.add_argument("--opencellid-kaggle", default="harshvinayak/opencelliddata",
                   help="Kaggle dataset slug for OpenCellID Africa towers.")
    args = p.parse_args()

    if not args.skip_kaggle:
        if not paths.EXPRESSO_RAW.exists():
            _kaggle_download(args.expresso_kaggle, paths.EXPRESSO_RAW.parent)
        else:
            print(f"[download] {paths.EXPRESSO_RAW} already exists — skipped")

        if not paths.OPENCELLID_RAW.exists():
            _kaggle_download(args.opencellid_kaggle, paths.OPENCELLID_RAW.parent)
        else:
            print(f"[download] {paths.OPENCELLID_RAW} already exists — skipped")

    geo.download_gadm(paths.GADM_GPKG)
    print("[download] All raw inputs ready.")


if __name__ == "__main__":
    main()
