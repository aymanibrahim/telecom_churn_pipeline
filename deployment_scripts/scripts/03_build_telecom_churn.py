"""CLI entrypoint: spatial-join cells + Expresso → telecom_churn.csv (+ 100k).

Wraps :func:`src.geo.build_telecom_churn`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src import config, geo, paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", type=Path, default=paths.OPENCELLID_SENEGAL,
                   help="Path to opencellid_senegal_90d.csv")
    p.add_argument("--expresso", type=Path, default=paths.EXPRESSO_SAMPLE,
                   help="Path to Expresso CSV (sample or full)")
    p.add_argument("--gadm", type=Path, default=paths.GADM_GPKG,
                   help="Path to gadm41_SEN.gpkg")
    p.add_argument("--out-full", type=Path, default=paths.TELECOM_CHURN_FULL)
    p.add_argument("--out-100k", type=Path, default=paths.TELECOM_CHURN_100K)
    p.add_argument("--sample-size", type=int,
                   default=config.CONFIG["sample"]["n"])
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    geo.build_telecom_churn(
        cells_path=args.cells,
        expresso_path=args.expresso,
        gadm_path=args.gadm,
        out_full=args.out_full,
        out_100k=args.out_100k,
        sample_size_100k=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
