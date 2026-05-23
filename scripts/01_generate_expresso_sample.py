"""CLI entrypoint: stratified-sample expresso.csv into expresso_sample_100k.csv.

Wraps :func:`src.sampling.generate_expresso_sample`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src import config, paths, sampling


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=paths.EXPRESSO_RAW,
                   help="Path to raw expresso.csv (default: %(default)s)")
    p.add_argument("--output", type=Path, default=paths.EXPRESSO_SAMPLE,
                   help="Destination sample CSV (default: %(default)s)")
    p.add_argument("--n", type=int, default=config.CONFIG["sample"]["n"],
                   help="Target sample size (default: %(default)s)")
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    sampling.generate_expresso_sample(
        input_path=args.input,
        output_path=args.output,
        n=args.n,
        chunksize=config.CONFIG["sample"]["input_chunksize"],
        estimated_rows=config.CONFIG["sample"]["input_estimated_rows"],
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
