"""CLI entrypoint: filter Africa_towers.csv into opencellid_senegal_90d.csv.

Wraps :func:`src.geo.build_senegal_cells`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src import config, geo, paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=paths.OPENCELLID_RAW,
                   help="Path to raw Africa_towers.csv")
    p.add_argument("--output", type=Path, default=paths.OPENCELLID_SENEGAL,
                   help="Destination CSV")
    p.add_argument("--mcc", type=int, default=config.CONFIG["opencellid"]["mcc"])
    p.add_argument("--mnc", type=int, default=config.CONFIG["opencellid"]["mnc"])
    p.add_argument("--window-start", default=config.CONFIG["opencellid"]["ref_window_start"])
    p.add_argument("--window-end", default=config.CONFIG["opencellid"]["ref_window_end"])
    args = p.parse_args()

    geo.build_senegal_cells(
        input_path=args.input,
        output_path=args.output,
        mcc=args.mcc,
        mnc=args.mnc,
        window_start=args.window_start,
        window_end=args.window_end,
    )


if __name__ == "__main__":
    main()
