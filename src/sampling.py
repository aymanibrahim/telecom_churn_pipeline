"""Stratified sampling of the raw Expresso dataset down to 100k rows.

Source notebook: notebooks/generate_expresso_sample_100k.ipynb

The raw expresso.csv is ~2M rows (248 MB). To keep training fast and
deterministic we draw a chunked stratified sample. The original notebook
sampled by chunk frequency (no churn-stratification); we keep that
behaviour but expose the ``stratify_col`` parameter for future overrides.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .logging_setup import get_logger

log = get_logger(__name__)


def generate_expresso_sample(
    input_path: Path,
    output_path: Path,
    n: int = 100_000,
    chunksize: int = 200_000,
    estimated_rows: int = 2_000_000,
    seed: int = 42,
) -> Path:
    """Stream-sample the Expresso CSV into a 100k-row file.

    Parameters
    ----------
    input_path
        Full Expresso CSV (~2M rows).
    output_path
        Destination CSV.
    n
        Target row count.
    chunksize
        Pandas read_csv chunksize.
    estimated_rows
        Used to compute the per-chunk sampling fraction.
    seed
        Random seed for reproducibility.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Expresso raw CSV not found: {input_path}")

    frac = n / estimated_rows
    log.info("Reading %s in chunks of %s (target ≈ %s rows, frac=%.4f)",
             input_path.name, f"{chunksize:,}", f"{n:,}", frac)

    sample_chunks: list[pd.DataFrame] = []
    total_in = 0
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize), start=1):
        total_in += len(chunk)
        sample_chunks.append(chunk.sample(frac=frac, random_state=seed))
        log.debug("chunk %3d : in=%7d  sampled=%6d  total_in=%d",
                  i, len(chunk), len(sample_chunks[-1]), total_in)

    combined = pd.concat(sample_chunks, ignore_index=True)
    log.info("Combined: %s rows from %s input rows",
             f"{len(combined):,}", f"{total_in:,}")

    if len(combined) >= n:
        out = combined.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        log.warning("combined < target (%s < %s); writing full combined frame",
                    f"{len(combined):,}", f"{n:,}")
        out = combined.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    log.info("Wrote %s : %s rows (%.1f MB)",
             output_path, f"{len(out):,}", size_mb)
    return output_path


# Backwards-compatible alias for legacy callers.
def run(input_path: Path, output_path: Path) -> Path:
    s = config.CONFIG["sample"]
    return generate_expresso_sample(
        input_path=input_path,
        output_path=output_path,
        n=s["n"],
        chunksize=s["input_chunksize"],
        estimated_rows=s["input_estimated_rows"],
        seed=s["seed"],
    )
