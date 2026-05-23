"""OpenCellID + GADM geospatial pipeline.

Two main entrypoints:

* ``build_senegal_cells``  : filter the global ``Africa_towers.csv`` down
                              to Senegal/Expresso cells in a 90-day window.
* ``build_telecom_churn``  : join the filtered cells to GADM admin levels,
                              compute network KPIs per admin polygon, and
                              merge those KPIs onto the Expresso customer
                              table.

Source notebooks
----------------
* notebooks/build_opencellid_senegal_90d_dataset.ipynb
* notebooks/build_telecom_churn_dataset.ipynb
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import config, network_kpis
from .logging_setup import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# OpenCellID dtype map (matches the source notebook)
# ---------------------------------------------------------------------------
OPENCELLID_DTYPES: dict[str, str] = {
    "radio": "category", "MCC": "int16", "MNC": "int8", "TAC": "int32",
    "CID": "int64", "unit": "int16", "LON": "float32", "LAT": "float32",
    "RANGE": "float32", "SAM": "int16", "changeable": "int8",
    "created": "int64", "updated": "int64", "averageSignal": "float32",
    "Country": "category", "Network": "category", "Continent": "category",
}
OPENCELLID_COLS: list[str] = list(OPENCELLID_DTYPES.keys())

# Expresso dtype map (lowercase column names happen at the very end).
EXPRESSO_DTYPES: dict[str, str] = {
    "user_id": "object", "REGION": "category", "TENURE": "category",
    "MONTANT": "float32", "FREQUENCE_RECH": "float32", "REVENUE": "float32",
    "ARPU_SEGMENT": "float32", "FREQUENCE": "float32", "DATA_VOLUME": "float32",
    "ON_NET": "float32", "ORANGE": "float32", "TIGO": "float32",
    "ZONE1": "float32", "ZONE2": "float32", "MRG": "category",
    "REGULARITY": "int16", "TOP_PACK": "category", "FREQ_TOP_PACK": "float32",
    "CHURN": "int8",
}
EXPRESSO_COLS: list[str] = list(EXPRESSO_DTYPES.keys())

# GADM URL — used by 00_download_raw.py if the file is missing.
GADM_URL: str = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_SEN.gpkg"


# ---------------------------------------------------------------------------
# Step 2 of the pipeline — build_senegal_cells
# ---------------------------------------------------------------------------
def build_senegal_cells(
    input_path: Path,
    output_path: Path,
    mcc: int | None = None,
    mnc: int | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> Path:
    """Filter Africa_towers.csv to Senegal + Expresso, within an observation window.

    Uses Dask so the 270 MB source file does not have to fit in RAM.
    Falls back to chunked pandas if dask is unavailable.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Africa_towers.csv not found: {input_path}")

    cfg = config.CONFIG["opencellid"]
    mcc = cfg["mcc"] if mcc is None else mcc
    mnc = cfg["mnc"] if mnc is None else mnc
    window_start = window_start or cfg["ref_window_start"]
    window_end = window_end or cfg["ref_window_end"]

    log.info("Loading %s (MCC=%s, MNC=%s)", input_path.name, mcc, mnc)

    # Try Dask first; fall back to pandas chunks if not installed.
    try:
        import dask.dataframe as dd
        from dask.diagnostics import ProgressBar

        ddf = dd.read_csv(
            input_path,
            usecols=OPENCELLID_COLS,
            dtype=OPENCELLID_DTYPES,
            blocksize="32MB",
        )
        log.info("Dask partitions: %s", ddf.npartitions)

        ddf = ddf[(ddf["MCC"] == mcc) & (ddf["MNC"] == mnc)].persist()
        with ProgressBar():
            cells = ddf.compute()

        for col in ("Country", "Network", "Continent"):
            if hasattr(cells[col], "cat"):
                cells[col] = cells[col].cat.remove_unused_categories()
    except ImportError:
        log.warning("Dask not available — falling back to pandas chunks.")
        chunks = []
        for chunk in pd.read_csv(
            input_path, usecols=OPENCELLID_COLS, dtype=OPENCELLID_DTYPES,
            chunksize=500_000,
        ):
            chunks.append(chunk[(chunk["MCC"] == mcc) & (chunk["MNC"] == mnc)])
        cells = pd.concat(chunks, ignore_index=True)

    cells = cells.reset_index(drop=True)
    log.info("After MCC/MNC filter: %s rows", f"{len(cells):,}")

    # UNIX timestamps → datetime
    cells["created_date"] = pd.to_datetime(cells["created"], unit="s")
    cells["updated_date"] = pd.to_datetime(cells["updated"], unit="s")

    end = pd.to_datetime(window_end)
    mask = (cells["created_date"] <= end) & (cells["updated_date"] <= end)
    cells = cells.loc[mask].reset_index(drop=True)
    log.info("After %s..%s window: %s rows",
             window_start, window_end, f"{len(cells):,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    log.info("Wrote %s (%.1f MB)", output_path, size_mb)
    return output_path


# ---------------------------------------------------------------------------
# GADM helpers (lazy-imported so geopandas is only required when needed)
# ---------------------------------------------------------------------------
def _import_geo():
    import geopandas as gpd
    from shapely.geometry import Point
    return gpd, Point


def _load_gadm_layer(gpkg: Path, layer: str, label: str):
    gpd, _ = _import_geo()
    gdf = gpd.read_file(gpkg, layer=layer).to_crs("EPSG:4326")
    log.info("%s: %s polygons | CRS: %s", label, len(gdf), gdf.crs)
    return gdf


def _spatial_join(gdf_towers, gdf_poly, poly_col: str, out_col: str):
    gpd, _ = _import_geo()
    joined = gpd.sjoin(
        gdf_towers,
        gdf_poly[[poly_col, "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    return joined.rename(columns={poly_col: out_col})


# ---------------------------------------------------------------------------
# Step 3 of the pipeline — build_telecom_churn
# ---------------------------------------------------------------------------
def build_telecom_churn(
    cells_path: Path,
    expresso_path: Path,
    gadm_path: Path,
    out_full: Path,
    out_100k: Path,
    sample_size_100k: int = 100_000,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Spatial-join cells to GADM, merge KPIs into the Expresso table.

    Returns
    -------
    (out_full, out_100k)
        Paths to the full and 100k-sample telecom_churn CSV files.
    """
    cells_path = Path(cells_path)
    expresso_path = Path(expresso_path)
    gadm_path = Path(gadm_path)
    out_full = Path(out_full)
    out_100k = Path(out_100k)

    for p in (cells_path, expresso_path, gadm_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    gpd, Point = _import_geo()

    # ------------------------------------------------------------------ GADM
    gdf_regions = _load_gadm_layer(gadm_path, "ADM_ADM_1", "ADM1 Regions")
    gdf_departments = _load_gadm_layer(gadm_path, "ADM_ADM_2", "ADM2 Departments")
    gdf_arrondissements = _load_gadm_layer(gadm_path, "ADM_ADM_3", "ADM3 Arrondissements")

    REGION_COL, DEPT_COL, ARR_COL = "NAME_1", "NAME_2", "NAME_3"

    # ---------------------------------------------------------------- Cells
    log.info("Loading cells: %s", cells_path.name)
    cells = pd.read_csv(cells_path)
    geometry = [Point(xy) for xy in zip(cells["LON"], cells["LAT"], strict=False)]
    gdf_cells = gpd.GeoDataFrame(cells, geometry=geometry, crs="EPSG:4326")
    log.info("%s cells loaded as GeoDataFrame", f"{len(gdf_cells):,}")

    # ----------------------------------------------- Spatial join + KPI tables
    gdf_region = _spatial_join(gdf_cells, gdf_regions, REGION_COL, "REGION")
    region_kpis = network_kpis.compute_region_kpis(gdf_region)
    region_kpis["REGION"] = region_kpis["REGION"].astype(str).str.strip().str.upper()

    gdf_dept = _spatial_join(gdf_cells, gdf_departments, DEPT_COL, "department")
    gdf_dept = _spatial_join(gdf_dept, gdf_regions, REGION_COL, "REGION")
    dept_kpis = network_kpis.compute_department_kpis_per_region(gdf_dept)
    dept_kpis["REGION"] = dept_kpis["REGION"].astype(str).str.strip().str.upper()

    gdf_arr = _spatial_join(gdf_cells, gdf_arrondissements, ARR_COL, "arrondissement")
    gdf_arr = _spatial_join(gdf_arr, gdf_regions, REGION_COL, "REGION")
    arr_kpis = network_kpis.compute_arrondissement_kpis_per_region(gdf_arr)
    arr_kpis["REGION"] = arr_kpis["REGION"].astype(str).str.strip().str.upper()

    # Drop dupes per REGION (small frames; full pandas merge is fine).
    region_kpis = region_kpis.drop_duplicates(subset=["REGION"])
    dept_kpis = dept_kpis.drop_duplicates(subset=["REGION"])
    arr_kpis = arr_kpis.drop_duplicates(subset=["REGION"])

    # -------------------------------------------------- Expresso (with Dask)
    try:
        import dask.dataframe as dd
        from dask.diagnostics import ProgressBar

        ddf = dd.read_csv(
            expresso_path,
            usecols=EXPRESSO_COLS,
            dtype=EXPRESSO_DTYPES,
            blocksize="32MB",
            assume_missing=True,
        )
        ddf["REGION"] = ddf["REGION"].astype(str).str.strip().str.upper()
        for kdf in (region_kpis, dept_kpis, arr_kpis):
            kdf["REGION"] = kdf["REGION"].astype(str)
        ddf = ddf.merge(region_kpis, on="REGION", how="left")
        ddf = ddf.merge(dept_kpis, on="REGION", how="left")
        ddf = ddf.merge(arr_kpis, on="REGION", how="left")
        with ProgressBar():
            telecom = ddf.compute()
    except ImportError:
        log.warning("Dask not available — using pandas read_csv (may need RAM).")
        df = pd.read_csv(expresso_path, usecols=EXPRESSO_COLS, dtype=EXPRESSO_DTYPES)
        df["REGION"] = df["REGION"].astype(str).str.strip().str.upper()
        df = df.merge(region_kpis, on="REGION", how="left")
        df = df.merge(dept_kpis, on="REGION", how="left")
        df = df.merge(arr_kpis, on="REGION", how="left")
        telecom = df

    telecom = telecom.reset_index(drop=True)
    telecom.columns = [c.lower() for c in telecom.columns]

    final_cols = [
        "user_id", "region", "tenure", "montant", "frequence_rech", "revenue",
        "arpu_segment", "frequence", "data_volume", "on_net", "orange", "tigo",
        "zone1", "zone2", "mrg", "regularity", "top_pack", "freq_top_pack",
        "region_tower_count", "region_avg_range", "region_avg_samples",
        "region_avg_signal", "region_coverage_index",
        "region_signal_strength_index", "region_network_quality_score",
        "department_signal_strength_index", "department_network_quality_score",
        "department_coverage_index",
        "arr_signal_strength_index", "arr_network_quality_score",
        "arr_coverage_index",
        "churn",
    ]
    telecom = telecom[[c for c in final_cols if c in telecom.columns]]

    log.info("telecom_churn final shape: %s", telecom.shape)
    log.info("Churn distribution:\n%s",
             telecom["churn"].value_counts(normalize=True).round(4))

    out_full.parent.mkdir(parents=True, exist_ok=True)
    telecom.to_csv(out_full, index=False)
    size_mb = out_full.stat().st_size / 1024 / 1024
    log.info("Wrote full %s (%.1f MB, %s rows)",
             out_full, size_mb, f"{len(telecom):,}")

    # 100k sample (stratified on churn).
    out_100k.parent.mkdir(parents=True, exist_ok=True)
    if len(telecom) > sample_size_100k:
        sampled = (
            telecom.groupby("churn", group_keys=False, observed=True)
            .apply(lambda g: g.sample(
                n=round(sample_size_100k * len(g) / len(telecom)),
                random_state=seed,
            ))
            .reset_index(drop=True)
        )
    else:
        sampled = telecom.copy()
    sampled.to_csv(out_100k, index=False)
    size_mb = out_100k.stat().st_size / 1024 / 1024
    log.info("Wrote 100k %s (%.1f MB, %s rows)",
             out_100k, size_mb, f"{len(sampled):,}")

    return out_full, out_100k


# ---------------------------------------------------------------------------
# GADM downloader (used by 00_download_raw.py)
# ---------------------------------------------------------------------------
def download_gadm(dest: Path, retries: int = 3) -> Path:
    """Download the Senegal GADM v4.1 GeoPackage if it's not already cached."""
    import requests

    dest = Path(dest)
    if dest.exists():
        mb = dest.stat().st_size / 1024 / 1024
        log.info("GADM already cached at %s (%.1f MB)", dest, mb)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        log.info("Downloading GADM SEN (attempt %s/%s)", attempt, retries)
        try:
            r = requests.get(GADM_URL, stream=True, timeout=180)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        f.write(chunk)
                log.info("Saved %s", dest)
                return dest
            log.warning("HTTP %s", r.status_code)
        except Exception as exc:
            log.warning("Attempt %s failed: %s", attempt, exc)
        time.sleep(3)
    raise RuntimeError(f"Failed to download GADM after {retries} attempts: {GADM_URL}")
