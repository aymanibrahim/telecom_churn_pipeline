"""
processing.py
-------------
PART 1 – STAGE 2: Data Processing & Transformation

Responsibilities
~~~~~~~~~~~~~~~~
* Load the GADM GeoPackage and build Senegal administrative boundary
  GeoDataFrames (ADM1 Region, ADM2 Department, ADM3 Arrondissement).
* Load the filtered OpenCellID tower snapshot produced by the ingestion
  stage and convert tower lat/lon coordinates into Shapely Point geometry.
* Perform point-in-polygon spatial joins at all three administrative levels.
* Compute network KPIs per zone:
      - tower_count, avg_range, avg_samples, avg_signal
      - coverage_index, signal_strength_index, network_quality_score
* Roll up Department and Arrondissement KPIs to the Region level so they
  can be joined onto the customer (Expresso) dataset.
* Load the Expresso dataset with Dask, normalise REGION strings, and merge
  all network KPI tables via broadcast joins.
* Compute the final integrated ``telecom_churn`` dataset (2 M rows) and
  a 100 k-row sample.
* Validate output schemas and row counts.
* Save both outputs to PROCESSED_DIR.

Prediction parity guarantee
~~~~~~~~~~~~~~~~~~~~~~~~~~~
All column selection, normalisation (str.strip().str.upper()), merge
order, and column renaming exactly replicate the notebook workflow so
that downstream model outputs are byte-identical to notebook results.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from shapely.geometry import Point

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import (
    ARRONDISSEMENT_COL,
    DEPARTMENT_COL,
    EXPRESSO_CAT_COLS,
    EXPRESSO_DTYPES,
    EXPRESSO_FILE,
    GADM_GPKG_FILE,
    GADM_LAYERS,
    OPENCELLID_90D_FILE,
    PROCESSED_DIR,
    RANDOM_STATE,
    REGION_COL,
    TARGET_SAMPLE_SIZE,
    TELECOM_CHURN_100K_FILE,
    TELECOM_CHURN_FINAL_COLS,
    TELECOM_CHURN_FILE,
)
from src.logger import get_logger

log = get_logger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1.  GADM LAYER LOADING                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _load_gadm_layer(gpkg: str | Path, layer_key: str, label: str) -> gpd.GeoDataFrame:
    """
    Load a single GADM layer from the GeoPackage and reproject to EPSG:4326.

    Parameters
    ----------
    gpkg      : Path to the GeoPackage file.
    layer_key : One of 'ADM1', 'ADM2', 'ADM3'.
    label     : Human-readable label for logging.
    """
    layer_name = GADM_LAYERS[layer_key]
    gdf = gpd.read_file(str(gpkg), layer=layer_name).to_crs("EPSG:4326")
    log.info("[Processing] %s: %d polygons | CRS: %s", label, len(gdf), gdf.crs)
    return gdf


def load_gadm_boundaries(gpkg: str | Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load ADM1 (Regions), ADM2 (Departments), ADM3 (Arrondissements) from
    the GADM GeoPackage.

    Returns
    -------
    gdf_regions, gdf_departments, gdf_arrondissements
    """
    log.info("[Processing] Loading GADM boundaries from %s …", gpkg)
    log.info("[Processing] Available layers: %s", fiona.listlayers(str(gpkg)))

    gdf_regions         = _load_gadm_layer(gpkg, "ADM1", "ADM1 Regions")
    gdf_departments     = _load_gadm_layer(gpkg, "ADM2", "ADM2 Departments")
    gdf_arrondissements = _load_gadm_layer(gpkg, "ADM3", "ADM3 Arrondissements")

    return gdf_regions, gdf_departments, gdf_arrondissements


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2.  TOWER GEODATAFRAME BUILDER                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_tower_geodataframe(opencellid_path: str | Path) -> gpd.GeoDataFrame:
    """
    Load the filtered OpenCellID CSV and convert it to a GeoDataFrame with
    Shapely Point geometry using LON / LAT columns.

    Parameters
    ----------
    opencellid_path : Path to ``opencellid_senegal_90d.csv``.

    Returns
    -------
    gpd.GeoDataFrame  (CRS = EPSG:4326)
    """
    log.info("[Processing] Loading OpenCellID tower data from %s …", opencellid_path)
    df = pd.read_csv(str(opencellid_path))
    log.info("[Processing] Towers loaded: %d rows", len(df))

    geometry = [Point(xy) for xy in zip(df["LON"], df["LAT"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    log.info("[Processing] Tower GeoDataFrame built: %s", gdf.shape)
    return gdf


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3.  SPATIAL UTILITY FUNCTIONS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def spatial_join_towers(
    gdf_towers: gpd.GeoDataFrame,
    gdf_poly:   gpd.GeoDataFrame,
    poly_col:   str,
    out_col:    str,
) -> gpd.GeoDataFrame:
    """
    Point-in-polygon spatial join.

    Joins tower points into administrative polygons, renames the polygon
    name column from ``poly_col`` to ``out_col``, and removes the
    ``index_right`` artifact produced by ``gpd.sjoin``.

    Parameters
    ----------
    gdf_towers : Tower GeoDataFrame (point geometry).
    gdf_poly   : Administrative boundary GeoDataFrame (polygon geometry).
    poly_col   : Column in *gdf_poly* that holds the zone name.
    out_col    : Desired name for the zone name column in the result.
    """
    result = gpd.sjoin(
        gdf_towers,
        gdf_poly[[poly_col, "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    return result.rename(columns={poly_col: out_col})


def compute_network_kpis(
    gdf:        gpd.GeoDataFrame,
    group_cols: list[str],
    prefix:     str,
) -> pd.DataFrame:
    """
    Aggregate tower-level data to admin-zone KPIs.

    All output columns are prefixed with ``{prefix}_``.

    KPIs computed
    -------------
    ``{prefix}_tower_count``           – number of cell towers in zone
    ``{prefix}_avg_range``             – mean tower range (m)
    ``{prefix}_avg_samples``           – mean measurement samples
    ``{prefix}_avg_signal``            – mean signal strength (dBm)
    ``{prefix}_coverage_index``        – tower_count × avg_range (reach proxy)
    ``{prefix}_signal_strength_index`` – avg_signal / avg_samples (quality proxy)
    ``{prefix}_network_quality_score`` – composite score:
                                         0.4*signal + 0.3*count + 0.3*samples

    Parameters
    ----------
    gdf        : GeoDataFrame with tower data and a zone name column.
    group_cols : Column(s) to group by (e.g. ``["REGION"]``).
    prefix     : Column-name prefix (e.g. ``"region"``).
    """
    agg = (
        gdf.groupby(group_cols)
        .agg(
            **{f"{prefix}_tower_count": ("CID",           "count")},
            **{f"{prefix}_avg_range":   ("RANGE",          "mean")},
            **{f"{prefix}_avg_samples": ("SAM",        "mean")},
            **{f"{prefix}_avg_signal":  ("averageSignal",  "mean")},
        )
        .reset_index()
    )

    agg[f"{prefix}_coverage_index"] = (
        agg[f"{prefix}_tower_count"] * agg[f"{prefix}_avg_range"]
    )
    agg[f"{prefix}_signal_strength_index"] = (
        agg[f"{prefix}_avg_signal"]
        / agg[f"{prefix}_avg_samples"].replace(0, np.nan)
    )
    agg[f"{prefix}_network_quality_score"] = (
        0.4 * agg[f"{prefix}_signal_strength_index"]
        + 0.3 * agg[f"{prefix}_tower_count"]
        + 0.3 * agg[f"{prefix}_avg_samples"]
    )
    return agg


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4.  REGIONAL KPI COMPUTATION                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_all_regional_kpis(
    gdf_cells:          gpd.GeoDataFrame,
    gdf_regions:        gpd.GeoDataFrame,
    gdf_departments:    gpd.GeoDataFrame,
    gdf_arrondissements:gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run spatial joins and KPI aggregations at all three GADM levels.

    Returns
    -------
    network_region_kpis     : Region-level KPIs (one row per region).
    dept_kpis_per_region    : Department KPIs rolled up to region level.
    arr_kpis_per_region     : Arrondissement KPIs rolled up to region level.
    """

    # ── ADM1: Region ────────────────────────────────────────────────────────
    log.info("[Processing] Computing ADM1 Region KPIs …")
    gdf_towers_region = spatial_join_towers(gdf_cells, gdf_regions, REGION_COL, "REGION")
    log.info("[Processing] Tower distribution by region:\n%s",
             gdf_towers_region["REGION"].value_counts(dropna=False).to_string())

    network_region_kpis = compute_network_kpis(gdf_towers_region, ["REGION"], "region")
    network_region_kpis["REGION"] = (
        network_region_kpis["REGION"].str.strip().str.upper()
    )
    log.info("[Processing] ADM1 KPIs computed: %d regions", len(network_region_kpis))

    # ── ADM2: Department → rolled up to Region ───────────────────────────────
    log.info("[Processing] Computing ADM2 Department KPIs …")
    gdf_towers_dept = spatial_join_towers(gdf_cells, gdf_departments, DEPARTMENT_COL, "department")
    gdf_towers_dept = spatial_join_towers(gdf_towers_dept, gdf_regions, REGION_COL, "REGION")

    network_dept_kpis = compute_network_kpis(
        gdf_towers_dept, ["REGION", "department"], "department"
    )
    network_dept_kpis["REGION"] = network_dept_kpis["REGION"].str.strip().str.upper()

    dept_kpis_per_region = (
        network_dept_kpis.groupby("REGION")
        .agg(
            department_coverage_index        =("department_coverage_index",        "sum"),
            department_signal_strength_index =("department_signal_strength_index", "mean"),
            department_network_quality_score =("department_network_quality_score", "mean"),
        )
        .reset_index()
    )
    log.info("[Processing] ADM2 → Region roll-up complete: %d rows", len(dept_kpis_per_region))

    # ── ADM3: Arrondissement → rolled up to Region ───────────────────────────
    log.info("[Processing] Computing ADM3 Arrondissement KPIs …")
    gdf_towers_arr = spatial_join_towers(
        gdf_cells, gdf_arrondissements, ARRONDISSEMENT_COL, "arrondissement"
    )
    gdf_towers_arr = spatial_join_towers(gdf_towers_arr, gdf_regions, REGION_COL, "REGION")

    network_arr_kpis = compute_network_kpis(
        gdf_towers_arr, ["REGION", "arrondissement"], "arrondissement"
    )
    network_arr_kpis["REGION"] = network_arr_kpis["REGION"].str.strip().str.upper()

    arr_kpis_per_region = (
        network_arr_kpis.groupby("REGION")
        .agg(
            arr_coverage_index        =("arrondissement_coverage_index",        "sum"),
            arr_signal_strength_index =("arrondissement_signal_strength_index", "mean"),
            arr_network_quality_score =("arrondissement_network_quality_score", "mean"),
        )
        .reset_index()
    )
    log.info("[Processing] ADM3 → Region roll-up complete: %d rows", len(arr_kpis_per_region))

    return network_region_kpis, dept_kpis_per_region, arr_kpis_per_region


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.  EXPRESSO + NETWORK KPI MERGE (DASK)                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _load_expresso_dask(expresso_path: str | Path) -> dd.DataFrame:
    """Load Expresso CSV with Dask, normalise REGION, and categorise."""
    log.info("[Processing] Loading Expresso dataset from %s …", expresso_path)
    columns = list(EXPRESSO_DTYPES.keys())
    ddf = dd.read_csv(
        str(expresso_path),
        usecols=columns,
        dtype=EXPRESSO_DTYPES,
        blocksize="32MB",
        assume_missing=True,
    )
    log.info("[Processing] Dask partitions: %d", ddf.npartitions)

    # Normalise REGION before categorise so cleaned strings become category labels
    ddf["REGION"] = ddf["REGION"].str.strip().str.upper()
    ddf = ddf.categorize(columns=EXPRESSO_CAT_COLS)
    return ddf


def merge_network_kpis_dask(
    ddf:                  dd.DataFrame,
    network_region_kpis:  pd.DataFrame,
    dept_kpis_per_region: pd.DataFrame,
    arr_kpis_per_region:  pd.DataFrame,
) -> dd.DataFrame:
    """
    Broadcast-merge three small pandas KPI DataFrames into the large Dask
    Expresso DataFrame on the ``REGION`` column.

    Dask broadcasts small DataFrames to every partition — no expensive
    shuffle is needed.
    """
    log.info("[Processing] Aligning REGION categories across KPI tables …")
    region_cats = ddf["REGION"].cat.categories

    for kpi_df in [network_region_kpis, dept_kpis_per_region, arr_kpis_per_region]:
        kpi_df["REGION"] = pd.Categorical(
            kpi_df["REGION"].str.strip().str.upper(),
            categories=region_cats,
        )

    # Deduplicate before merge (safety guard)
    network_region_kpis  = network_region_kpis.drop_duplicates(subset=["REGION"])
    dept_kpis_per_region = dept_kpis_per_region.drop_duplicates(subset=["REGION"])
    arr_kpis_per_region  = arr_kpis_per_region.drop_duplicates(subset=["REGION"])

    log.info("[Processing] Building Dask merge graph …")
    ddf = ddf.merge(network_region_kpis,  on="REGION", how="left")
    ddf = ddf.merge(dept_kpis_per_region, on="REGION", how="left")
    ddf = ddf.merge(arr_kpis_per_region,  on="REGION", how="left")
    log.info("[Processing] Merge graph built | Partitions: %d", ddf.npartitions)
    return ddf


def _compute_and_finalise(ddf: dd.DataFrame) -> pd.DataFrame:
    """
    Trigger Dask computation, lower-case all column names, select the
    canonical column order, and validate row count.
    """
    log.info("[Processing] Computing Dask graph → pandas DataFrame …")
    with ProgressBar():
        df = ddf.compute()

    df = df.reset_index(drop=True)
    log.info("[Processing] Computed shape: %s", df.shape)

    # Normalise column names to lower-case (matches notebook output)
    df.columns = [c.lower() for c in df.columns]

    existing = [c for c in TELECOM_CHURN_FINAL_COLS if c in df.columns]
    missing  = [c for c in TELECOM_CHURN_FINAL_COLS if c not in df.columns]
    if missing:
        log.warning("[Processing] Expected columns not found: %s", missing)

    df = df[existing]
    log.info("[Processing] Final column set: %d columns", len(existing))
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.  DATA VALIDATION                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def validate_processed_output(df: pd.DataFrame, label: str) -> None:
    """
    Run post-processing sanity checks on the integrated dataset.

    Checks
    ------
    * Minimum row count > 0.
    * ``churn`` column exists and has only 0/1 values.
    * ``region`` column is present and non-empty.
    * No wholly-empty columns.
    * Logs missing-value summary.
    """
    log.info("[Processing] Validating '%s' …", label)

    if len(df) == 0:
        raise ValueError(f"[Processing] '{label}' is empty after processing.")

    if "churn" not in df.columns:
        raise ValueError(f"[Processing] 'churn' target column missing from '{label}'.")

    invalid_churn = ~df["churn"].isin([0, 1, np.nan])
    if invalid_churn.any():
        raise ValueError(
            f"[Processing] '{label}' contains non-binary values in 'churn': "
            f"{df.loc[invalid_churn, 'churn'].unique()}"
        )

    churn_rate = df["churn"].mean()
    log.info("[Processing] Churn rate: %.2f%%", churn_rate * 100)
    if not (0.05 <= churn_rate <= 0.50):
        log.warning(
            "[Processing] Unusual churn rate %.2f%% — verify data integrity.",
            churn_rate * 100
        )

    null_pct = df.isnull().mean().sort_values(ascending=False).head(10)
    log.info("[Processing] Top missing-value columns:\n%s", null_pct.to_string())

    all_null = df.columns[df.isnull().all()].tolist()
    if all_null:
        log.warning("[Processing] Columns that are entirely null: %s", all_null)

    log.info(
        "[Processing] ✓ Validation passed for '%s': %d rows × %d cols",
        label, len(df), df.shape[1]
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7.  SAVE OUTPUTS                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _save_csv(df: pd.DataFrame, path: Path, label: str) -> str:
    """Save a DataFrame to CSV and log the file size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path), index=False)
    size_mb = os.path.getsize(str(path)) / 1_048_576
    log.info("[Processing] ✓ %s saved → %s  (%d rows, %.1f MB)", label, path, len(df), size_mb)
    return str(path)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8.  AIRFLOW ENTRYPOINTS                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def transform_data(**kwargs) -> dict:
    """
    Airflow ``PythonOperator`` callable — orchestrates the full processing
    and transformation stage:

    1. Load GADM administrative boundaries.
    2. Build tower GeoDataFrame from the ingested OpenCellID snapshot.
    3. Compute network KPIs at Region, Department, and Arrondissement levels.
    4. Load Expresso dataset with Dask and merge network KPIs.
    5. Compute the final integrated ``telecom_churn`` (2 M rows) DataFrame.
    6. Validate output and save to disk.
    7. Generate and save the 100 k-row sample.

    Returns
    -------
    dict
        Paths to produced files.
    """
    log.info("[Processing] ══════ PROCESSING STAGE STARTED ══════")
    t0 = time.time()

    # ── 1. GADM boundaries ──────────────────────────────────────────────────
    gdf_regions, gdf_departments, gdf_arrondissements = load_gadm_boundaries(GADM_GPKG_FILE)

    # ── 2. Tower GeoDataFrame ───────────────────────────────────────────────
    gdf_cells = build_tower_geodataframe(OPENCELLID_90D_FILE)

    # ── 3. Network KPIs ──────────────────────────────────────────────────────
    network_region_kpis, dept_kpis_per_region, arr_kpis_per_region = compute_all_regional_kpis(
        gdf_cells, gdf_regions, gdf_departments, gdf_arrondissements
    )

    # ── 4. Load & merge Expresso (full 2 M dataset) ──────────────────────────
    ddf = _load_expresso_dask(EXPRESSO_FILE)
    ddf = merge_network_kpis_dask(
        ddf, network_region_kpis, dept_kpis_per_region, arr_kpis_per_region
    )

    # ── 5. Compute final DataFrame ───────────────────────────────────────────
    telecom_churn = _compute_and_finalise(ddf)
    del ddf
    gc.collect()

    # ── 6. Validate & save full dataset ─────────────────────────────────────
    validate_processed_output(telecom_churn, "telecom_churn (2M)")
    full_path = _save_csv(telecom_churn, TELECOM_CHURN_FILE, "telecom_churn")

    # ── 7. Generate & save 100 k sample ─────────────────────────────────────
    sample_100k = telecom_churn.sample(n=TARGET_SAMPLE_SIZE, random_state=RANDOM_STATE)
    validate_processed_output(sample_100k, "telecom_churn_100k")
    sample_path = _save_csv(sample_100k, TELECOM_CHURN_100K_FILE, "telecom_churn_100k")

    elapsed = time.time() - t0
    result = {
        "telecom_churn_path":       full_path,
        "telecom_churn_100k_path":  sample_path,
    }
    log.info("[Processing] ══════ PROCESSING STAGE COMPLETE (%.1fs) ══════", elapsed)
    log.info("[Processing] Outputs: %s", result)
    return result


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9.  STANDALONE EXECUTION                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import pprint
    pprint.pprint(transform_data())
