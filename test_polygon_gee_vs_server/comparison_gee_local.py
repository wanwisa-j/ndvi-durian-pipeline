#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_ndvi_local_vs_gee.py
============================
ดึง NDVI max monthly จาก 2 source โดยใช้ logic เดียวกัน:
  - Local  : อ่านจาก Sentinel-2 .jp2 บน server โดยตรง (อิงโค้ดพี่)
  - GEE    : ดึงจาก COPERNICUS/S2_SR_HARMONIZED ผ่าน Earth Engine

Filter / masking ใช้หลักการเดียวกันทั้งคู่:
  - อ่าน SCL band โดยตรงจาก *_SCL_20m.jp2  (ไม่ใช้ omni)
  - clear mask = SCL ∈ {4, 5, 7}  (vegetation, bare soil, unclassified)
  - NDVI = (B8 - B4) / (B8 + B4)
  - Aggregate: monthly max ของ pixel ทั้งหมดในทุก polygon (province level)

Output:
  {output_dir}/ndvi_max_local.parquet   — columns: year_month, ndvi_max
  {output_dir}/ndvi_max_gee.parquet     — columns: year_month, ndvi_max
  {output_dir}/ndvi_comparison.png      — line chart เทียบ 2 source
"""

# ======================================================================================
# CONFIGURATION  —  แก้ตรงนี้เท่านั้น
# ======================================================================================

# --- paths ---
GOLDEN_DURIAN_PATH = '/fs2/wanwisa/durian/test_polygon_gee_vs_server/test_rayong.gpkg'
SENTINEL2_TILE_PATH = '/fs2/angkanap/00_MAP_TH/03_SENTINEL-2_TILES/sentinel_2_index_shapefile.shp'
S2_FOLDER_OLD = "/fs7/sentinel2/tiles"   # ปีเก่า → กลางปี 2025
S2_FOLDER_NEW = "/fs2/sentinel2/tiles"   # กลางปี 2025 → ปัจจุบัน
S2_FOLDER_CUTOFF = "2025-07-01"          # วันแรกที่ใช้ folder ใหม่ (YYYY-MM-DD)
OUTPUT_DIR = "/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare_local_gee"

# --- target ---
PROVINCE = 'rayong'
YEAR_START = 2019
YEAR_END = 2026

# --- raster params ---
RES = 10
BAND1_NAME = '08'   # NIR
BAND2_NAME = '04'   # Red
SAT = 'B'

# --- parallel ---
N_JOBS = 8

# --- GEE ---
GEE_PROJECT_ID = "project-7a0a063b-e83f-4545-a99"   # แก้ตรงนี้
GEE_SCALE = 10
GEE_BATCH_SIZE = 5_000

# ======================================================================================
# IMPORTS
# ======================================================================================

import os
import gc
import glob
import time
import warnings
import threading
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from osgeo import gdal, osr
from shapely import contains_xy
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import ee

warnings.filterwarnings("ignore")

# ======================================================================================
# SETUP
# ======================================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "compare.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ======================================================================================
# SHARED HELPERS
# ======================================================================================

def get_raster_crs(raster_path):
    ds = gdal.Open(raster_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    epsg_code = srs.GetAttrValue("AUTHORITY", 1)
    ds = None
    return int(epsg_code)


def create_points_in_polygon_fast(geom, spacing=10):
    minx, miny, maxx, maxy = geom.bounds
    x = np.arange(minx, maxx, spacing)
    y = np.arange(miny, maxy, spacing)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()
    mask = contains_xy(geom, xx, yy)
    return xx[mask], yy[mask]


def _read_band(band, xoff, yoff, xsize, ysize):
    """อ่าน raster band → numpy array ผ่าน ReadRaster()+frombuffer() ไม่พึ่ง _gdal_array"""
    type_map = {1: np.uint8, 2: np.uint16, 3: np.int16,
                4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64}
    dtype = type_map.get(band.DataType, np.float32)
    raw = band.ReadRaster(int(xoff), int(yoff), int(xsize), int(ysize), int(xsize), int(ysize), band.DataType)
    return np.frombuffer(raw, dtype=dtype).reshape(ysize, xsize).copy()


def sample_raster_batch(ds, xs, ys):
    gt = ds.GetGeoTransform()
    px = ((xs - gt[0]) / gt[1]).astype(np.int32)
    py = ((ys - gt[3]) / gt[5]).astype(np.int32)
    band = ds.GetRasterBand(1)
    xmin, xmax = px.min(), px.max()
    ymin, ymax = py.min(), py.max()
    arr = _read_band(band, xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)
    vals = arr[py - ymin, px - xmin].astype(np.float32)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        vals = np.where(vals == nodata, np.nan, vals)
    return vals


def s2_timeseries_fullpath(band_no, res, year_start, year_end, tile_name, sat):
    """
    ค้นหา scene จาก 2 folder ตาม cutoff date
    ไม่ใช้ omni — ดึง SCL band โดยตรงเหมือน GEE
    """
    tile_suffix = tile_name[0:2] + "/" + tile_name[2:3] + "/" + tile_name[3:5]
    years = set(range(year_start, year_end + 1))
    scene_list, scl_list, date_list = [], [], []

    for s2_folder in (S2_FOLDER_OLD, S2_FOLDER_NEW):
        tile_dir = os.path.join(s2_folder, tile_suffix)
        if not os.path.exists(tile_dir):
            continue

        for fn in os.listdir(tile_dir):
            parts = fn.split('_')
            if len(parts) < 3:
                continue
            year = parts[2][0:4]
            if not year.isdigit() or int(year) not in years:
                continue
            if fn[2:3] != sat:
                continue
            try:
                # SCL อยู่ที่ R20m เสมอ (native resolution ของ SCL)
                scl_glob = glob.glob(os.path.join(
                    tile_dir, fn, 'GRANULE', '*', 'IMG_DATA', 'R20m', '*_SCL_20m.jp2'))
                if len(scl_glob) != 1:
                    continue
                band_glob = glob.glob(os.path.join(
                    tile_dir, fn, 'GRANULE', '*', 'IMG_DATA',
                    f'R{res}m', f'*_B{band_no}_{res}m.jp2'))
                if len(band_glob) != 1:
                    continue

                scene_date = pd.to_datetime(
                    os.path.basename(band_glob[0])[7:15], format='%Y%m%d')

                expected_folder = (
                    S2_FOLDER_NEW if scene_date >= pd.Timestamp(S2_FOLDER_CUTOFF)
                    else S2_FOLDER_OLD
                )
                if s2_folder != expected_folder:
                    continue

                scene_list.append(band_glob[0])
                scl_list.append(scl_glob[0])
                date_list.append(os.path.basename(band_glob[0])[7:15])

            except:
                continue

    if not scene_list:
        return pd.DataFrame()

    df = pd.DataFrame({
        'date': pd.to_datetime(date_list, format='%Y%m%d'),
        'scl': scl_list,
        f'b{band_no}': scene_list,
    }).sort_values('date').reset_index(drop=True)
    df['date'] = df['date'].dt.strftime('%Y/%m/%d')
    return df

# ======================================================================================
# LOCAL PIPELINE — polygon-level NDVI extraction (identical to พี่)
# ======================================================================================

def process_polygon_local(idx, poly_row, s2_ts):
    """
    ดึง NDVI pixel-level ของ 1 polygon ตลอด timeseries
    ใช้ SCL band โดยตรง (ไม่ใช้ omni) — logic เหมือน GEE ทุกอย่าง:
      - อ่าน SCL จาก R20m .jp2
      - clear mask = SCL ∈ {4, 5, 7}  (vegetation, bare soil, unclassified)
      - NDVI = (B8 - B4) / (B8 + B4)
    SCL อยู่ที่ 20m ส่วน band อยู่ที่ 10m → sample แยก resolution แล้วใช้ mask เดียวกัน
    """
    geom = poly_row.geometry
    xs, ys = create_points_in_polygon_fast(geom, spacing=10)
    if len(xs) == 0:
        return None

    # grid สำหรับ SCL (20m) — ใช้ spacing=20 เพื่อ match resolution
    xs_scl, ys_scl = create_points_in_polygon_fast(geom, spacing=20)
    if len(xs_scl) == 0:
        return None

    records = []

    for ts_row in s2_ts.itertuples(index=False):
        d_col = ts_row.date
        try:
            scl_ds = gdal.Open(ts_row.scl)
            b1_ds  = gdal.Open(getattr(ts_row, f'b{BAND1_NAME}'))
            b2_ds  = gdal.Open(getattr(ts_row, f'b{BAND2_NAME}'))

            # sample SCL ที่ 20m grid
            scl_vals = sample_raster_batch(scl_ds, xs_scl, ys_scl).astype(np.int32)

            # clear mask: SCL ∈ {4, 5, 7} — เหมือน GEE เป๊ะ
            clear_mask_scl = np.isin(scl_vals, [4, 5, 7])

            # ถ้าไม่มี clear pixel เลย ข้ามทั้ง scene
            if not clear_mask_scl.any():
                scl_ds = b1_ds = b2_ds = None
                continue

            # sample B8, B4 ที่ 10m grid (xs, ys)
            # แต่ mask มาจาก SCL 20m → nearest-neighbor map 20m → 10m
            # วิธี: สร้าง clear_mask บน 10m grid จาก SCL grid โดย snap แต่ละ 10m point
            # ไปหา SCL cell ที่ใกล้ที่สุด (floor division)
            gt_scl = scl_ds.GetGeoTransform()
            px_scl = np.floor((xs - gt_scl[0]) / gt_scl[1]).astype(np.int32)
            py_scl = np.floor((ys - gt_scl[3]) / gt_scl[5]).astype(np.int32)

            xmin_s = px_scl.min(); ymin_s = py_scl.min()
            xmax_s = px_scl.max(); ymax_s = py_scl.max()
            w_s = xmax_s - xmin_s + 1; h_s = ymax_s - ymin_s + 1

            scl_arr = _read_band(scl_ds.GetRasterBand(1), int(xmin_s), int(ymin_s), int(w_s), int(h_s))
            scl_at_10m = scl_arr[py_scl - ymin_s, px_scl - xmin_s].astype(np.int32)

            # clear mask บน 10m grid
            clear_mask = np.isin(scl_at_10m, [4, 5, 7])

            vn = np.full(len(xs), np.nan, dtype=np.float32)

            if clear_mask.any():
                tmp1 = sample_raster_batch(b1_ds, xs[clear_mask], ys[clear_mask])
                tmp2 = sample_raster_batch(b2_ds, xs[clear_mask], ys[clear_mask])
                denom = tmp1 + tmp2
                valid = denom != 0
                vn_tmp = np.full(len(tmp1), np.nan, dtype=np.float32)
                vn_tmp[valid] = (tmp1[valid] - tmp2[valid]) / denom[valid]
                vn[clear_mask] = vn_tmp

            records.append({'date': d_col, 'ndvi': vn})

            scl_ds = b1_ds = b2_ds = None

        except Exception as e:
            log.warning(f"  polygon {idx} date {d_col}: {e}")
            continue

    return records


def run_local_pipeline(tile_polygons, s2_ts, tile):
    """
    รัน parallel extraction สำหรับทุก polygon ใน tile
    return: DataFrame ที่มี columns [date, ndvi_max] ระดับ tile
    """
    log.info(f"  [local] tile {tile}: {len(tile_polygons)} polygons")

    all_results = Parallel(n_jobs=N_JOBS)(
        delayed(process_polygon_local)(idx, row, s2_ts)
        for idx, row in tile_polygons.iterrows()
    )

    # รวม ndvi ทุก pixel ทุก polygon ต่อวันที่
    date_ndvi = {}
    for result in all_results:
        if result is None:
            continue
        for rec in result:
            d = rec['date']
            if d not in date_ndvi:
                date_ndvi[d] = []
            vals = rec['ndvi']
            date_ndvi[d].append(vals[~np.isnan(vals)])

    rows = []
    for d, val_list in date_ndvi.items():
        if val_list:
            all_vals = np.concatenate(val_list)
            if len(all_vals) > 0:
                rows.append({'date': pd.to_datetime(d, format='%Y/%m/%d'),
                             'ndvi_max': float(np.nanmax(all_vals))})

    if not rows:
        return pd.DataFrame(columns=['date', 'ndvi_max'])

    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def extract_ndvi_max_local(tile_polygons_dict, s2_ts_dict):
    """
    วนทุก tile → รวม daily ndvi_max → aggregate monthly max
    return: DataFrame [year_month, ndvi_max]
    """
    daily_frames = []

    for tile, polygons in tile_polygons_dict.items():
        if tile not in s2_ts_dict:
            continue
        df = run_local_pipeline(polygons, s2_ts_dict[tile], tile)
        if not df.empty:
            daily_frames.append(df)

    if not daily_frames:
        return pd.DataFrame(columns=['year_month', 'ndvi_max'])

    daily_all = pd.concat(daily_frames, ignore_index=True)

    # monthly max
    daily_all['year_month'] = daily_all['date'].dt.to_period('M')
    monthly = (
        daily_all.groupby('year_month')['ndvi_max']
        .max()
        .reset_index()
    )
    monthly['year_month'] = monthly['year_month'].dt.to_timestamp()
    return monthly.sort_values('year_month').reset_index(drop=True)

# ======================================================================================
# GEE PIPELINE — identical SCL filter & NDVI formula
# ======================================================================================

_gee_semaphore = threading.Semaphore(6)

MAX_RETRIES   = 5
RETRY_BACKOFF = 5.0
DOWNLOAD_TIMEOUT = 300
RETRYABLE_HTTP_CODES = {500, 502, 503, 429}


def mask_scl_and_ndvi_gee(img):
    """
    ใช้ SCL band filter clear pixels (vegetation=4, bare=5, unclassified=7)
    NDVI = (B8 - B4) / (B8 + B4)  — สูตรเดียวกับ local
    """
    scl = img.select("SCL")
    valid_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(7))
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
    return ndvi.updateMask(valid_mask)


def build_ee_fc(batch_df):
    lon_idx = batch_df.columns.get_loc("lon")
    lat_idx = batch_df.columns.get_loc("lat")
    features = []
    for row in batch_df.itertuples(index=False):
        features.append(ee.Feature(
            ee.Geometry.Point([float(row[lon_idx]), float(row[lat_idx])]),
            {}
        ))
    return ee.FeatureCollection(features)


def get_monthly_ndvi_image_gee(year, month, aoi):
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, "month")
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(aoi)
        .map(mask_scl_and_ndvi_gee)
    )
    count = col.size().getInfo()
    if count == 0:
        return None
    log.info(f"  [gee] {year}-{month:02d}: {count} scenes")
    return col.max().rename("ndvi_max")


def extract_batch_gee(img, batch_df, ym, batch_id):
    """1 GEE reduceRegions call → list of ndvi values"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with _gee_semaphore:
                fc = img.reduceRegions(
                    collection=build_ee_fc(batch_df),
                    reducer=ee.Reducer.first(),
                    scale=GEE_SCALE,
                )
                url = fc.getDownloadURL(filetype="CSV")
                resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))

            drop_cols = [c for c in df.columns if c in [".geo", "system:index"]]
            df = df.drop(columns=drop_cols, errors="ignore")

            col = next((c for c in df.columns if "first" in c or c == "ndvi_max"), None)
            if col is None:
                return np.array([])
            vals = df[col].dropna().values.astype(np.float32)
            return vals

        except Exception as e:
            if attempt == MAX_RETRIES:
                log.error(f"  [gee] batch{batch_id} {ym} FAILED: {e}")
                return np.array([])
            delay = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning(f"  [gee] batch{batch_id} {ym} retry {attempt}: {e} — wait {delay}s")
            time.sleep(delay)


def extract_ndvi_max_gee(points_wgs84):
    """
    points_wgs84: GeoDataFrame (หรือ DataFrame) ที่มี lon, lat columns
    return: DataFrame [year_month, ndvi_max]
    """
    lon_min, lat_min = points_wgs84["lon"].min(), points_wgs84["lat"].min()
    lon_max, lat_max = points_wgs84["lon"].max(), points_wgs84["lat"].max()
    aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    rows = []

    for year in range(YEAR_START, YEAR_END + 1):
        for month in range(1, 13):
            ym = f"{year}-{month:02d}"
            log.info(f"  [gee] processing {ym}")

            img = get_monthly_ndvi_image_gee(year, month, aoi)
            if img is None:
                log.info(f"  [gee] {ym}: no scenes — skip")
                continue

            # batch ทุก point
            n = len(points_wgs84)
            all_vals = []

            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {}
                for bid in range(0, n, GEE_BATCH_SIZE):
                    batch = points_wgs84.iloc[bid: bid + GEE_BATCH_SIZE]
                    fut = executor.submit(extract_batch_gee, img, batch, ym, bid)
                    futures[fut] = bid

                for fut in as_completed(futures):
                    vals = fut.result()
                    if vals is not None and len(vals) > 0:
                        all_vals.append(vals)

            if all_vals:
                combined = np.concatenate(all_vals)
                combined = combined[~np.isnan(combined)]
                if len(combined) > 0:
                    rows.append({
                        'year_month': pd.Timestamp(f"{year}-{month:02d}-01"),
                        'ndvi_max': float(np.max(combined))
                    })

    if not rows:
        return pd.DataFrame(columns=['year_month', 'ndvi_max'])

    return pd.DataFrame(rows).sort_values('year_month').reset_index(drop=True)

# ======================================================================================
# PLOT
# ======================================================================================

def plot_comparison(local_df, gee_df, output_dir):
    fig, ax = plt.subplots(figsize=(14, 5))

    if not local_df.empty:
        ax.plot(
            local_df['year_month'], local_df['ndvi_max'],
            color='#1D9E75', linewidth=1.8, marker='o', markersize=3,
            label='Local S2 (server)'
        )

    if not gee_df.empty:
        ax.plot(
            gee_df['year_month'], gee_df['ndvi_max'],
            color='#378ADD', linewidth=1.8, marker='s', markersize=3,
            linestyle='--', label='GEE (COPERNICUS/S2_SR_HARMONIZED)'
        )

    ax.set_title(f'NDVI Max Monthly — {PROVINCE.capitalize()} ({YEAR_START}–{YEAR_END})',
                 fontsize=13, fontweight='normal')
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('NDVI Max', fontsize=11)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.grid(axis='x', linestyle=':', alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()

    png_path = os.path.join(output_dir, "ndvi_comparison.png")
    jpg_path = os.path.join(output_dir, "ndvi_comparison.jpg")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(jpg_path, dpi=150, bbox_inches='tight', quality=95)
    plt.close()
    log.info(f"  Plot saved → {png_path}")
    log.info(f"  Plot saved → {jpg_path}")

# ======================================================================================
# MAIN
# ======================================================================================

def main():
    start_time = time.time()
    log.info("=" * 60)
    log.info(f"PROVINCE: {PROVINCE.upper()}  |  {YEAR_START}–{YEAR_END}")
    log.info("=" * 60)

    # --------------------------------------------------------------------------
    # LOAD INPUT DATA
    # --------------------------------------------------------------------------
    log.info("Loading polygons...")
    ext = os.path.splitext(GOLDEN_DURIAN_PATH)[1].lower()
    if ext == '.gpkg':
        golden_durian = gpd.read_file(GOLDEN_DURIAN_PATH).to_crs("EPSG:4326")
    else:
        golden_durian = gpd.read_parquet(GOLDEN_DURIAN_PATH).to_crs("EPSG:4326")

    log.info("Loading Sentinel-2 tile grid...")
    sentinel2_tile = gpd.read_file(SENTINEL2_TILE_PATH).rename(columns={"Name": "tile"})

    log.info("Spatial join...")
    joined = gpd.sjoin(golden_durian, sentinel2_tile, how="inner", predicate="intersects")
    tiles = joined['tile'].unique().tolist()
    log.info(f"  Tiles found: {tiles}")

    # --------------------------------------------------------------------------
    # PREPARE S2 TIMESERIES PATHS (สำหรับ local)
    # --------------------------------------------------------------------------
    tile_polygons_dict = {}
    s2_ts_dict = {}

    for tile in tiles:
        log.info(f"  Scanning timeseries for tile {tile}...")
        t0 = time.time()
        s2_b1 = s2_timeseries_fullpath(BAND1_NAME, RES, YEAR_START, YEAR_END, tile, SAT)
        s2_b2 = s2_timeseries_fullpath(BAND2_NAME, RES, YEAR_START, YEAR_END, tile, SAT)
        log.info(f"  Scan done in {time.time()-t0:.1f}s  b1={len(s2_b1)} b2={len(s2_b2)} scenes")

        if s2_b1.empty or s2_b2.empty:
            log.warning(f"  No imagery for tile {tile} — skip")
            continue

        s2_ts = pd.merge(s2_b1, s2_b2, on=['date', 'scl'], how='inner')
        if s2_ts.empty:
            continue

        epsg = get_raster_crs(s2_ts[f'b{BAND1_NAME}'].iloc[0])
        polygons_proj = joined[joined['tile'] == tile].to_crs(epsg=epsg)

        tile_polygons_dict[tile] = polygons_proj
        s2_ts_dict[tile] = s2_ts

    # --------------------------------------------------------------------------
    # PREPARE POINTS FOR GEE (centroid ของแต่ละ polygon — เบากว่า pixel grid)
    # --------------------------------------------------------------------------
    log.info("Preparing points for GEE...")
    centroids = golden_durian.copy()
    centroids['lon'] = golden_durian.geometry.centroid.x
    centroids['lat'] = golden_durian.geometry.centroid.y
    points_wgs84 = centroids[['lon', 'lat']].dropna().reset_index(drop=True)
    log.info(f"  GEE points: {len(points_wgs84):,}")

    # --------------------------------------------------------------------------
    # RUN LOCAL
    # --------------------------------------------------------------------------
    local_out = os.path.join(OUTPUT_DIR, "ndvi_max_local.parquet")
    if os.path.exists(local_out):
        log.info("ndvi_max_local.parquet already exists — loading...")
        local_df = pd.read_parquet(local_out)
    else:
        log.info("\n--- Extracting LOCAL S2 ---")
        local_df = extract_ndvi_max_local(tile_polygons_dict, s2_ts_dict)
        local_df.to_parquet(local_out, index=False)
        log.info(f"  Saved → {local_out}  shape: {local_df.shape}")

    # --------------------------------------------------------------------------
    # RUN GEE
    # --------------------------------------------------------------------------
    gee_out = os.path.join(OUTPUT_DIR, "ndvi_max_gee.parquet")
    if os.path.exists(gee_out):
        log.info("ndvi_max_gee.parquet already exists — loading...")
        gee_df = pd.read_parquet(gee_out)
    else:
        log.info("\n--- Extracting GEE ---")
        ee.Initialize(project=GEE_PROJECT_ID)
        gee_df = extract_ndvi_max_gee(points_wgs84)
        gee_df.to_parquet(gee_out, index=False)
        log.info(f"  Saved → {gee_out}  shape: {gee_df.shape}")

    # --------------------------------------------------------------------------
    # PLOT
    # --------------------------------------------------------------------------
    log.info("\n--- Plotting ---")
    plot_comparison(local_df, gee_df, OUTPUT_DIR)

    duration = (time.time() - start_time) / 60
    log.info(f"\nDONE  |  Total: {duration:.2f} mins")
    log.info(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()