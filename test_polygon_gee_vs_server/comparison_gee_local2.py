#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_ndvi.py
===============
ดึง NDVI pixel-level (10m grid ใน polygon) จาก 2 source:
  - Local  : อ่านจาก Sentinel-2 .jp2 บน server (windowed read + baseline correction)
  - GEE    : ดึงจาก COPERNICUS/S2_SR_HARMONIZED เฉพาะวันที่มีใน local

วันที่อ้างอิง: local เป็นหลัก → GEE ดึงเฉพาะวันที่ตรงกัน

Output:
  {OUTPUT_DIR}/ndvi_local.parquet   — columns: plot_id, point_id, <YYYY/MM/DD> ...
  {OUTPUT_DIR}/ndvi_gee.parquet     — columns: plot_id, point_id, <YYYY/MM/DD> ...
  {OUTPUT_DIR}/points_meta.parquet  — columns: plot_id, point_id, x_proj, y_proj
"""

# ======================================================================================
# CONFIGURATION
# ======================================================================================

GOLDEN_DURIAN_PATH  = '/fs2/wanwisa/durian/test_polygon_gee_vs_server/test_rayong.gpkg'
SENTINEL2_TILE_PATH = '/fs2/angkanap/00_MAP_TH/03_SENTINEL-2_TILES/sentinel_2_index_shapefile.shp'
S2_FOLDER_OLD       = "/fs7/sentinel2/tiles"
S2_FOLDER_NEW       = "/fs2/sentinel2/tiles"
S2_FOLDER_CUTOFF    = "2025-07-01"
OUTPUT_DIR          = "/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare"

PROVINCE   = 'rayong'
YEAR_START = 2019
YEAR_END   = 2026

RES      = 10
BAND_NIR = '08'   # B08
BAND_RED = '04'   # B04
SAT      = 'B'

# Parallel workers — None = auto-select จาก RAM ที่มี
N_JOBS   = None

# Memory guard
MAX_RAM_GB         = 60.0   # hard budget ของ process tree ทั้งหมด (GB)
MEM_SOFT_RATIO     = 0.90   # submit worker ใหม่ได้เมื่อ usage < MAX_RAM_GB × ratio
MEM_POLL_SEC       = 3.0
MEM_MAX_WAIT_SEC   = 300.0
MIN_FREE_RAM_GB    = 8.0    # system free RAM ต่ำสุดก่อน submit
EST_RAM_PER_DATE_GB = 0.5   # ประมาณ RAM ต่อ 1 date-worker (ปรับตามขนาด polygon)

GEE_PROJECT_ID = "project-7a0a063b-e83f-4545-a99"
GEE_SCALE      = 10
GEE_BATCH_SIZE = 5_000

# SCL clear classes (vegetation=4, bare=5, unclassified=7)
SCL_CLEAR = [4, 5, 7]

from datetime import date as _date
BASELINE_CUTOFF = _date(2022, 1, 25)

# ======================================================================================
# IMPORTS
# ======================================================================================

import os, gc, glob, time, warnings, threading, io, logging, resource
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import ee

from osgeo import gdal, osr
from shapely import contains_xy

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False

warnings.filterwarnings("ignore")

# ======================================================================================
# SETUP
# ======================================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
gdal.UseExceptions()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "collect.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ======================================================================================
# MEMORY HELPERS  (ported from code4)
# ======================================================================================

def _get_process_tree_gb(metric="auto"):
    """PSS > USS > RSS ของ process tree ทั้งหมด (GB)"""
    if not _HAS_PSUTIL:
        return 0.0, "unavailable"
    order = ["pss", "uss", "rss"] if metric == "auto" else [metric, "uss", "rss"]
    seen = set()
    order = [x for x in order if not (x in seen or seen.add(x))]
    try:
        proc  = _psutil.Process()
        procs = [proc] + proc.children(recursive=True)
        for m in order:
            try:
                total = 0
                for p in procs:
                    try:
                        if m in ("pss", "uss"):
                            total += getattr(p.memory_full_info(), m, 0) or 0
                        else:
                            total += p.memory_info().rss
                    except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                        pass
                return total / (1024 ** 3), m
            except (AttributeError, PermissionError, OSError):
                continue
    except Exception:
        pass
    return 0.0, "error"


def _get_sys_mem_gb():
    """(total, available, used) GB"""
    if not _HAS_PSUTIL:
        return 0.0, 0.0, 0.0
    try:
        vm = _psutil.virtual_memory()
        g  = 1.0 / (1024 ** 3)
        return vm.total * g, vm.available * g, vm.used * g
    except Exception:
        return 0.0, 0.0, 0.0


def cleanup_memory():
    """gc + malloc_trim(0) — คืน arena กลับ OS ทันที"""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def auto_select_workers(max_ram_gb, est_per_worker_gb, soft_ratio=0.90,
                         min_free_gb=8.0):
    """คำนวณ worker count ที่ปลอดภัยจาก RAM จริงที่มี"""
    cpu_cap = min(os.cpu_count() or 1, 16)
    if not _HAS_PSUTIL:
        log.info("[CONFIG] psutil ไม่มี → workers=1")
        return 1
    _, sys_avail, _ = _get_sys_mem_gb()
    budget = max(0.0, min(sys_avail - min_free_gb, max_ram_gb * soft_ratio))
    workers = max(1, min(cpu_cap, int(budget / est_per_worker_gb)))
    log.info(
        f"[CONFIG] auto workers={workers}  cpu_cap={cpu_cap}  "
        f"sys_avail={sys_avail:.1f} GB  budget={budget:.1f} GB  "
        f"est_per_worker={est_per_worker_gb:.1f} GB"
    )
    return workers


def wait_for_memory(max_ram_gb, soft_ratio=MEM_SOFT_RATIO,
                     poll_sec=MEM_POLL_SEC, max_wait_sec=MEM_MAX_WAIT_SEC,
                     min_free_gb=MIN_FREE_RAM_GB):
    """Block จนกว่า memory จะต่ำกว่า threshold — ป้องกัน OOM ก่อน submit worker"""
    if not _HAS_PSUTIL:
        return
    threshold = max_ram_gb * soft_ratio
    waited    = 0.0
    while True:
        _, sys_avail, sys_used = _get_sys_mem_gb()
        tree_gb, metric = _get_process_tree_gb()

        # RSS อาจ inflate จาก CoW — ถ้า sys_avail ยังโอเคให้ผ่าน
        if metric == "rss" and tree_gb >= threshold and sys_avail >= min_free_gb:
            log.debug(f"[MEM] rss={tree_gb:.1f} GB inflated (CoW) sys_avail={sys_avail:.1f} GB OK")
            return

        over_tree   = tree_gb >= threshold and metric in ("pss", "uss")
        system_tight = sys_avail > 0 and sys_avail < min_free_gb

        if not over_tree and not system_tight:
            return

        if waited >= max_wait_sec:
            if sys_avail >= min_free_gb:
                log.warning(
                    f"[MEM] max_wait={max_wait_sec}s reached — "
                    f"sys_avail={sys_avail:.1f} GB OK, continuing"
                )
                return
            log.warning(
                f"[MEM] max_wait={max_wait_sec}s reached — "
                f"sys_avail={sys_avail:.1f} GB still tight, keep waiting"
            )

        log.info(
            f"[MEM] tree_{metric}={tree_gb:.1f}/{max_ram_gb} GB  "
            f"sys_avail={sys_avail:.1f} GB  waited={waited:.0f}s — holding"
        )
        time.sleep(poll_sec)
        waited += poll_sec

# ======================================================================================
# HELPERS — raster
# ======================================================================================

def get_raster_crs(raster_path: str) -> int:
    ds  = gdal.Open(raster_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    code = srs.GetAttrValue("AUTHORITY", 1)
    ds   = None
    return int(code)


def _read_band_windowed(band, geom_bounds, geotransform):
    """อ่านเฉพาะ window ที่ครอบ geometry — ไม่โหลดทั้ง tile"""
    xmin, ymin, xmax, ymax = geom_bounds
    gt                      = geotransform
    xOrigin, pixelW, _, yOrigin, _, pixelH = gt

    xoff  = max(int((xmin - xOrigin) / pixelW), 0)
    yoff  = max(int((yOrigin - ymax) / abs(pixelH)), 0)
    xsize = min(int((xmax - xmin) / pixelW) + 2, band.XSize - xoff)
    ysize = min(int((ymax - ymin) / abs(pixelH)) + 2, band.YSize - yoff)

    type_map = {1: np.uint8, 2: np.uint16, 3: np.int16,
                4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64}
    dtype = type_map.get(band.DataType, np.float32)
    raw   = band.ReadRaster(xoff, yoff, xsize, ysize, xsize, ysize, band.DataType)
    arr   = np.frombuffer(raw, dtype=dtype).reshape(ysize, xsize).copy()
    return arr, xoff, yoff


def sample_window(ds, xs, ys):
    """Sample pixel values จาก dataset ที่ตำแหน่ง xs, ys — windowed read"""
    gt   = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)

    arr, xoff, yoff = _read_band_windowed(
        band, (xs.min(), ys.min(), xs.max(), ys.max()), gt
    )
    px = ((xs - gt[0]) / gt[1]).astype(np.int32) - xoff
    py = ((ys - gt[3]) / gt[5]).astype(np.int32) - yoff
    px = np.clip(px, 0, arr.shape[1] - 1)
    py = np.clip(py, 0, arr.shape[0] - 1)

    vals   = arr[py, px].astype(np.float32)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        vals[vals == nodata] = np.nan
    return vals


def create_grid_in_polygon(geom, spacing=10):
    """สร้าง point grid ภายใน polygon"""
    minx, miny, maxx, maxy = geom.bounds
    x  = np.arange(minx + spacing / 2, maxx, spacing)
    y  = np.arange(miny + spacing / 2, maxy, spacing)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()
    mask   = contains_xy(geom, xx, yy)
    return xx[mask], yy[mask]


def apply_baseline(arr: np.ndarray, acq_date) -> np.ndarray:
    """ถ้า acq_date > 2022-01-25 → DN - 1000 (offset correction)"""
    if isinstance(acq_date, str):
        acq_date = datetime.strptime(acq_date, '%Y/%m/%d').date()
    return arr - 1000.0 if acq_date > BASELINE_CUTOFF else arr.copy()


def calc_ndvi(b08: np.ndarray, b04: np.ndarray) -> np.ndarray:
    denom = b08 + b04
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(denom != 0, (b08 - b04) / denom, np.nan)
    return ndvi.astype(np.float32)

# ======================================================================================
# HELPERS — Sentinel-2 file discovery
# ======================================================================================

def s2_timeseries_fullpath(band_no, res, year_start, year_end, tile_name, sat):
    tile_suffix = tile_name[0:2] + "/" + tile_name[2:3] + "/" + tile_name[3:5]
    years       = set(range(year_start, year_end + 1))
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
                expected_folder = (S2_FOLDER_NEW if scene_date >= pd.Timestamp(S2_FOLDER_CUTOFF)
                                   else S2_FOLDER_OLD)
                if s2_folder != expected_folder:
                    continue

                scene_list.append(band_glob[0])
                scl_list.append(scl_glob[0])
                date_list.append(os.path.basename(band_glob[0])[7:15])
            except Exception:
                continue

    if not scene_list:
        return pd.DataFrame()

    df = pd.DataFrame({
        'date': pd.to_datetime(date_list, format='%Y%m%d'),
        'scl':  scl_list,
        f'b{band_no}': scene_list,
    }).sort_values('date').reset_index(drop=True)
    df['date_str'] = df['date'].dt.strftime('%Y/%m/%d')
    return df

# ======================================================================================
# LOCAL PIPELINE
# worker function ต้องอยู่ระดับ module สำหรับ ProcessPoolExecutor (pickle)
# ======================================================================================

def _worker_init_fn():
    """Worker init: suppress GDAL warning — ไม่ใช้ RLIMIT (ให้ wait_for_memory จัดการแทน)"""
    from osgeo import gdal as _g
    _g.UseExceptions()


def _process_one_date(args):
    """
    1 worker process = 1 (polygon, date) pair
    args = (plot_id, xs, ys, scl_path, b08_path, b04_path, date_str, acq_date_iso)
    คืน (date_str, ndvi_arr float32) หรือ (date_str, None)
    """
    plot_id, xs, ys, scl_path, b08_path, b04_path, date_str, acq_date_iso = args
    from osgeo import gdal as _g
    from datetime import date as _d
    import numpy as _np

    acq_date = _d.fromisoformat(acq_date_iso)
    n_pts    = len(xs)
    ndvi_arr = _np.full(n_pts, _np.nan, dtype=_np.float32)
    scl_ds   = b08_ds = b04_ds = None
    try:
        scl_ds = _g.Open(scl_path)
        b08_ds = _g.Open(b08_path)
        b04_ds = _g.Open(b04_path)

        gt_scl = scl_ds.GetGeoTransform()
        px_scl = _np.floor((xs - gt_scl[0]) / gt_scl[1]).astype(_np.int32)
        py_scl = _np.floor((ys - gt_scl[3]) / gt_scl[5]).astype(_np.int32)
        x_scl  = gt_scl[0] + px_scl * gt_scl[1]
        y_scl  = gt_scl[3] + py_scl * gt_scl[5]

        # ใช้ sample_window จาก module-level (pickle-safe เพราะ top-level function)
        scl_vals = sample_window(scl_ds, x_scl, y_scl).astype(_np.int32)
        clear    = _np.isin(scl_vals, SCL_CLEAR)

        if clear.any():
            b08_raw  = sample_window(b08_ds, xs[clear], ys[clear])
            b04_raw  = sample_window(b04_ds, xs[clear], ys[clear])
            b08_corr = apply_baseline(b08_raw, acq_date)
            b04_corr = apply_baseline(b04_raw, acq_date)
            ndvi_arr[clear] = calc_ndvi(b08_corr, b04_corr)

        return date_str, ndvi_arr

    except Exception:
        return date_str, None
    finally:
        scl_ds = b08_ds = b04_ds = None
        import gc as _gc; _gc.collect()


def extract_local_one_polygon(plot_id, poly_geom, s2_ts, n_workers):
    """
    ProcessPoolExecutor + sliding-window submission (เหมือน code4)
    submit worker ใหม่ทันทีที่ worker เก่าเสร็จ + wait_for_memory ก่อน submit
    คืน (date_ndvi_dict, xs, ys)
    """
    xs, ys = create_grid_in_polygon(poly_geom, spacing=10)
    if len(xs) == 0:
        return {}, np.array([]), np.array([])

    args_list = []
    for row in s2_ts.itertuples(index=False):
        acq_date = row.date.date() if hasattr(row.date, 'date') else \
                   datetime.strptime(row.date_str, '%Y/%m/%d').date()
        args_list.append((
            plot_id, xs, ys,
            row.scl,
            getattr(row, f'b{BAND_NIR}'),
            getattr(row, f'b{BAND_RED}'),
            row.date_str,
            acq_date.isoformat(),
        ))

    date_ndvi  = {}
    n_total    = len(args_list)
    arg_iter   = iter(args_list)
    exhausted  = False

    def _submit_one(pool, a):
        wait_for_memory(MAX_RAM_GB)   # block ถ้า memory ตึง
        return pool.submit(_process_one_date, a)

    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_worker_init_fn) as pool:
        pending  = {}

        # seed pool
        while len(pending) < n_workers and not exhausted:
            try:
                a  = next(arg_iter)
                f  = _submit_one(pool, a)
                pending[f] = a[6]  # date_str
            except StopIteration:
                exhausted = True

        # sliding window drain
        while pending:
            for future in as_completed(pending):
                d_str = pending.pop(future)
                try:
                    _, ndvi_arr = future.result()
                    if ndvi_arr is not None:
                        date_ndvi[d_str] = ndvi_arr
                    else:
                        log.warning(f"  plot {plot_id} date {d_str}: worker error")
                except Exception as e:
                    log.warning(f"  plot {plot_id} date {d_str}: {e}")

                cleanup_memory()

                _, sys_avail, sys_used = _get_sys_mem_gb()
                tree_gb, metric = _get_process_tree_gb()
                log.debug(
                    f"  [MEM] plot={plot_id} date={d_str} done  "
                    f"tree_{metric}={tree_gb:.1f} GB  sys_avail={sys_avail:.1f} GB  "
                    f"dates_done={len(date_ndvi)}/{n_total}"
                )

                # submit next
                if not exhausted:
                    try:
                        a = next(arg_iter)
                        f = _submit_one(pool, a)
                        pending[f] = a[6]
                    except StopIteration:
                        exhausted = True
                break  # re-enter as_completed loop

    return date_ndvi, xs, ys


def run_local_all_polygons(polygons_gdf, s2_ts, n_workers):
    """วน polygon sequentially, ภายในแต่ละ polygon parallel ระดับ date"""
    all_records = []
    meta_rows   = []

    for plot_id, row in enumerate(polygons_gdf.itertuples()):
        log.info(
            f"  [local] polygon {plot_id} — "
            f"{len(s2_ts)} dates × {n_workers} workers"
        )
        date_ndvi, xs, ys = extract_local_one_polygon(
            plot_id, row.geometry, s2_ts, n_workers
        )

        if len(xs) == 0:
            log.warning(f"  plot {plot_id}: no grid points in polygon")
            continue

        for pt_id in range(len(xs)):
            rec = {'plot_id': plot_id, 'point_id': pt_id}
            for d, arr in date_ndvi.items():
                val    = arr[pt_id]
                rec[d] = float(val) if np.isfinite(val) else np.nan
            all_records.append(rec)
            meta_rows.append({
                'plot_id':  plot_id,
                'point_id': pt_id,
                'x_proj':   float(xs[pt_id]),
                'y_proj':   float(ys[pt_id]),
            })

        log.info(
            f"  [local] polygon {plot_id}: "
            f"{len(xs)} pts × {len(date_ndvi)} dates"
        )
        cleanup_memory()

    return all_records, meta_rows

# ======================================================================================
# GEE PIPELINE — ดึงเฉพาะวันที่มีใน local
# ======================================================================================

_gee_semaphore  = threading.Semaphore(6)
MAX_RETRIES     = 5
RETRY_BACKOFF   = 5.0
DOWNLOAD_TIMEOUT = 300


def mask_ndvi_gee(img):
    scl  = img.select("SCL")
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(7))
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
    return ndvi.updateMask(mask).set('system:time_start', img.get('system:time_start'))


def get_gee_image_for_date(target_date_str: str, aoi):
    d     = pd.Timestamp(target_date_str)
    start = ee.Date(d.strftime('%Y-%m-%d'))
    end   = start.advance(1, 'day')
    col   = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(aoi)
        .map(mask_ndvi_gee)
    )
    if col.size().getInfo() == 0:
        return None
    return col.mosaic().rename("ndvi")


def extract_gee_batch(img, batch_df, date_str, batch_id):
    lon_idx = batch_df.columns.get_loc("lon")
    lat_idx = batch_df.columns.get_loc("lat")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with _gee_semaphore:
                features = [
                    ee.Feature(
                        ee.Geometry.Point([float(r[lon_idx]), float(r[lat_idx])]), {}
                    )
                    for r in batch_df.itertuples(index=False)
                ]
                fc  = img.reduceRegions(
                    collection=ee.FeatureCollection(features),
                    reducer=ee.Reducer.first(),
                    scale=GEE_SCALE,
                )
                url  = fc.getDownloadURL(filetype="CSV")
                resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                df_out = pd.read_csv(io.StringIO(resp.text))

            drop_cols = [c for c in df_out.columns if c in [".geo", "system:index"]]
            df_out    = df_out.drop(columns=drop_cols, errors="ignore")
            col_name  = next(
                (c for c in df_out.columns if "first" in c or c == "ndvi"), None
            )
            if col_name is None:
                return np.full(len(batch_df), np.nan, dtype=np.float32)
            return df_out[col_name].values.astype(np.float32)

        except Exception as e:
            if attempt == MAX_RETRIES:
                log.error(f"  [gee] batch{batch_id} {date_str} FAILED: {e}")
                return np.full(len(batch_df), np.nan, dtype=np.float32)
            delay = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning(f"  [gee] retry {attempt} {date_str}: {e} — wait {delay}s")
            time.sleep(delay)


def run_gee_for_dates(points_wgs84: pd.DataFrame, date_list: list):
    lon_min, lat_min = points_wgs84["lon"].min(), points_wgs84["lat"].min()
    lon_max, lat_max = points_wgs84["lon"].max(), points_wgs84["lat"].max()
    aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    date_ndvi = {}
    n         = len(points_wgs84)

    for date_str in date_list:
        log.info(f"  [gee] {date_str}")
        img = get_gee_image_for_date(date_str, aoi)
        if img is None:
            log.info(f"  [gee] {date_str}: no image — fill NaN")
            date_ndvi[date_str] = np.full(n, np.nan, dtype=np.float32)
            continue

        all_vals = np.full(n, np.nan, dtype=np.float32)
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            for bid in range(0, n, GEE_BATCH_SIZE):
                batch      = points_wgs84.iloc[bid: bid + GEE_BATCH_SIZE]
                fut        = executor.submit(extract_gee_batch, img, batch, date_str, bid)
                futures[fut] = bid
            for fut in as_completed(futures):
                bid  = futures[fut]
                vals = fut.result()
                end  = min(bid + GEE_BATCH_SIZE, n)
                all_vals[bid:end] = vals[: end - bid]

        date_ndvi[date_str] = all_vals
        log.info(
            f"  [gee] {date_str}: valid={int(np.isfinite(all_vals).sum())}/{n}"
        )

    return date_ndvi

# ======================================================================================
# MAIN
# ======================================================================================

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info(f"PROVINCE: {PROVINCE.upper()}  |  {YEAR_START}–{YEAR_END}")
    log.info("=" * 60)

    local_out = os.path.join(OUTPUT_DIR, "ndvi_local.parquet")
    gee_out   = os.path.join(OUTPUT_DIR, "ndvi_gee.parquet")
    meta_out  = os.path.join(OUTPUT_DIR, "points_meta.parquet")

    if all(os.path.exists(p) for p in [local_out, gee_out, meta_out]):
        log.info("All outputs already exist → skip. Delete files to re-run.")
        return

    # ---- auto workers ----
    n_workers = N_JOBS or auto_select_workers(
        MAX_RAM_GB, EST_RAM_PER_DATE_GB, MEM_SOFT_RATIO, MIN_FREE_RAM_GB
    )
    log.info(f"[CONFIG] n_workers={n_workers}  max_ram_gb={MAX_RAM_GB}")

    # ---- load polygons ----
    log.info("Loading polygons...")
    ext      = os.path.splitext(GOLDEN_DURIAN_PATH)[1].lower()
    polygons = (gpd.read_file(GOLDEN_DURIAN_PATH) if ext == '.gpkg'
                else gpd.read_parquet(GOLDEN_DURIAN_PATH))
    polygons = polygons.reset_index(drop=True)
    log.info(f"  {len(polygons)} polygons loaded")

    # ---- spatial join → tiles ----
    log.info("Spatial join → tiles...")
    poly_wgs  = polygons.to_crs("EPSG:4326")
    tile_grid = gpd.read_file(SENTINEL2_TILE_PATH).rename(columns={"Name": "tile"})
    joined    = gpd.sjoin(poly_wgs, tile_grid, how="inner", predicate="intersects")
    tiles     = joined['tile'].unique().tolist()
    log.info(f"  Tiles: {tiles}")

    # ---- S2 timeseries ----
    log.info("Scanning S2 timeseries...")
    s2_all_frames = []
    for tile in tiles:
        s2_b08 = s2_timeseries_fullpath(BAND_NIR, RES, YEAR_START, YEAR_END, tile, SAT)
        s2_b04 = s2_timeseries_fullpath(BAND_RED, RES, YEAR_START, YEAR_END, tile, SAT)
        if s2_b08.empty or s2_b04.empty:
            log.warning(f"  tile {tile}: no imagery")
            continue
        s2_ts = pd.merge(s2_b08, s2_b04, on=['date', 'date_str', 'scl'], how='inner')
        s2_all_frames.append((tile, s2_ts))
        log.info(f"  tile {tile}: {len(s2_ts)} scenes")

    if not s2_all_frames:
        log.error("No Sentinel-2 imagery found. Exiting.")
        return

    best_tile, best_ts = max(s2_all_frames, key=lambda x: len(x[1]))
    log.info(f"  Using tile {best_tile} ({len(best_ts)} scenes) as primary")

    epsg         = get_raster_crs(best_ts[f'b{BAND_NIR}'].iloc[0])
    polygons_proj = polygons.to_crs(epsg=epsg)

    # ---- LOCAL ----
    log.info("\n--- LOCAL extraction ---")
    if not os.path.exists(local_out):
        records_local, meta_rows = run_local_all_polygons(
            polygons_proj, best_ts, n_workers
        )
        local_df = pd.DataFrame(records_local)
        meta_df  = pd.DataFrame(meta_rows)
        local_df.to_parquet(local_out, index=False)
        meta_df.to_parquet(meta_out,   index=False)
        log.info(f"  [local] saved → {local_out}  shape={local_df.shape}")
    else:
        log.info("  [local] loading existing parquet...")
        local_df = pd.read_parquet(local_out)
        meta_df  = pd.read_parquet(meta_out)

    date_cols = sorted([c for c in local_df.columns if c not in ('plot_id', 'point_id')])
    log.info(f"  Date columns: {len(date_cols)} dates")

    # ---- GEE ----
    log.info("\n--- GEE extraction ---")
    if not os.path.exists(gee_out):
        ee.Initialize(project=GEE_PROJECT_ID)

        from pyproj import Transformer
        transformer = Transformer.from_crs(
            f"EPSG:{epsg}", "EPSG:4326", always_xy=True
        )
        lons, lats = transformer.transform(
            meta_df['x_proj'].values, meta_df['y_proj'].values
        )
        points_wgs84 = pd.DataFrame({
            'plot_id':  meta_df['plot_id'].values,
            'point_id': meta_df['point_id'].values,
            'lon': lons,
            'lat': lats,
        })

        date_ndvi_gee = run_gee_for_dates(points_wgs84, date_cols)

        gee_rows = []
        for i, m in meta_df.iterrows():
            row = {'plot_id': int(m['plot_id']), 'point_id': int(m['point_id'])}
            for d in date_cols:
                arr    = date_ndvi_gee.get(d)
                row[d] = float(arr[i]) if (arr is not None and i < len(arr)) else np.nan
            gee_rows.append(row)
        gee_df = pd.DataFrame(gee_rows)
        gee_df.to_parquet(gee_out, index=False)
        log.info(f"  [gee] saved → {gee_out}  shape={gee_df.shape}")
    else:
        log.info("  [gee] loading existing parquet...")

    log.info(f"\nDONE  |  {(time.time()-t0)/60:.2f} mins")
    log.info(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()