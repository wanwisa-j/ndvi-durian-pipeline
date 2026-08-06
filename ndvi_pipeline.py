"""
ndvi_pipeline.py  —  scene-first architecture + deep performance instrumentation
=================================================================================

Architecture (from Decision.md 2026-06-08):
  Tile → Scene → sample all polygons   (was: Tile → Polygon → Scene)

GDAL opens:
  Before  ~23 million  (31k polygons × 250 scenes × 3 bands)
  After   ~750         (250 scenes × 3 bands)

Final output (approved structure):
  PROVINCE/
  ├── point_master.parquet        GeoParquet WGS84 — plot_id,point_id,x,y,lat,lon,geometry
  ├── ndvi_mean.parquet           GeoParquet WGS84 — plot_id,point_id,lat,lon,geometry,YYYY-MM,...
  ├── ndvi_max.parquet
  ├── ndvi_min.parquet
  ├── ndvi_median.parquet
  ├── performance_summary.json
  └── logs/

Point source (authoritative):
  data/grid_points/{PROVINCE_STEM_UPPER}.parquet  — pre-generated, do not regenerate
  If not found → exit with error (set GRID_POINTS_DIR to override path)

Internal working files (not final output):
  _ndvi_tile_{tree}_{tile}.parquet  — deleted after aggregate
"""

from __future__ import annotations

import gc
import glob
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from osgeo import gdal, osr
from shapely import contains_xy

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# =============================================================================
# CONFIGURATION
# =============================================================================

def _require_env(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise EnvironmentError(f"Required env var '{key}' not set.")
    return val

def _env(key: str, default) -> str:
    return os.environ.get(key, str(default)).strip()

_FS_SWAPS = (
    [(f"/fs{n}/", f"/workspaces/fs{n}/") for n in range(1, 9)] +
    [(f"/workspaces/fs{n}/", f"/fs{n}/") for n in range(1, 9)]
)

def _resolve_path(p: str) -> str:
    if not p or os.path.exists(p):
        return p
    for src, dst in _FS_SWAPS:
        if src in p:
            alt = p.replace(src, dst, 1)
            if os.path.exists(alt):
                return alt
    return p

def _resolve_output_path(p: str) -> str:
    if not p or os.path.exists(p):
        return p
    for src, dst in _FS_SWAPS:
        if src in p:
            dst_root = dst.rstrip("/")
            if os.path.isdir(dst_root):
                return p.replace(src, dst, 1)
    return p

province            = _require_env("PROVINCE").lower()
golden_durian_path  = _resolve_path(_require_env("GOLDEN_DURIAN_PATH"))
output_dir          = _resolve_output_path(_require_env("OUTPUT_DIR"))
sentinel2_tile_path = _resolve_path(_env("SENTINEL2_TILE_PATH",
    "/fs2/angkanap/00_MAP_TH/03_SENTINEL-2_TILES/sentinel_2_index_shapefile.shp"))
s2_folder_old    = _resolve_path(_env("S2_FOLDER_OLD",    "/fs7/sentinel2/tiles"))
s2_folder_new    = _resolve_path(_env("S2_FOLDER_NEW",    "/fs2/sentinel2/tiles"))
s2_folder_cutoff = pd.Timestamp(_env("S2_FOLDER_CUTOFF", "2025-07-01"))
tree       = _env("TREE", "segment")
date_start = pd.Timestamp(_env("DATE_START", "2019-01-01"))
date_end   = pd.Timestamp(_env("DATE_END",   "2026-12-31"))
band1_name = _env("BAND1", "08")
band2_name          = _env("BAND2", "04")
sat                 = _env("SAT", "B")
res                 = 10

# Grid points directory (auto-discovered from GOLDEN_DURIAN_PATH if empty)
GRID_POINTS_DIR      = _env("GRID_POINTS_DIR", "")
GRID_POINTS_FILE     = _env("GRID_POINTS_FILE", "")
FORCE_REBUILD_POINTS = _env("FORCE_REBUILD_POINTS", "0").lower() in ("1", "true", "yes")

SOFT_RATIO       = float(_env("SOFT_RATIO", 0.92))
MIN_FREE_RAM_GB  = float(_env("MIN_FREE_RAM_GB", 10.0))
POLL_SECONDS     = float(_env("POLL_SECONDS", 5.0))
MAX_WAIT_SECONDS = float(_env("MAX_WAIT_SECONDS", 600.0))

DN_CORRECTION_CUTOFF = pd.Timestamp("2022-01-25")
DN_CORRECTION_VALUE  = 1000

# =============================================================================
# LOGGING
# =============================================================================

_LOG_FILE: Optional[str] = None

def setup_logging(log_dir: str, province: str) -> str:
    global _LOG_FILE
    os.makedirs(log_dir, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir, f"ndvi_{province}_{ts}.log")
    _LOG_FILE = log_path
    fmt  = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return log_path

def _worker_setup_logging(log_path: str) -> None:
    fmt  = logging.Formatter(
        "%(asctime)s [%(levelname)s][pid-%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

# =============================================================================
# PERFORMANCE ACCUMULATOR
# =============================================================================

class _PerfAccum:
    """Thread-safe accumulator for wall-clock totals across all workers."""

    _TIMING_FIELDS = (
        "startup_load_polygons",
        "startup_load_tile_grid",
        "startup_spatial_join",
        "point_gen",
        "raster_open",
        "raster_scl_total", "raster_scl_io",
        "raster_b1_total",  "raster_b1_io",
        "raster_b2_total",  "raster_b2_io",
        "ndvi_calc",
        "parquet_write",
        "agg_schema_scan",
        "agg_read",
        "agg_month_compute",
        "agg_write",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        for f in self._TIMING_FIELDS:
            setattr(self, f, 0.0)
        self.n_scenes_done     = 0
        self.n_clear_pts_total = 0

    def add(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, getattr(self, k) + v)

    def _grand_total(self) -> float:
        raster = (self.raster_open +
                  self.raster_scl_total + self.raster_b1_total + self.raster_b2_total)
        agg    = (self.agg_schema_scan + self.agg_read +
                  self.agg_month_compute + self.agg_write)
        return (self.startup_load_polygons + self.startup_load_tile_grid +
                self.startup_spatial_join + self.point_gen +
                raster + self.ndvi_calc + self.parquet_write + agg)

    def _pct(self, val: float) -> float:
        gt = self._grand_total()
        return 100.0 * val / gt if gt > 0 else 0.0

    def summary(self) -> str:
        gt   = self._grand_total()
        rscl = self.raster_scl_total
        rb1  = self.raster_b1_total
        rb2  = self.raster_b2_total

        def row(label: str, sec: float) -> str:
            return f"  {label:<32} {sec/60:7.2f} min  {self._pct(sec):5.1f}%"

        lines = [
            "",
            "=" * 62,
            "  PERFORMANCE BREAKDOWN",
            "=" * 62,
            row("Startup: load polygons",       self.startup_load_polygons),
            row("Startup: load tile grid",      self.startup_load_tile_grid),
            row("Startup: spatial join",        self.startup_spatial_join),
            row("Point load / filter",          self.point_gen),
            "-" * 62,
            row("Raster open  (gdal.Open×3)",   self.raster_open),
            row("Raster SCL   total",           rscl),
            f"  {'  └─ ReadAsArray only':<32} {self.raster_scl_io/60:7.2f} min",
            row("Raster B08   total",           rb1),
            f"  {'  └─ ReadAsArray only':<32} {self.raster_b1_io/60:7.2f} min",
            row("Raster B04   total",           rb2),
            f"  {'  └─ ReadAsArray only':<32} {self.raster_b2_io/60:7.2f} min",
            row("NDVI compute (incl DN-corr)",  self.ndvi_calc),
            "-" * 62,
            row("Parquet write (tile output)",  self.parquet_write),
            "-" * 62,
            row("Aggregate: schema scan",       self.agg_schema_scan),
            row("Aggregate: parquet read",      self.agg_read),
            row("Aggregate: month compute",     self.agg_month_compute),
            row("Aggregate: parquet write",     self.agg_write),
            "=" * 62,
            row("TOTAL (tracked)",              gt),
            "=" * 62,
            f"  Scenes processed : {self.n_scenes_done}",
            f"  Clear pts sampled: {self.n_clear_pts_total:,}",
            "=" * 62,
            "",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        gt = self._grand_total()
        d: dict = {"grand_total_s": round(gt, 2)}
        for f in self._TIMING_FIELDS:
            v = getattr(self, f)
            d[f"{f}_s"]   = round(v, 3)
            d[f"{f}_pct"] = round(self._pct(v), 1)
        d["n_scenes_done"]     = self.n_scenes_done
        d["n_clear_pts_total"] = self.n_clear_pts_total
        return d


_perf = _PerfAccum()

# =============================================================================
# RESOURCE MONITOR
# =============================================================================

class ResourceMonitor:
    def __init__(self, interval: float = 60.0) -> None:
        self._interval = interval
        self._stop     = threading.Event()
        self._thread   = threading.Thread(
            target=self._run, daemon=True, name="ResourceMonitor")

    def start(self) -> None:
        if _HAS_PSUTIL:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        log = logging.getLogger()
        while not self._stop.wait(self._interval):
            try:
                cpu  = psutil.cpu_percent(interval=None)
                vm   = psutil.virtual_memory()
                proc = psutil.Process()
                rss  = proc.memory_info().rss / (1024 ** 3)
                kids = len(proc.children())
                log.info(
                    "RESOURCE  cpu=%.1f%%  ram_used=%.1fGB(%.1f%%)  "
                    "ram_avail=%.1fGB  proc_rss=%.1fGB  workers=%d",
                    cpu,
                    vm.used / (1024 ** 3), vm.percent,
                    vm.available / (1024 ** 3),
                    rss, kids,
                )
            except Exception as exc:
                log.debug("ResourceMonitor: %s", exc)

# =============================================================================
# MEMORY HELPERS
# =============================================================================

def _get_system_memory_gb() -> tuple[float, float, float]:
    if not _HAS_PSUTIL:
        return 0.0, 0.0, 0.0
    vm = psutil.virtual_memory()
    g  = 1.0 / (1024 ** 3)
    return vm.total * g, vm.available * g, vm.used * g

def get_cgroup_memory_gb():
    try:
        lp, cp = "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"
        if not (os.path.exists(lp) and os.path.exists(cp)):
            return None, None
        with open(lp) as f:
            raw = f.read().strip()
        lb = None if raw.lower() == "max" else int(raw)
        with open(cp) as f:
            cb = int(f.read().strip())
        g = 1.0 / (1024 ** 3)
        return (None if lb is None else lb * g), cb * g
    except Exception:
        return None, None

def _get_effective_available_memory_gb() -> float:
    _, sys_av, _ = _get_system_memory_gb()
    lim, cur     = get_cgroup_memory_gb()
    if lim is not None and cur is not None:
        return min(sys_av, max(0.0, lim - cur))
    return sys_av

def _get_process_tree_memory_gb() -> tuple[float, str]:
    if not _HAS_PSUTIL:
        return 0.0, "unavailable"
    proc      = psutil.Process()
    all_procs = [proc] + proc.children(recursive=True)
    for metric in ("pss", "uss", "rss"):
        try:
            total = 0
            for p in all_procs:
                try:
                    if metric in ("pss", "uss"):
                        total += getattr(p.memory_full_info(), metric, 0) or 0
                    else:
                        total += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total / (1024 ** 3), metric
        except Exception:
            continue
    return 0.0, "error"

def wait_for_memory_budget(
    max_ram_gb=None,
    soft_ratio=SOFT_RATIO,
    poll_seconds=POLL_SECONDS,
    max_wait_seconds=MAX_WAIT_SECONDS,
    min_free_ram_gb=MIN_FREE_RAM_GB,
):
    if not _HAS_PSUTIL:
        return
    if max_ram_gb is None:
        cg, _ = get_cgroup_memory_gb()
        if cg is not None:
            max_ram_gb   = cg
            limit_source = f"cgroup ({cg:.0f}GB)"
        else:
            tot, _, _    = _get_system_memory_gb()
            max_ram_gb   = tot
            limit_source = f"system ({tot:.0f}GB)"
    else:
        limit_source = f"explicit ({max_ram_gb:.0f}GB)"
    threshold = max_ram_gb * soft_ratio
    waited    = 0.0
    while True:
        _, sys_av, _ = _get_system_memory_gb()
        eff_av       = _get_effective_available_memory_gb()
        tree, metric = _get_process_tree_memory_gb()
        if metric == "rss":
            over = tree >= threshold and sys_av < min_free_ram_gb
        elif metric in ("pss", "uss"):
            over = tree >= threshold
        else:
            over = False
        if not over and eff_av >= min_free_ram_gb:
            return
        if waited >= max_wait_seconds:
            logging.warning("Memory guard timeout %.0fs — forcing continue.", waited)
            return
        logging.info(
            "Memory guard [%s %.1fGB]: tree_%s=%.1fGB avail=%.1fGB waited=%.0fs",
            limit_source, threshold, metric, tree, eff_av, waited)
        time.sleep(poll_seconds)
        waited += poll_seconds

def cleanup_memory() -> None:
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def auto_select_workers(estimated_per_worker_gb: float = 2.0,
                        reserve_gb: float = 10.0) -> int:
    cpu_cap = min(os.cpu_count() or 1, 16)
    if not _HAS_PSUTIL:
        return 1
    _, sys_av, _   = _get_system_memory_gb()
    cg_lim, cg_cur = get_cgroup_memory_gb()
    budgets        = [max(0.0, sys_av - reserve_gb)]
    cg_av          = None
    if cg_lim is not None and cg_cur is not None:
        cg_av = max(0.0, cg_lim - cg_cur)
        budgets.append(max(0.0, cg_av - reserve_gb))
    budget  = min(budgets)
    workers = max(1, min(cpu_cap, int(budget / estimated_per_worker_gb)))
    logging.info("AUTO WORKERS sys_av=%.1fGB cg_av=%s budget=%.1fGB workers=%d",
                 sys_av, f"{cg_av:.1f}GB" if cg_av else "none", budget, workers)
    return workers

N_JOBS_ENV = _env("N_JOBS", "auto")
N_JOBS     = auto_select_workers(2.0) if N_JOBS_ENV.lower() == "auto" else int(N_JOBS_ENV)

# =============================================================================
# RASTER UTILITIES
# =============================================================================

def get_raster_crs(raster_path: str) -> int:
    ds  = gdal.Open(raster_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    ds  = None
    return int(srs.GetAttrValue("AUTHORITY", 1))

def s2_timeseries_fullpath(
    band_no: str, res: int, date_start: pd.Timestamp, date_end: pd.Timestamp,
    tile_name: str, sat: str,
) -> pd.DataFrame:
    tile = f"{tile_name[0:2]}/{tile_name[2:3]}/{tile_name[3:5]}"
    log  = logging.getLogger()

    # date_str → (band_path, omni_path, folder_label); NEW overwrites OLD on dupe
    scenes: dict[str, tuple[str, str, str]] = {}

    def _scan_folder(folder: str, label: str, overwrite: bool) -> None:
        tile_dir = os.path.join(folder, tile)
        if not os.path.exists(tile_dir):
            return
        for fn in os.listdir(tile_dir):
            parts = fn.split("_")
            if len(parts) < 3:
                continue
            if fn[2:3] != sat:
                continue
            try:
                og = glob.glob(os.path.join(tile_dir, fn, "GRANULE", "*",
                               "IMG_DATA", "R20m", "*_OMNI_SCL_20m.tif"))
                if len(og) != 1:
                    continue
                bg = glob.glob(os.path.join(tile_dir, fn, "GRANULE", "*",
                               "IMG_DATA", f"R{res}m", f"*_B{band_no}_{res}m.jp2"))
                if len(bg) != 1:
                    continue
                date_str = os.path.basename(bg[0])[7:15]
                dt       = pd.to_datetime(date_str, format="%Y%m%d")
                if not (date_start <= dt <= date_end):
                    continue
                if overwrite or date_str not in scenes:
                    scenes[date_str] = (bg[0], og[0], label)
            except Exception:
                continue

    # OLD first; NEW second — NEW wins on date duplicates
    _scan_folder(s2_folder_old, "OLD(/fs7)", overwrite=False)
    _scan_folder(s2_folder_new, "NEW(/fs2)", overwrite=True)

    if not scenes:
        return pd.DataFrame()

    scene_list, omni_list, date_list = [], [], []
    for date_str in sorted(scenes):
        band_path, omni_path, folder_label = scenes[date_str]
        if s2_folder_cutoff is not None:
            dt       = pd.to_datetime(date_str, format="%Y%m%d")
            expected = "OLD(/fs7)" if dt < s2_folder_cutoff else "NEW(/fs2)"
            log.info(
                "SCENE PATH CHECK  date=%s  folder=%s  expected=%s  ok=%s  path=%s",
                date_str, folder_label, expected, folder_label == expected, band_path,
            )
        scene_list.append(band_path)
        omni_list.append(omni_path)
        date_list.append(date_str)

    df = pd.DataFrame({
        "date":        pd.to_datetime(date_list, format="%Y%m%d"),
        "omni":        omni_list,
        f"b{band_no}": scene_list,
    }).sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y/%m/%d")
    return df

# =============================================================================
# RASTER SAMPLING
# =============================================================================

def create_points_in_polygon(geom, spacing: int = 10):
    minx, miny, maxx, maxy = geom.bounds
    xx, yy = np.meshgrid(np.arange(minx, maxx, spacing),
                          np.arange(miny, maxy, spacing))
    xx, yy = xx.ravel(), yy.ravel()
    mask   = contains_xy(geom, xx, yy)
    return xx[mask], yy[mask]

def sample_raster_windowed(
    ds, xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, float]:
    """Sample raster at (xs, ys) via bounding-box window read.

    Returns:
        vals    – float32 array of sampled values
        t_io    – ReadAsArray wall-clock time (GDAL I/O only, no numpy)
    """
    gt   = ds.GetGeoTransform()
    px   = ((xs - gt[0]) / gt[1]).astype(np.int32)
    py   = ((ys - gt[3]) / gt[5]).astype(np.int32)
    oob  = (px < 0) | (px >= ds.RasterXSize) | (py < 0) | (py >= ds.RasterYSize)
    px   = np.clip(px, 0, ds.RasterXSize - 1)
    py   = np.clip(py, 0, ds.RasterYSize - 1)
    xmn, xmx = int(px.min()), int(px.max())
    ymn, ymx = int(py.min()), int(py.max())
    band = ds.GetRasterBand(1)

    _t0   = time.perf_counter()
    arr   = band.ReadAsArray(xmn, ymn, xmx - xmn + 1, ymx - ymn + 1)
    t_io  = time.perf_counter() - _t0

    if arr is None:
        return np.full(len(xs), np.nan, dtype=np.float32), t_io

    vals  = arr[py - ymn, px - xmn].astype(np.float32)
    nd    = band.GetNoDataValue()
    if nd is not None:
        vals = np.where(vals == nd, np.nan, vals)
    if oob.any():
        vals[oob] = np.nan
    return vals, t_io

def apply_dn_correction(values: np.ndarray, date_str: str) -> np.ndarray:
    """Subtract 1000 for scenes acquired after 2022-01-25 (ESA baseline 04.00)."""
    if pd.Timestamp(date_str.replace("/", "-")) > DN_CORRECTION_CUTOFF:
        return values - DN_CORRECTION_VALUE
    return values

# =============================================================================
# PROVINCE GRID  (pre-generated, authoritative source of sampling points)
# =============================================================================

_GRID_REQUIRED_COLS = {"plot_id", "point_id", "lon", "lat"}


def discover_province_grid() -> str | None:
    """Find the pre-generated grid parquet for the active province.

    Search order:
      1. GRID_POINTS_FILE env var (specific file override)
      2. GRID_POINTS_DIR env var (if set)
      3. Auto: dirname(dirname(GOLDEN_DURIAN_PATH))/grid_points/

    Match rule: normalize both stems to uppercase, strip trailing space+digits,
    take first sorted match.
    """
    if GRID_POINTS_FILE and os.path.isfile(GRID_POINTS_FILE):
        logging.info("GRID FILE override: %s", GRID_POINTS_FILE)
        return GRID_POINTS_FILE

    grid_dir = _resolve_path(GRID_POINTS_DIR) if GRID_POINTS_DIR else None
    if not grid_dir:
        grid_dir = os.path.join(
            os.path.dirname(os.path.dirname(golden_durian_path)), "grid_points")
        grid_dir = _resolve_path(grid_dir)

    if not os.path.isdir(grid_dir):
        logging.warning("GRID DIR not found: %s", grid_dir)
        return None

    # Normalize polygon stem: e.g. "RAYONG_durian_v2_cuda_post_post" → "RAYONG_DURIAN_V2_CUDA_POST_POST"
    poly_stem = os.path.splitext(os.path.basename(golden_durian_path))[0].upper()

    best = None
    for fn in sorted(os.listdir(grid_dir)):
        if not fn.endswith(".parquet"):
            continue
        # Strip trailing " 1", " 2", etc. then strip "_GRID" suffix (new naming convention)
        fn_stem = os.path.splitext(fn)[0].upper().rstrip(" 0123456789").strip()
        fn_stem = fn_stem.replace(" ", "_").removesuffix("_GRID")
        poly_norm = poly_stem.replace(" ", "_")
        if fn_stem == poly_norm or fn_stem.startswith(poly_norm):
            best = os.path.join(grid_dir, fn)
            break

    if best:
        logging.info("GRID DISCOVERED  %s", best)
    else:
        logging.warning("GRID NOT FOUND for stem=%s in %s", poly_stem, grid_dir)
    return best


def load_and_validate_province_grid(grid_path: str) -> pd.DataFrame | None:
    """Load and validate province grid.  Returns DataFrame or None on failure.

    Loaded columns: plot_id, point_id, lon, lat
    plot_id = 0-based polygon row index in golden_durian_path
    """
    t0 = time.perf_counter()
    try:
        schema_cols = set(pq.read_schema(grid_path).names)
        missing     = _GRID_REQUIRED_COLS - schema_cols
        if missing:
            logging.error("PROVINCE GRID INVALID: missing columns %s  %s", missing, grid_path)
            return None

        df = pd.read_parquet(grid_path, columns=["plot_id", "point_id", "lon", "lat"])
        if len(df) == 0:
            logging.error("PROVINCE GRID INVALID: 0 rows  %s", grid_path)
            return None

        el   = time.perf_counter() - t0
        sz_mb = os.path.getsize(grid_path) / (1024 ** 2)
        logging.info(
            "POINT CACHE HIT  province=%s  path=%s  rows=%d  %.1fMB  elapsed=%.1fs",
            province, grid_path, len(df), sz_mb, el,
        )
        return df

    except Exception as exc:
        logging.error("PROVINCE GRID load error: %s  %s", exc, grid_path)
        return None


def _pts_from_province_grid_for_tile(
    province_grid: pd.DataFrame,
    tile_poly_ids: list,   # values of golden_durian["plot_id"] for this tile
                            # (not row position -- must match province_grid["plot_id"],
                            # which is the source polygon file's own plot_id when
                            # present, same as grid_point.py's fallback otherwise)
    s2_epsg: int,
    tile: str,
) -> tuple[pd.DataFrame, int] | tuple[None, int]:
    """Filter province grid to tile's polygons and reproject to tile UTM.

    Returns:
        (pts_df, n_polys)
        pts_df  – columns: poly_idx, plot_id, point_id, x(UTM), y(UTM), lat, lon
                  sorted by poly_idx then point_id
                  poly_idx = tile-local slot (0..len(tile_poly_ids)-1)
        n_polys – len(tile_poly_ids), includes polygons with 0 points

    On failure: (None, len(tile_poly_ids))
    """
    n_polys = len(tile_poly_ids)
    t0      = time.perf_counter()

    id_set = set(tile_poly_ids)
    pts    = province_grid[province_grid["plot_id"].isin(id_set)].copy()

    if pts.empty:
        logging.warning("GRID FILTER tile=%s: 0 points for %d polygons", tile, n_polys)
        return None, n_polys

    # Map plot_id (global row index) → tile-local slot
    plot_id_to_slot = {pid: slot for slot, pid in enumerate(tile_poly_ids)}
    pts["poly_idx"] = pts["plot_id"].map(plot_id_to_slot).astype(np.int32)
    pts = pts.sort_values(["poly_idx", "point_id"]).reset_index(drop=True)

    # Reproject lon/lat → tile UTM (safe for cross-zone provinces)
    t_repr  = time.perf_counter()
    gdf     = gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts["lon"], pts["lat"]),
        crs="EPSG:4326",
    )
    gdf_utm = gdf.to_crs(epsg=s2_epsg)
    pts["x"] = gdf_utm.geometry.x.values.astype(np.float64)
    pts["y"] = gdf_utm.geometry.y.values.astype(np.float64)
    t_repr_el = time.perf_counter() - t_repr

    n_polys_with_pts = pts["poly_idx"].nunique()
    elapsed          = time.perf_counter() - t0

    logging.info(
        "GRID FILTER tile=%s  polys_in_tile=%d  polys_with_pts=%d  "
        "pts=%d  avg_pts=%.1f  reproject=%.2fs  total=%.2fs  EPSG:%d",
        tile, n_polys, n_polys_with_pts, len(pts),
        len(pts) / max(n_polys_with_pts, 1),
        t_repr_el, elapsed, s2_epsg,
    )
    _perf.add(point_gen=elapsed)

    return pts[["poly_idx", "plot_id", "point_id", "x", "y", "lat", "lon"]], n_polys

# =============================================================================
# POINT MASTER  (province-wide WGS84 GeoParquet — final output)
# =============================================================================

def _point_master_path() -> str:
    return os.path.join(output_dir, province, "point_master.parquet")


def build_or_update_point_master(province_grid_df: pd.DataFrame) -> None:
    """Build province-wide GeoParquet directly from pre-generated grid.

    Schema: plot_id, point_id, x(=lon), y(=lat), lat, lon, geometry (WGS84)
    """
    out_path = _point_master_path()
    if os.path.exists(out_path) and not FORCE_REBUILD_POINTS:
        sz_mb = os.path.getsize(out_path) / (1024 ** 2)
        logging.info("POINT MASTER exists — skip  %s  (%.1fMB)", out_path, sz_mb)
        return

    t0  = time.perf_counter()
    df  = province_grid_df[["plot_id", "point_id", "lon", "lat"]].copy()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    gdf["x"] = gdf["lon"]   # WGS84: x = longitude
    gdf["y"] = gdf["lat"]   # WGS84: y = latitude
    gdf = gdf[["plot_id", "point_id", "x", "y", "lat", "lon", "geometry"]]

    gdf.to_parquet(out_path, index=False, compression="snappy", row_group_size=200_000)
    sz_mb = os.path.getsize(out_path) / (1024 ** 2)
    logging.info("POINT MASTER saved  rows=%d  %.1fMB  %.2fs  %s",
                 len(gdf), sz_mb, time.perf_counter() - t0, out_path)

# =============================================================================
# WORKER STATE  (set once per worker via ProcessPoolExecutor initializer)
# =============================================================================

_W_LOG_PATH : Optional[str]        = None
_W_OFFSETS  : Optional[np.ndarray] = None
_W_XS       : Optional[np.ndarray] = None
_W_YS       : Optional[np.ndarray] = None

def _worker_init(
    log_path: str,
    offsets_b: bytes,
    xs_b: bytes,
    ys_b: bytes,
) -> None:
    global _W_LOG_PATH, _W_OFFSETS, _W_XS, _W_YS
    _W_LOG_PATH = log_path
    _W_OFFSETS  = np.frombuffer(offsets_b, dtype=np.int64).copy()
    _W_XS       = np.frombuffer(xs_b,      dtype=np.float64).copy()
    _W_YS       = np.frombuffer(ys_b,      dtype=np.float64).copy()
    _worker_setup_logging(log_path)

# =============================================================================
# PER-SCENE WORKER
# =============================================================================

def process_scene(
    date_str: str,
    omni_path: str,
    b1_path: str,
    b2_path: str,
) -> tuple[str, np.ndarray, dict]:
    """Open one scene once, sample all polygon points, return NDVI.

    Returns:
        date_str    – echo of input
        ndvi_flat   – float32 array, length = sum(offsets) across all polygons
        prof        – per-scene timing dict:
                        t_open, t_scl_total, t_scl_io, t_b1_total, t_b1_io,
                        t_b2_total, t_b2_io, t_ndvi, n_clear_pts, n_total_pts
    """
    log      = logging.getLogger()
    n_pts    = int(_W_OFFSETS[-1])   # total points = last element of prefix-sum offsets
    n_polys  = len(_W_OFFSETS) - 1
    ndvi_out = np.full(n_pts, np.nan, dtype=np.float32)

    prof: dict = {
        "t_open": 0.0,
        "t_scl_total": 0.0, "t_scl_io": 0.0,
        "t_b1_total":  0.0, "t_b1_io":  0.0,
        "t_b2_total":  0.0, "t_b2_io":  0.0,
        "t_ndvi": 0.0,
        "n_polys": n_polys,
        "n_clear_pts": 0,
        "n_total_pts": n_pts,
    }

    omni_ds = b1_ds = b2_ds = None
    _t_scene = time.perf_counter()

    try:
        _t = time.perf_counter()
        omni_ds = gdal.Open(omni_path)
        b1_ds   = gdal.Open(b1_path)
        b2_ds   = gdal.Open(b2_path)
        prof["t_open"] = time.perf_counter() - _t

        if omni_ds is None or b1_ds is None or b2_ds is None:
            log.warning("process_scene %s — could not open rasters", date_str)
            return date_str, ndvi_out, prof

        for i in range(n_polys):
            s  = int(_W_OFFSETS[i])
            e  = int(_W_OFFSETS[i + 1])
            if s == e:
                continue    # polygon had 0 points in grid
            xs = _W_XS[s:e]
            ys = _W_YS[s:e]
            vn = np.full(e - s, np.nan, dtype=np.float32)

            try:
                _t = time.perf_counter()
                ov, t_scl_io = sample_raster_windowed(omni_ds, xs, ys)
                prof["t_scl_total"] += time.perf_counter() - _t
                prof["t_scl_io"]    += t_scl_io

                clear = ov == 0
                if clear.any():
                    xs_c, ys_c = xs[clear], ys[clear]
                    n_clear    = int(clear.sum())
                    prof["n_clear_pts"] += n_clear

                    _t = time.perf_counter()
                    t1_raw, t_b1_io = sample_raster_windowed(b1_ds, xs_c, ys_c)
                    prof["t_b1_total"] += time.perf_counter() - _t
                    prof["t_b1_io"]    += t_b1_io

                    _t = time.perf_counter()
                    t2_raw, t_b2_io = sample_raster_windowed(b2_ds, xs_c, ys_c)
                    prof["t_b2_total"] += time.perf_counter() - _t
                    prof["t_b2_io"]    += t_b2_io

                    # S2 metadata: NODATA=0, SATURATED=65535 — jp2 drivers rarely
                    # embed these in band metadata so GetNoDataValue() often returns None
                    _nd_mask = (t1_raw == 0) | (t1_raw == 65535) | \
                               (t2_raw == 0) | (t2_raw == 65535)
                    if _nd_mask.any():
                        t1_raw = np.where(_nd_mask, np.nan, t1_raw)
                        t2_raw = np.where(_nd_mask, np.nan, t2_raw)

                    _t = time.perf_counter()
                    t1    = apply_dn_correction(t1_raw, date_str)
                    t2    = apply_dn_correction(t2_raw, date_str)
                    denom = t1 + t2
                    valid = np.isfinite(denom) & (denom != 0)
                    vc    = np.full(n_clear, np.nan, dtype=np.float32)
                    vc[valid] = (t1[valid] - t2[valid]) / denom[valid]
                    vn[clear] = vc
                    prof["t_ndvi"] += time.perf_counter() - _t

            except Exception as exc:
                log.error("process_scene %s poly_slot=%d: %s", date_str, i, exc)

            ndvi_out[s:e] = vn

    except Exception as exc:
        log.error("process_scene %s fatal: %s", date_str, exc)
    finally:
        omni_ds = b1_ds = b2_ds = None

    t_total = time.perf_counter() - _t_scene
    t_other = max(0.0, t_total - (
        prof["t_open"] + prof["t_scl_total"] +
        prof["t_b1_total"] + prof["t_b2_total"] + prof["t_ndvi"]))

    log.info(
        "SCENE %-12s  open=%5.2fs  "
        "scl=%5.2fs(io=%5.2fs)  b1=%5.2fs(io=%5.2fs)  b2=%5.2fs(io=%5.2fs)  "
        "ndvi=%5.2fs  other=%5.2fs  total=%5.2fs  clear_pts=%d",
        date_str,
        prof["t_open"],
        prof["t_scl_total"], prof["t_scl_io"],
        prof["t_b1_total"],  prof["t_b1_io"],
        prof["t_b2_total"],  prof["t_b2_io"],
        prof["t_ndvi"], t_other, t_total,
        prof["n_clear_pts"],
    )
    return date_str, ndvi_out, prof

# =============================================================================
# AGGREGATE  (ndvi_tile → 4 GeoParquet output files)
# =============================================================================

def _log_output_validation(log, label: str, month_cols: list) -> None:
    n         = len(month_cols)
    first     = month_cols[0]  if month_cols else "—"
    last      = month_cols[-1] if month_cols else "—"
    is_sorted = month_cols == sorted(month_cols)
    seen: set = set()
    dupes     = [c for c in month_cols if c in seen or seen.add(c)]  # type: ignore[func-returns-value]
    fmt_ok    = all(re.fullmatch(r"\d{4}-\d{2}", c) for c in month_cols)
    log.info(
        "OUTPUT VALIDATION [%s]  months=%d  first=%s  last=%s  "
        "sorted=%s  fmt_ok=%s  dupes=%s",
        label, n, first, last, is_sorted, fmt_ok,
        dupes if dupes else "none",
    )


def aggregate_ndvi(province: str, output_dir: str, tree: str) -> None:
    log          = logging.getLogger()
    province_dir = os.path.join(output_dir, province)
    pattern      = os.path.join(province_dir, f"_ndvi_tile_{tree}_*.parquet")
    files        = sorted(glob.glob(pattern))
    if not files:
        log.warning("AGGREGATE: no tile parquets found matching %s", pattern)
        return

    log.info("AGGREGATE START  files=%d", len(files))
    t_agg_start = time.perf_counter()

    _skip   = {"plot_id", "point_id", "lat", "lon"}
    months: set = set()
    t0 = time.perf_counter()
    for f in files:
        try:
            for col in pq.read_schema(f).names:
                if col in _skip:
                    continue
                try:
                    months.add(pd.to_datetime(col, format="%Y/%m/%d").strftime("%Y-%m"))
                except Exception:
                    pass
        except Exception as exc:
            log.warning("AGGREGATE schema scan skip %s: %s", f, exc)
    t_schema = time.perf_counter() - t0
    _perf.add(agg_schema_scan=t_schema)
    all_month_cols = sorted(months)
    log.info("AGGREGATE: %d months  schema_scan=%.2fs", len(all_month_cols), t_schema)
    if not all_month_cols:
        log.warning("AGGREGATE: no date columns found — aborting")
        return

    mean_path   = os.path.join(province_dir, "ndvi_mean.parquet")
    max_path    = os.path.join(province_dir, "ndvi_max.parquet")
    min_path    = os.path.join(province_dir, "ndvi_min.parquet")
    median_path = os.path.join(province_dir, "ndvi_median.parquet")
    for p in [mean_path, max_path, min_path, median_path]:
        if os.path.exists(p):
            os.remove(p)

    out_paths   = [mean_path, max_path, min_path, median_path]
    stat_labels = ["mean", "max", "min", "median"]
    # Collect plain DataFrames per stat — geometry added once at write time
    dfs_collect: list[list[pd.DataFrame]] = [[], [], [], []]
    t_read_total = t_compute_total = 0.0
    meta_cols    = ["plot_id", "point_id", "lat", "lon"]

    for fi, f in enumerate(files, 1):
        log.info("AGGREGATE [%d/%d] %s", fi, len(files), os.path.basename(f))
        try:
            t0    = time.perf_counter()
            df    = pd.read_parquet(f)
            # Drop legacy geometry column if present in old tile parquets
            if "geometry" in df.columns:
                df = df.drop(columns=["geometry"])
            t_r   = time.perf_counter() - t0
            t_read_total += t_r
            sz_mb = os.path.getsize(f) / (1024 ** 2)
            log.info("  read %.1fMB  rows=%d  %.2fs  %.1fMB/s",
                     sz_mb, len(df), t_r, sz_mb / t_r if t_r > 0 else 0)

            t0        = time.perf_counter()
            date_cols = [c for c in df.columns if c not in set(meta_cols)]
            if not date_cols:
                continue
            df[date_cols] = df[date_cols].astype(np.float32)

            month_map: dict = {}
            for col in date_cols:
                try:
                    mk = pd.to_datetime(col, format="%Y/%m/%d").strftime("%Y-%m")
                    month_map.setdefault(mk, []).append(col)
                except Exception:
                    pass
            if not month_map:
                continue

            n_pts = len(df)

            def _build(stat_fn) -> pd.DataFrame:
                out = df[meta_cols].copy()
                for m in all_month_cols:
                    if m in month_map:
                        out[m] = stat_fn(df[month_map[m]]).astype(np.float32)
                    else:
                        out[m] = np.full(n_pts, np.nan, dtype=np.float32)
                return out

            tile_stats = [
                _build(lambda b: b.mean(axis=1,   skipna=True)),
                _build(lambda b: b.max(axis=1,    skipna=True)),
                _build(lambda b: b.min(axis=1,    skipna=True)),
                _build(lambda b: b.median(axis=1, skipna=True)),
            ]
            t_compute_total += time.perf_counter() - t0

            for wi, df_s in enumerate(tile_stats):
                dfs_collect[wi].append(df_s)

            del df, tile_stats
            cleanup_memory()

        except Exception as exc:
            log.error("AGGREGATE skip %s: %s", f, exc)

    _perf.add(agg_read=t_read_total, agg_month_compute=t_compute_total)

    # Concat all tiles per stat, add geometry, write as GeoParquet
    t_write_total = 0.0
    for wi, (dfs, out_p, label) in enumerate(zip(dfs_collect, out_paths, stat_labels)):
        if not dfs:
            log.warning("AGGREGATE: no data collected for %s — skipping", label)
            continue
        t0       = time.perf_counter()
        combined = pd.concat(dfs, ignore_index=True)
        del dfs
        cleanup_memory()

        month_cols = sorted(c for c in combined.columns if c not in set(meta_cols))

        # Polygons on tile boundaries appear in multiple tile parquets → dedup.
        # groupby.first() takes first non-NaN per column, so border points keep
        # whichever tile had valid NDVI rather than silently dropping data.
        n_before = len(combined)
        combined = (
            combined[meta_cols + month_cols]
            .groupby(["plot_id", "point_id"], sort=False, as_index=False)
            .first()
        )
        n_after = len(combined)
        if n_before != n_after:
            log.info("AGGREGATE [%s] dedup: %d → %d rows (%d dups removed)",
                     label, n_before, n_after, n_before - n_after)

        _log_output_validation(log, label, month_cols)
        combined = combined[meta_cols + month_cols]

        gdf = gpd.GeoDataFrame(
            combined,
            geometry=gpd.points_from_xy(combined["lon"], combined["lat"]),
            crs="EPSG:4326",
        )
        del combined
        gdf.to_parquet(out_p)
        t_wr = time.perf_counter() - t0
        t_write_total += t_wr
        sz_mb = os.path.getsize(out_p) / (1024 ** 2)
        log.info("AGGREGATE wrote %s  rows=%d  %.1fMB  %.2fs",
                 label, len(gdf), sz_mb, t_wr)
        del gdf
        cleanup_memory()

    t_el = time.perf_counter() - t_agg_start
    _perf.add(agg_write=t_write_total)
    log.info("AGGREGATE COMPLETE  total=%.1fs  read=%.1fs  compute=%.1fs  write=%.1fs",
             t_el, t_read_total, t_compute_total, t_write_total)
    cleanup_memory()

    for p in out_paths:
        if os.path.exists(p):
            sz = os.path.getsize(p) / (1024 ** 2)
            log.info("saved → %s  (%.1fMB)", p, sz)

# =============================================================================
# PERFORMANCE SUMMARY JSON
# =============================================================================

def save_performance_summary(
    run_start: float,
    tiles_done: int,
    total_points: int,
) -> None:
    out_path = os.path.join(output_dir, province, "performance_summary.json")
    elapsed  = time.perf_counter() - run_start
    data     = {
        "province":           province,
        "run_date":           datetime.now().isoformat(timespec="seconds"),
        "total_elapsed_min":  round(elapsed / 60, 2),
        "tiles_processed":    tiles_done,
        "scenes_processed":   _perf.n_scenes_done,
        "points_written":     total_points,
        "clear_pts_sampled":  _perf.n_clear_pts_total,
        "n_jobs":             N_JOBS,
        "timing":             _perf.to_dict(),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    logging.info("PERF SUMMARY → %s", out_path)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log_dir  = os.path.join(output_dir, province, "logs")
    log_path = setup_logging(log_dir, province)

    logging.info("=" * 70)
    logging.info("NDVI Pipeline (scene-first)  province=%s  dates=%s to %s",
                 province, date_start.date(), date_end.date())
    logging.info("output_dir=%s  N_JOBS=%d  FORCE_REBUILD_POINTS=%s",
                 output_dir, N_JOBS, FORCE_REBUILD_POINTS)
    cg_lim, cg_cur      = get_cgroup_memory_gb()
    sys_tot, sys_av, _  = _get_system_memory_gb()
    logging.info("sys=%.0fGB avail=%.0fGB  cgroup_lim=%s cgroup_cur=%s",
                 sys_tot, sys_av,
                 f"{cg_lim:.0f}GB" if cg_lim else "none",
                 f"{cg_cur:.0f}GB" if cg_cur else "none")

    os.makedirs(os.path.join(output_dir, province), exist_ok=True)
    run_start = time.perf_counter()

    monitor = ResourceMonitor(interval=60.0)
    monitor.start()

    # ------------------------------------------------------------------
    # 1. Load polygons
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    logging.info("STARTUP: loading polygons  %s", golden_durian_path)
    if golden_durian_path.endswith(".parquet"):
        golden_durian = gpd.read_parquet(golden_durian_path).to_crs("EPSG:4326")
    else:
        golden_durian = gpd.read_file(golden_durian_path).to_crs("EPSG:4326")
    golden_durian = golden_durian.reset_index(drop=True)
    # Mirror grid_point.py's own plot_id fallback exactly: when the source
    # polygon file already carries a plot_id column (real survey exports
    # always do -- it's the record ID, not a 0..N-1 row position), the grid
    # was built keyed on THOSE values, not on row position. Using
    # joined_gdf.index below instead of this column made tile_poly_ids
    # (0-based positions) get compared against province_grid["plot_id"]
    # (arbitrary survey IDs) in _pts_from_province_grid_for_tile -- a type
    # AND value mismatch that silently matched nothing ("GRID FILTER
    # tile=...: 0 points for N polygons") for every polygon file with a
    # pre-existing plot_id column.
    if "plot_id" not in golden_durian.columns:
        golden_durian["plot_id"] = np.arange(len(golden_durian))
    t_poly = time.perf_counter() - t0
    _perf.add(startup_load_polygons=t_poly)
    logging.info("STARTUP: %d polygons loaded  %.2fs", len(golden_durian), t_poly)

    # ------------------------------------------------------------------
    # 2. Load tile grid
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    logging.info("STARTUP: loading S2 tile grid  %s", sentinel2_tile_path)
    sentinel2_tile = gpd.read_file(sentinel2_tile_path).rename(columns={"Name": "tile"})
    t_grid = time.perf_counter() - t0
    _perf.add(startup_load_tile_grid=t_grid)
    logging.info("STARTUP: %d tiles loaded  %.2fs", len(sentinel2_tile), t_grid)

    # ------------------------------------------------------------------
    # 3. Spatial join
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    logging.info("STARTUP: spatial join polygons ↔ S2 tiles")
    joined_gdf = gpd.sjoin(golden_durian, sentinel2_tile, how="inner", predicate="intersects")
    tiles      = joined_gdf["tile"].unique().tolist()
    t_join     = time.perf_counter() - t0
    _perf.add(startup_spatial_join=t_join)
    logging.info("STARTUP: tiles=%d  matched_rows=%d  %.2fs",
                 len(tiles), len(joined_gdf), t_join)

    # ------------------------------------------------------------------
    # 4. Load province grid (authoritative point source)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    grid_path        = discover_province_grid()
    province_grid_df = None

    if grid_path:
        province_grid_df = load_and_validate_province_grid(grid_path)

    if province_grid_df is None:
        poly_stem = os.path.splitext(os.path.basename(golden_durian_path))[0].upper()
        logging.error(
            "PROVINCE GRID NOT FOUND for province=%s\n"
            "  Expected file matching stem '%s' in data/grid_points/\n"
            "  Set GRID_POINTS_DIR env var to override search path.\n"
            "  Grid files available: %s",
            province, poly_stem,
            str(sorted(os.listdir(os.path.dirname(grid_path or ".")))) if grid_path else "—",
        )
        monitor.stop()
        sys.exit(1)

    _perf.add(point_gen=time.perf_counter() - t0)   # grid load time

    # Build point_master.parquet from province grid
    build_or_update_point_master(province_grid_df)

    # Filter grid per tile (fast: just select rows + reproject, no geometry computation)
    logging.info("GRID FILTER: preparing points for %d tiles", len(tiles))
    # all_tile_pts: tile → (pts_df, n_polys)
    all_tile_pts: dict[str, tuple[pd.DataFrame, int]] = {}

    for tile in tiles:
        s2_b1 = s2_timeseries_fullpath(band1_name, res, date_start, date_end, tile, sat)
        if s2_b1.empty:
            continue
        s2_epsg       = get_raster_crs(s2_b1[f"b{band1_name}"].iloc[0])
        tile_poly_ids = joined_gdf.loc[joined_gdf["tile"] == tile, "plot_id"].tolist()
        pts, n_polys  = _pts_from_province_grid_for_tile(
            province_grid_df, tile_poly_ids, s2_epsg, tile)
        if pts is not None and not pts.empty:
            all_tile_pts[tile] = (pts, n_polys)

    # ------------------------------------------------------------------
    # 5. Per-tile scene-first extraction
    # ------------------------------------------------------------------
    total_points_written = 0
    total_tiles_done     = 0

    for tile in tiles:
        logging.info("=" * 60)
        logging.info("TILE START: %s", tile)
        t_tile_start = time.perf_counter()

        s2_b1 = s2_timeseries_fullpath(band1_name, res, date_start, date_end, tile, sat)
        s2_b2 = s2_timeseries_fullpath(band2_name, res, date_start, date_end, tile, sat)
        if s2_b1.empty or s2_b2.empty:
            logging.warning("TILE %s: no imagery — skip", tile)
            continue
        s2_ts = pd.merge(s2_b1, s2_b2, on=["date", "omni"], how="inner")
        if s2_ts.empty:
            logging.warning("TILE %s: no merged scenes — skip", tile)
            continue

        n_scenes = len(s2_ts)
        logging.info("TILE %s: scenes=%d  band%s × band%s",
                     tile, n_scenes, band1_name, band2_name)

        tile_out = os.path.join(output_dir, province,
                                f"_ndvi_tile_{tree}_{tile}.parquet")
        if os.path.exists(tile_out):
            logging.info("TILE %s: output exists — skip (delete to rerun)", tile)
            total_tiles_done += 1
            continue

        if tile not in all_tile_pts:
            logging.warning("TILE %s: no points — skip", tile)
            continue

        pts, n_polys = all_tile_pts[tile]
        n_total_pts  = len(pts)
        n_poly_pts   = pts["poly_idx"].nunique()
        s2_epsg      = get_raster_crs(s2_ts[f"b{band1_name}"].iloc[0])
        logging.info("TILE %s: total_pts=%d  polys_with_pts=%d/%d  avg_pts=%.1f  EPSG:%d",
                     tile, n_total_pts, n_poly_pts, n_polys,
                     n_total_pts / max(n_poly_pts, 1), s2_epsg)

        # Flat arrays for worker — offsets cover ALL n_polys slots (including empties)
        counts  = np.zeros(n_polys, dtype=np.int64)
        np.add.at(counts, pts["poly_idx"].values, 1)
        offsets = np.zeros(n_polys + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        xs_flat  = pts["x"].values.astype(np.float64)
        ys_flat  = pts["y"].values.astype(np.float64)
        lat_flat = pts["lat"].values.astype(np.float32)
        lon_flat = pts["lon"].values.astype(np.float32)

        init_args  = (log_path, offsets.tobytes(), xs_flat.tobytes(), ys_flat.tobytes())
        scene_args = [
            (row.date, row.omni,
             getattr(row, f"b{band1_name}"),
             getattr(row, f"b{band2_name}"))
            for row in s2_ts.itertuples(index=False)
        ]
        date_cols    = s2_ts["date"].tolist()
        ndvi_results: dict[str, np.ndarray] = {}

        pending:   dict = {}
        scene_iter = iter(scene_args)
        exhausted  = False
        n_failed   = 0
        tile_acc   = {k: 0.0 for k in (
            "t_open", "t_scl_total", "t_scl_io",
            "t_b1_total", "t_b1_io", "t_b2_total", "t_b2_io", "t_ndvi")}
        tile_clear_pts = 0

        def _submit():
            nonlocal exhausted
            try:
                args = next(scene_iter)
                wait_for_memory_budget()
                f = pool.submit(process_scene, *args)
                pending[f] = args[0]
                return True
            except StopIteration:
                exhausted = True
                return False

        logging.info("TILE %s: launching %d scenes on %d workers",
                     tile, n_scenes, N_JOBS)
        t_scenes = time.perf_counter()

        with ProcessPoolExecutor(
            max_workers=N_JOBS,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            while len(pending) < N_JOBS and not exhausted:
                _submit()

            while pending:
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for fut in done:
                    date_str = pending.pop(fut)
                    try:
                        d, arr, prof = fut.result()
                        ndvi_results[d] = arr
                        _perf.add(
                            raster_open=prof["t_open"],
                            raster_scl_total=prof["t_scl_total"],
                            raster_scl_io=prof["t_scl_io"],
                            raster_b1_total=prof["t_b1_total"],
                            raster_b1_io=prof["t_b1_io"],
                            raster_b2_total=prof["t_b2_total"],
                            raster_b2_io=prof["t_b2_io"],
                            ndvi_calc=prof["t_ndvi"],
                            n_clear_pts_total=prof["n_clear_pts"],
                            n_scenes_done=1,
                        )
                        for k in tile_acc:
                            tile_acc[k] += prof.get(k, 0.0)
                        tile_clear_pts += prof["n_clear_pts"]
                    except Exception as exc:
                        n_failed += 1
                        logging.error("SCENE %s FAILED: %s", date_str, exc)
                    cleanup_memory()

                for _ in done:
                    if not exhausted:
                        _submit()

        t_scenes_el = time.perf_counter() - t_scenes
        logging.info(
            "TILE %s SCENES DONE  elapsed=%.1fs  failures=%d\n"
            "  breakdown (sum across all scenes):\n"
            "    open:   %.2fs\n"
            "    scl:    %.2fs  (io=%.2fs  index=%.2fs)\n"
            "    b08:    %.2fs  (io=%.2fs  index=%.2fs)\n"
            "    b04:    %.2fs  (io=%.2fs  index=%.2fs)\n"
            "    ndvi:   %.2fs\n"
            "    clear_pts: %d",
            tile, t_scenes_el, n_failed,
            tile_acc["t_open"],
            tile_acc["t_scl_total"], tile_acc["t_scl_io"],
            max(0, tile_acc["t_scl_total"] - tile_acc["t_scl_io"]),
            tile_acc["t_b1_total"],  tile_acc["t_b1_io"],
            max(0, tile_acc["t_b1_total"] - tile_acc["t_b1_io"]),
            tile_acc["t_b2_total"],  tile_acc["t_b2_io"],
            max(0, tile_acc["t_b2_total"] - tile_acc["t_b2_io"]),
            tile_acc["t_ndvi"],
            tile_clear_pts,
        )

        # Save tile NDVI parquet: plot_id, point_id, lat, lon, {date_cols}
        # No geometry — aggregate adds it
        t_save  = time.perf_counter()
        _tile_dict = {
            "plot_id":  pts["plot_id"].values,
            "point_id": pts["point_id"].values,
            "lat":      lat_flat,
            "lon":      lon_flat,
        }
        for d in sorted(date_cols):
            _tile_dict[d] = (ndvi_results[d] if d in ndvi_results
                             else np.full(n_total_pts, np.nan, dtype=np.float32))
        df_tile = pd.DataFrame(_tile_dict)

        os.makedirs(os.path.dirname(tile_out), exist_ok=True)
        df_tile.to_parquet(tile_out, index=False, compression="snappy",
                           row_group_size=200_000)

        t_save_el = time.perf_counter() - t_save
        sz_mb     = os.path.getsize(tile_out) / (1024 ** 2)
        _perf.add(parquet_write=t_save_el)
        logging.info(
            "TILE %s SAVED  rows=%d  %.1fMB  %.2fMB/s  %.2fs  → %s",
            tile, n_total_pts, sz_mb,
            sz_mb / t_save_el if t_save_el > 0 else 0, t_save_el, tile_out,
        )

        total_points_written += n_total_pts
        total_tiles_done     += 1

        logging.info(
            "TILE %s COMPLETE  elapsed=%.1fs  scenes=%d  polys_with_pts=%d  pts=%d",
            tile, time.perf_counter() - t_tile_start,
            n_scenes, n_poly_pts, n_total_pts,
        )

        del xs_flat, ys_flat, lat_flat, lon_flat, offsets, ndvi_results, df_tile
        cleanup_memory()

    # ------------------------------------------------------------------
    # 6. Aggregate → 4 GeoParquet output files
    # ------------------------------------------------------------------
    logging.info("=" * 70)
    logging.info("AGGREGATE START")
    t0 = time.perf_counter()
    aggregate_ndvi(province, output_dir, tree)
    logging.info("AGGREGATE elapsed=%.1fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # 7. Cleanup intermediate tile NDVI files
    # ------------------------------------------------------------------
    interim = glob.glob(
        os.path.join(output_dir, province, f"_ndvi_tile_{tree}_*.parquet"))
    if interim:
        logging.info("CLEANUP: removing %d intermediate tile parquets", len(interim))
        for f in interim:
            try:
                os.remove(f)
            except Exception as exc:
                logging.warning("CLEANUP: could not remove %s: %s", f, exc)

    # ------------------------------------------------------------------
    # 8. Validate all 4 output files were produced
    # ------------------------------------------------------------------
    province_dir = os.path.join(output_dir, province)
    required_outputs = [
        os.path.join(province_dir, "ndvi_mean.parquet"),
        os.path.join(province_dir, "ndvi_max.parquet"),
        os.path.join(province_dir, "ndvi_min.parquet"),
        os.path.join(province_dir, "ndvi_median.parquet"),
    ]
    missing = [p for p in required_outputs if not os.path.exists(p)]

    # ------------------------------------------------------------------
    # 9. Performance summary
    # ------------------------------------------------------------------
    monitor.stop()
    save_performance_summary(run_start, total_tiles_done, total_points_written)
    logging.info(_perf.summary())

    elapsed_min = (time.perf_counter() - run_start) / 60
    logging.info("=" * 70)

    if missing:
        logging.error(
            "PIPELINE FAILED — %d required output(s) missing:\n  %s",
            len(missing), "\n  ".join(missing),
        )
        sys.exit(1)

    logging.info(
        "PIPELINE COMPLETE  tiles=%d  points=%d  elapsed=%.2fmin",
        total_tiles_done, total_points_written, elapsed_min,
    )
    logging.info("log → %s", log_path)


if __name__ == "__main__":
    main()
