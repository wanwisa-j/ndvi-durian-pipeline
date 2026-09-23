#!/usr/bin/env python3
"""
Generate Sentinel-2 Grid Points
================================
สร้าง point grid 10m aligned กับ Sentinel-2 pixels สำหรับแต่ละ polygon

Input:
    ไฟล์ polygon ใน --polygon-dir  (.parquet หรือ .gpkg)

Output:
    ไฟล์ .parquet ใน --grid-dir  ชื่อตาม stem ของ input (uppercase)

Columns output:
    province, plot_id, plot_uid, point_id, feature_id,
    x_utm, y_utm, lon, lat, geometry

ตัวอย่าง:
    python3 grid_point.py \\
        --polygon-dir /data/polygons \\
        --grid-dir    /data/grid_points

Notes:
    - Grid aligned กับ Sentinel-2 10m pixels (EPSG:32647)
    - Point วางที่ pixel center
    - point_id reset ภายในแต่ละ plot
    - feature_id stable ทุกครั้งที่รัน (hash จาก plot_uid +좌표)
"""

import argparse
import hashlib
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.validation import make_valid

try:
    from shapely import contains_xy as _contains_xy
    try:
        from shapely import prepare as _prepare
    except ImportError:                      # shapely <2.0 without prepare()
        def _prepare(_geom):                 # no-op fallback
            return None
    _FAST_CONTAINS = True
except ImportError:
    from shapely.prepared import prep as _prep
    _FAST_CONTAINS = False


SPACING = 10

# Per-polygon watchdog: one pathological geometry must not hang the whole run
# for hours. SIGALRM only fires on the main thread, which is where this script
# runs (invoked as __main__, single-threaded).
GRID_TIMEOUT_SEC = 60


@contextmanager
def _time_limit(seconds: int, msg: str):
    def _handler(signum, frame):
        raise TimeoutError(msg)

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def detect_utm_epsg(gdf: gpd.GeoDataFrame) -> int:
    """Auto-select UTM zone 47N (EPSG:32647) or 48N (EPSG:32648) from polygon centroid lon.

    Uses the pre-computed ``centroid_x`` column when present (canopy inference
    output has it) so we never pay for a ``unary_union`` over millions of
    vertices just to pick a CRS. Falls back to a cheap representative_point().
    """
    if "centroid_x" in gdf.columns and gdf["centroid_x"].notna().any():
        centroid_lon = float(np.nanmean(gdf["centroid_x"].to_numpy(dtype="float64")))
    else:
        pts = gdf.geometry.representative_point()
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            pts = pts.to_crs(4326)
        centroid_lon = float(pts.x.mean())
    return 32648 if centroid_lon >= 102.0 else 32647


def snap_coord(v, spacing=10):
    return np.floor(v / spacing) * spacing


def _grid_one(geom, spacing):
    """Grid a single (non-multi) geometry's bbox and keep interior centers."""
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(snap_coord(minx, spacing) + spacing / 2, maxx, spacing)
    ys = np.arange(snap_coord(miny, spacing) + spacing / 2, maxy, spacing)
    if len(xs) == 0 or len(ys) == 0:
        return np.empty(0), np.empty(0)

    xx, yy = np.meshgrid(xs, ys)
    px = xx.ravel()
    py = yy.ravel()

    if _FAST_CONTAINS:
        _prepare(geom)                       # build edge STRtree once, then query
        mask = _contains_xy(geom, px, py)
    else:
        prepared = _prep(geom)
        mask = np.fromiter(
            (prepared.contains(Point(x, y)) for x, y in zip(px, py)),
            dtype=bool, count=len(px),
        )
    return px[mask], py[mask]


def _polygon_parts(geom):
    """Flatten any geometry to a flat list of simple Polygon parts.

    make_valid() can hand back a GeometryCollection (polygon + stray line
    spurs); contains_xy() on that is undefined, so keep only Polygon area.
    """
    gt = geom.geom_type
    if gt == "Polygon":
        return [geom]
    if gt in ("MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(_polygon_parts(g))
        return out
    return []


def create_grid(poly, spacing=10):
    """Return (x_array, y_array) of grid point centers inside poly (vectorized).

    Gridded part-by-part: a sparse MultiPolygon (canopy blobs spread across a
    district) can have a bbox 20x its real extent, so gridding the whole
    envelope wastes millions of point-in-polygon tests on empty space.
    Point set is identical to gridding the whole geometry; only the order in
    which points come out (hence within-plot point_id) differs for multipolygons.
    """
    xa, ya = [], []
    for part in _polygon_parts(poly):
        if part.is_empty:
            continue
        gx, gy = _grid_one(part, spacing)
        if gx.size:
            xa.append(gx)
            ya.append(gy)
    if not xa:
        return np.empty(0), np.empty(0)
    return np.concatenate(xa), np.concatenate(ya)


def build_feature_id(plot_uid, x_utm, y_utm):
    txt = f"{plot_uid}_{int(round(x_utm))}_{int(round(y_utm))}"
    return hashlib.md5(txt.encode()).hexdigest()


def load_polygon_file(polygon_path: Path) -> gpd.GeoDataFrame:
    if polygon_path.suffix.lower() == ".parquet":
        return gpd.read_parquet(polygon_path)
    return gpd.read_file(polygon_path)


def process_polygon_file(
    polygon_path: Path,
    grid_dir: Path,
    overwrite: bool = False,
    output_file: Path | None = None,
):
    province    = polygon_path.stem.upper()
    output_path = output_file if output_file is not None else grid_dir / f"{polygon_path.stem}_grid.parquet"

    if output_path.exists() and not overwrite:
        print(f"SKIP {province} (already exists) — ใช้ --overwrite เพื่อรันใหม่")
        return

    print("=" * 60)
    print(f"Province: {province}")
    print("=" * 60)

    gdf = load_polygon_file(polygon_path)
    print(f"Polygons: {len(gdf):,}")
    gdf = gdf.reset_index(drop=True)

    if "plot_id" not in gdf.columns:
        print("plot_id not found -> สร้าง sequential id")
        gdf["plot_id"] = np.arange(len(gdf))

    # Fix invalid geometries BEFORE anything that needs valid input.
    # GEOS raises "TopologyException: side location conflict" on self-
    # intersecting / otherwise invalid polygons, so validity must be fixed
    # first, not after.
    #
    # Drop null geometries FIRST: make_valid() assumes every value is a real
    # shapely geometry and crashes with AttributeError on a bare None (a
    # source row with no polygon at all) instead of passing it through --
    # the null/empty filter below only catches what make_valid() itself
    # produces, it never runs if make_valid() has already raised on a
    # pre-existing None.
    null_geom = gdf.geometry.isnull().sum()
    if null_geom:
        print(f"{null_geom} row(s) with no geometry -> dropped before make_valid")
        gdf = gdf[gdf.geometry.notnull()]
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    # On older GEOS make_valid() can still leave a geometry invalid; a stray
    # invalid ring is exactly what can send contains_xy() into a multi-hour
    # spin, so scrub it with buffer(0) as a last resort.
    still_bad = ~gdf.geometry.is_valid.to_numpy()
    if still_bad.any():
        print(f"make_valid left {int(still_bad.sum())} invalid -> buffer(0)")
        gdf.loc[still_bad, "geometry"] = gdf.loc[still_bad, "geometry"].buffer(0)
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    gdf = gdf.reset_index(drop=True)
    print(f"Valid polygons: {len(gdf):,}")

    target_crs = detect_utm_epsg(gdf)
    print(f"UTM zone: EPSG:{target_crs}")
    gdf = gdf.to_crs(target_crs)

    n         = len(gdf)
    plot_ids  = gdf["plot_id"].to_numpy()
    geoms     = gdf.geometry.to_numpy()

    x_parts, y_parts, pid_parts, ptid_parts = [], [], [], []
    total_points = 0
    skipped      = []
    t0           = time.time()

    for i in range(n):
        if i % 250 == 0:
            print(f"  {i:>6}/{n}  points={total_points:,}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        try:
            with _time_limit(GRID_TIMEOUT_SEC,
                             f"grid timeout i={i} plot_id={plot_ids[i]}"):
                xs, ys = create_grid(geoms[i], SPACING)
        except TimeoutError as e:
            print(f"  SKIP {e}", flush=True)
            skipped.append(int(plot_ids[i]))
            continue

        k = xs.size
        if k == 0:
            continue
        x_parts.append(xs)
        y_parts.append(ys)
        pid_parts.append(np.full(k, plot_ids[i]))
        ptid_parts.append(np.arange(k))
        total_points += k

    if skipped:
        head = skipped[:20]
        print(f"WARNING: {len(skipped)} polygon(s) skipped on {GRID_TIMEOUT_SEC}s "
              f"timeout: plot_id={head}{' ...' if len(skipped) > 20 else ''}")

    if not x_parts:
        print("No grid points produced — nothing to save")
        return

    x        = np.concatenate(x_parts)
    y        = np.concatenate(y_parts)
    plot_id_arr  = np.concatenate(pid_parts)
    point_id_arr = np.concatenate(ptid_parts)
    print(f"Total points: {len(x):,}  ({time.time() - t0:.0f}s)")

    plot_uid_arr = [f"{province}_{p}" for p in plot_id_arr]
    grid_gdf = gpd.GeoDataFrame(
        {
            "province":   province,
            "plot_id":    plot_id_arr,
            "plot_uid":   plot_uid_arr,
            "point_id":   point_id_arr,
            "feature_id": [build_feature_id(u, xx, yy)
                           for u, xx, yy in zip(plot_uid_arr, x, y)],
            "x_utm":      x,
            "y_utm":      y,
        },
        geometry=gpd.points_from_xy(x, y),
        crs=f"EPSG:{target_crs}",
    )

    grid_wgs84       = grid_gdf.to_crs(4326)
    grid_gdf["lon"]  = grid_wgs84.geometry.x
    grid_gdf["lat"]  = grid_wgs84.geometry.y

    cols = ["province", "plot_id", "plot_uid", "point_id", "feature_id",
            "x_utm", "y_utm", "lon", "lat", "geometry"]
    grid_gdf = grid_gdf[cols]

    print(f"Saving -> {output_path}")
    grid_gdf.to_parquet(output_path, index=False)
    print(f"Saved {len(grid_gdf):,} points\n")


def main():
    parser = argparse.ArgumentParser(
        description="สร้าง Sentinel-2 10m grid points จาก polygon file"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--polygon-file",
        help="ไฟล์ polygon ไฟล์เดียว (.parquet หรือ .gpkg)",
    )
    source.add_argument(
        "--polygon-dir",
        help="directory ที่มีไฟล์ polygon (.parquet หรือ .gpkg)",
    )
    parser.add_argument(
        "--grid-dir", required=True,
        help="directory สำหรับ output grid point parquet"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="รันใหม่แม้ output file จะมีอยู่แล้ว"
    )
    parser.add_argument(
        "--output-file", default=None,
        help="Override output .parquet path (default: {grid-dir}/{stem}_grid.parquet)",
    )
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)

    output_file = Path(args.output_file) if args.output_file else None

    if args.polygon_file:
        polygon_path = Path(args.polygon_file)
        if not polygon_path.exists():
            raise FileNotFoundError(f"ไม่พบ polygon-file: {polygon_path}")
        process_polygon_file(polygon_path, grid_dir, overwrite=args.overwrite, output_file=output_file)
    else:
        polygon_dir = Path(args.polygon_dir)
        if not polygon_dir.exists():
            raise FileNotFoundError(f"ไม่พบ polygon-dir: {polygon_dir}")
        files = sorted(polygon_dir.glob("*.parquet")) + sorted(polygon_dir.glob("*.gpkg"))
        if not files:
            raise FileNotFoundError(f"ไม่พบไฟล์ .parquet หรือ .gpkg ใน {polygon_dir}")
        print(f"พบ {len(files)} polygon file(s)")
        for f in files:
            process_polygon_file(f, grid_dir, overwrite=args.overwrite)

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
