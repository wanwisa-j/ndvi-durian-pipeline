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
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.validation import make_valid

try:
    from shapely import contains_xy as _contains_xy
    _FAST_CONTAINS = True
except ImportError:
    from shapely.prepared import prep as _prep
    _FAST_CONTAINS = False


SPACING = 10


def detect_utm_epsg(gdf: gpd.GeoDataFrame) -> int:
    """Auto-select UTM zone 47N (EPSG:32647) or 48N (EPSG:32648) from polygon centroid lon."""
    centroid_lon = gdf.to_crs(4326).geometry.unary_union.centroid.x
    return 32648 if centroid_lon >= 102.0 else 32647


def snap_coord(v, spacing=10):
    return np.floor(v / spacing) * spacing


def create_grid(poly, spacing=10):
    """Return (x_array, y_array) of grid point centers inside poly (vectorized)."""
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(snap_coord(minx, spacing) + spacing / 2, maxx, spacing)
    ys = np.arange(snap_coord(miny, spacing) + spacing / 2, maxy, spacing)
    if len(xs) == 0 or len(ys) == 0:
        return np.empty(0), np.empty(0)

    xx, yy = np.meshgrid(xs, ys)
    px = xx.ravel()
    py = yy.ravel()

    if _FAST_CONTAINS:
        mask = _contains_xy(poly, px, py)
    else:
        prepared = _prep(poly)
        mask = np.fromiter(
            (prepared.contains(Point(x, y)) for x, y in zip(px, py)),
            dtype=bool, count=len(px),
        )
    return px[mask], py[mask]


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

    # Fix invalid geometries BEFORE anything that needs valid input --
    # detect_utm_epsg()'s unary_union() included. GEOS raises
    # "TopologyException: side location conflict" on self-intersecting /
    # otherwise invalid polygons, so validity must be fixed first, not after.
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    print(f"Valid polygons: {len(gdf):,}")

    target_crs = detect_utm_epsg(gdf)
    print(f"UTM zone: EPSG:{target_crs}")
    gdf = gdf.to_crs(target_crs)

    all_rows     = []
    total_points = 0

    for row in gdf.itertuples():
        plot_id  = row.plot_id
        plot_uid = f"{province}_{plot_id}"
        xs, ys = create_grid(row.geometry, SPACING)

        for point_idx, (px, py) in enumerate(zip(xs, ys)):
            all_rows.append({
                "province":   province,
                "plot_id":    plot_id,
                "plot_uid":   plot_uid,
                "point_id":   point_idx,
                "feature_id": build_feature_id(plot_uid, px, py),
                "x_utm":      px,
                "y_utm":      py,
                "geometry":   Point(px, py),
            })
        total_points += len(xs)

    grid_gdf = gpd.GeoDataFrame(all_rows, crs=f"EPSG:{target_crs}")
    print(f"Total points: {len(grid_gdf):,}")

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
