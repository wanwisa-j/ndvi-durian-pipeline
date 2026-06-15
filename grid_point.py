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


TARGET_CRS = 32647
SPACING    = 10


def snap_coord(v, spacing=10):
    return np.floor(v / spacing) * spacing


def create_grid(poly, spacing=10):
    minx, miny, maxx, maxy = poly.bounds
    minx = snap_coord(minx, spacing)
    miny = snap_coord(miny, spacing)
    maxx = snap_coord(maxx, spacing)
    maxy = snap_coord(maxy, spacing)

    xs = np.arange(minx, maxx, spacing)
    ys = np.arange(miny, maxy, spacing)

    pts = []
    for x in xs:
        for y in ys:
            px = x + spacing / 2
            py = y + spacing / 2
            p  = Point(px, py)
            try:
                if poly.intersects(p):
                    pts.append(p)
            except Exception:
                continue
    return pts


def build_feature_id(plot_uid, x_utm, y_utm):
    txt = f"{plot_uid}_{int(round(x_utm))}_{int(round(y_utm))}"
    return hashlib.md5(txt.encode()).hexdigest()


def load_polygon_file(polygon_path: Path) -> gpd.GeoDataFrame:
    if polygon_path.suffix.lower() == ".parquet":
        return gpd.read_parquet(polygon_path)
    return gpd.read_file(polygon_path)


def process_polygon_file(polygon_path: Path, grid_dir: Path, overwrite: bool = False):
    province    = polygon_path.stem.upper()
    output_path = grid_dir / f"{province}.parquet"

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

    gdf = gdf.to_crs(TARGET_CRS)
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    print(f"Valid polygons: {len(gdf):,}")

    all_rows     = []
    total_points = 0

    for row in gdf.itertuples():
        plot_id  = row.plot_id
        plot_uid = f"{province}_{plot_id}"
        pts      = create_grid(row.geometry, SPACING)

        for point_idx, p in enumerate(pts):
            all_rows.append({
                "province":   province,
                "plot_id":    plot_id,
                "plot_uid":   plot_uid,
                "point_id":   point_idx,
                "feature_id": build_feature_id(plot_uid, p.x, p.y),
                "x_utm":      p.x,
                "y_utm":      p.y,
                "geometry":   p,
            })
        total_points += len(pts)

    grid_gdf = gpd.GeoDataFrame(all_rows, crs=f"EPSG:{TARGET_CRS}")
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
    parser.add_argument(
        "--polygon-dir", required=True,
        help="directory ที่มีไฟล์ polygon (.parquet หรือ .gpkg)"
    )
    parser.add_argument(
        "--grid-dir", required=True,
        help="directory สำหรับ output grid point parquet"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="รันใหม่แม้ output file จะมีอยู่แล้ว"
    )
    args = parser.parse_args()

    polygon_dir = Path(args.polygon_dir)
    grid_dir    = Path(args.grid_dir)

    if not polygon_dir.exists():
        raise FileNotFoundError(f"ไม่พบ polygon-dir: {polygon_dir}")

    grid_dir.mkdir(parents=True, exist_ok=True)

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
