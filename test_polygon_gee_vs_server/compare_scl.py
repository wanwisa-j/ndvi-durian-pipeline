import pandas as pd
from pyproj import Transformer
from osgeo import gdal
import numpy as np
import ee
import os

PLOT_ID = 1
DATE_FILTER = "2025-12-06"
DATE_COL    = "2025/12/06"

ee.Initialize(project="project-7a0a063b-e83f-4545-a99")

# โหลด parquet ทั้งหมด
meta     = pd.read_parquet("/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare/filter_unmatch/points_meta.parquet")
local_df = pd.read_parquet("/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare/ndvi_local.parquet")
gee_df   = pd.read_parquet("/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare/ndvi_gee.parquet")

# กรอง plot
pts = meta[meta["plot_id"] == PLOT_ID].sort_values("point_id").copy()
print(f"points: {len(pts)}")

# แปลง CRS
transformer = Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
pts["lon"], pts["lat"] = transformer.transform(pts["x_proj"].values, pts["y_proj"].values)

# ---- LOCAL SCL ----
gdal.UseExceptions()
SCL_PATH = "/fs2/sentinel2/tiles/47/P/QQ/S2B_MSIL2A_20251206T034029_N0511_R061_T47PQQ_20251206T055750.SAFE/GRANULE/L2A_T47PQQ_A045704_20251206T035019/IMG_DATA/R20m/T47PQQ_20251206T034029_SCL_20m.jp2"

ds   = gdal.Open(SCL_PATH)
gt   = ds.GetGeoTransform()
band = ds.GetRasterBand(1)

local_scl = []
for _, row in pts.iterrows():
    px  = int(np.floor((row["x_proj"] - gt[0]) / gt[1]))
    py  = int(np.floor((row["y_proj"] - gt[3]) / gt[5]))
    arr = band.ReadAsArray(px, py, 1, 1)
    local_scl.append(int(arr[0, 0]) if arr is not None else np.nan)

pts["local_scl"] = local_scl
ds = None

# ---- GEE SCL ----
fc = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([r["lon"], r["lat"]]), {"point_id": int(r["point_id"])})
    for _, r in pts.iterrows()
])

img    = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate(DATE_FILTER, "2025-12-07")
    .filter(ee.Filter.eq("MGRS_TILE", "47PQQ")) 
    .first()
)
scl_fc = img.select("SCL").reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=20)

gee_scl_df = pd.DataFrame([
    {"point_id": f["properties"]["point_id"], "gee_scl": f["properties"].get("first")}
    for f in scl_fc.getInfo()["features"]
])

# ---- merge ----
local_sub = local_df[local_df["plot_id"] == PLOT_ID].sort_values("point_id")
gee_sub   = gee_df[gee_df["plot_id"] == PLOT_ID].sort_values("point_id")

debug_df = (
    pts
    .merge(gee_scl_df, on="point_id", how="left")
    .sort_values("point_id")
)
debug_df["local_ndvi"] = local_sub[DATE_COL].values
debug_df["gee_ndvi"]   = gee_sub[DATE_COL].values

debug_df.to_csv("plot1_20251206_scl_debug.csv", index=False)
print(debug_df[["point_id", "local_scl", "gee_scl", "local_ndvi", "gee_ndvi"]])
print(img.get("PRODUCT_ID").getInfo())
print(img.get("system:index").getInfo())
print(img.select("B4").projection().getInfo())
print(img.select("B8").projection().getInfo())
ndvi = img.normalizedDifference(["B8","B4"])

print(ndvi.projection().getInfo())
#debug_df[["point_id", "local_scl", "gee_scl", "local_ndvi", "gee_ndvi"]].to_csv("local_scl_all.csv", index=False)