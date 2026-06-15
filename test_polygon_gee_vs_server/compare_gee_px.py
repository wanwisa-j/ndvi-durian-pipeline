import ee
import pandas as pd
import numpy as np
from pyproj import Transformer
from osgeo import gdal

ee.Initialize(project="project-7a0a063b-e83f-4545-a99")

# ==================================================
# INPUT
# ==================================================
PLOT_ID = 1
DATE = "2025-12-06"

META_FILE = "/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare/filter_unmatch/points_meta.parquet"

LOCAL_B4 = "/fs2/sentinel2/tiles/47/P/QQ/S2B_MSIL2A_20251206T034029_N0511_R061_T47PQQ_20251206T055750.SAFE/GRANULE/L2A_T47PQQ_A045704_20251206T035019/IMG_DATA/R10m/T47PQQ_20251206T034029_B04_10m.jp2"
LOCAL_B8 = "/fs2/sentinel2/tiles/47/P/QQ/S2B_MSIL2A_20251206T034029_N0511_R061_T47PQQ_20251206T055750.SAFE/GRANULE/L2A_T47PQQ_A045704_20251206T035019/IMG_DATA/R10m/T47PQQ_20251206T034029_B08_10m.jp2"
LOCAL_SCL = "/fs2/sentinel2/tiles/47/P/QQ/S2B_MSIL2A_20251206T034029_N0511_R061_T47PQQ_20251206T055750.SAFE/GRANULE/L2A_T47PQQ_A045704_20251206T035019/IMG_DATA/R20m/T47PQQ_20251206T034029_SCL_20m.jp2"

# ==================================================
# LOAD POINTS
# ==================================================
meta = pd.read_parquet(META_FILE)
print("B4 :", LOCAL_B4)
print("B8 :", LOCAL_B8)
print("SCL:", LOCAL_SCL)

print(gdal.Open(LOCAL_B4))
print(gdal.Open(LOCAL_B8))
print(gdal.Open(LOCAL_SCL))
pts = (
    meta[meta["plot_id"] == PLOT_ID]
    .sort_values("point_id")
    .copy()
)

# ==================================================
# UTM -> WGS84
# ==================================================
transformer = Transformer.from_crs(
    "EPSG:32647",
    "EPSG:4326",
    always_xy=True
)

pts["lon"], pts["lat"] = transformer.transform(
    pts["x_proj"].values,
    pts["y_proj"].values
)

# ==================================================
# LOCAL SAMPLE
# ==================================================
def sample_ds(ds, xs, ys):

    gt = ds.GetGeoTransform()

    px = np.floor(
        (xs - gt[0]) / gt[1]
    ).astype(int)

    py = np.floor(
        (ys - gt[3]) / gt[5]
    ).astype(int)

    arr = ds.GetRasterBand(1).ReadAsArray()

    vals = arr[py, px]

    nodata = ds.GetRasterBand(1).GetNoDataValue()

    if nodata is not None:
        vals = vals.astype(float)
        vals[vals == nodata] = np.nan

    return vals


ds_b4 = gdal.Open(LOCAL_B4)
ds_b8 = gdal.Open(LOCAL_B8)
ds_scl = gdal.Open(LOCAL_SCL)

xs = pts["x_proj"].values
ys = pts["y_proj"].values

local_b4 = sample_ds(ds_b4, xs, ys).astype(float)
local_b8 = sample_ds(ds_b8, xs, ys).astype(float)
local_scl = sample_ds(ds_scl, xs, ys)

# baseline correction
acq_date = pd.Timestamp(DATE)

if acq_date > pd.Timestamp("2022-01-25"):
    local_b4 = local_b4 - 1000
    local_b8 = local_b8 - 1000

local_ndvi = (
    (local_b8 - local_b4)
    / (local_b8 + local_b4)
)

# ==================================================
# GEE IMAGE
# ==================================================
img = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate(DATE, pd.Timestamp(DATE).strftime("%Y-%m-%d"))
)

img = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate(
        DATE,
        (pd.Timestamp(DATE) + pd.Timedelta(days=1))
        .strftime("%Y-%m-%d")
    )
    .filter(ee.Filter.eq("MGRS_TILE", "47PQQ"))
    .first()
)

ndvi = (
    img.normalizedDifference(["B8", "B4"])
    .rename("NDVI")
)

stack = ee.Image.cat([
    img.select("SCL"),
    img.select("B4"),
    img.select("B8"),
    ndvi
])

# ==================================================
# FEATURE COLLECTION
# ==================================================
features = []

for _, r in pts.iterrows():

    features.append(
        ee.Feature(
            ee.Geometry.Point(
                [float(r["lon"]), float(r["lat"])]
            ),
            {
                "point_id": int(r["point_id"])
            }
        )
    )

fc = ee.FeatureCollection(features)

sample_fc = stack.reduceRegions(
    collection=fc,
    reducer=ee.Reducer.first(),
    scale=10
)

# ==================================================
# GEE RESULT
# ==================================================
rows = []

for f in sample_fc.getInfo()["features"]:

    p = f["properties"]

    rows.append({
        "point_id": p["point_id"],
        "gee_scl": p.get("SCL"),
        "gee_b4": p.get("B4"),
        "gee_b8": p.get("B8"),
        "gee_ndvi": p.get("NDVI"),
    })

gee_df = (
    pd.DataFrame(rows)
    .sort_values("point_id")
)

# ==================================================
# MERGE
# ==================================================
local_df = pd.DataFrame({
    "point_id": pts["point_id"].values,
    "local_scl": local_scl,
    "local_b4": local_b4,
    "local_b8": local_b8,
    "local_ndvi": local_ndvi,
})

out = (
    gee_df.merge(
        local_df,
        on="point_id",
        how="left"
    )
)

out = out[[
    "point_id",
    "gee_scl",
    "gee_b4",
    "gee_b8",
    "gee_ndvi",
    "local_scl",
    "local_b4",
    "local_b8",
    "local_ndvi",
]]

out.to_csv(
    f"compare_{DATE}.csv",
    index=False
)

print(out.head(30))