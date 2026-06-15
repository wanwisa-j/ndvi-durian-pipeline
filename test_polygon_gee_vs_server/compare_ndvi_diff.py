import pandas as pd
import numpy as np

PLOT_ID = 1
DATE = "2025/12/26"

local_df = pd.read_parquet(
    "/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare/filter_unmatch/ndvi_local_matched.parquet"
)

gee_df = pd.read_parquet(
    "/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare/filter_unmatch/ndvi_gee_matched.parquet"
)

local_sub = (
    local_df[local_df["plot_id"] == PLOT_ID]
    .sort_values("point_id")
)

gee_sub = (
    gee_df[gee_df["plot_id"] == PLOT_ID]
    .sort_values("point_id")
)

local_col = local_sub[DATE].values.astype(np.float32)
gee_col   = gee_sub[DATE].values.astype(np.float32)

print("LOCAL")
print("valid =", np.isfinite(local_col).sum())
print("mean  =", np.nanmean(local_col))
print("median=", np.nanmedian(local_col))

print()

print("GEE")
print("valid =", np.isfinite(gee_col).sum())
print("mean  =", np.nanmean(gee_col))
print("median=", np.nanmedian(gee_col))

df = pd.DataFrame({
    "point_id": local_sub["point_id"].values,
    "local": local_col,
    "gee": gee_col,
})

df["status"] = np.select(
    [
        np.isfinite(df["local"]) & np.isfinite(df["gee"]),
        np.isfinite(df["local"]) & df["gee"].isna(),
        df["local"].isna() & np.isfinite(df["gee"]),
    ],
    [
        "both",
        "local_only",
        "gee_only",
    ],
    default="none"
)

both_df = df[df["status"] == "both"].copy()

print(
    both_df[
        ["point_id", "local", "gee"]
    ].head(10)
)

print("\nStatus count")
print(df["status"].value_counts())

print("\nGEE ONLY")
print(
    df[df["status"] == "gee_only"]
    .sort_values("gee")
)

print("\nLOCAL ONLY")
print(
    df[df["status"] == "local_only"]
    .sort_values("local")
)