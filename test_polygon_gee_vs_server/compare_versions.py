#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_versions.py
===================
เทียบ ndvi_local.parquet (pipeline เก่า) กับ ndvi_mean.parquet (pipeline ใหม่)
โดยแปลงทั้งคู่เป็น polygon-level monthly mean แล้วพล็อตบนกราฟเดียวกัน

ndvi_local.parquet  : rows=points, cols=YYYY/MM/DD  (individual S2 dates)
ndvi_mean.parquet   : rows=points, cols=YYYY-MM      (monthly mean, v2 pipeline)

Matching mode (MATCH_MONTHS=True):
  เปรียบเทียบเฉพาะเดือนที่ทั้งสอง version มีค่า (intersection)
  → กรองออก: เดือนที่ v1 NaN ทั้งหมด หรือ v2 NaN ทั้งหมด

ผลลัพธ์:
  compare_mean_plot<N>.png  — per-polygon graph พร้อม individual dates ของ v1
  compare_mean_all.png      — all polygons in one figure
  diff_summary.csv          — month-level diff + จำนวน scenes ที่ v1 ใช้
"""

# =============================================================================
# CONFIG
# =============================================================================

LOCAL_PARQUET    = "/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare/ndvi_local.parquet"
PIPELINE_PARQUET = "/fs2/wanwisa/durian/outputs/test/ndvi_mean.parquet"
OUTPUT_DIR       = "/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare/version_compare"
PROVINCE         = "test_rayong"

# True  → ใช้เฉพาะเดือนที่ทั้งคู่มีค่า (intersection)
# False → ใช้ทุกเดือนที่มีใน source ใดก็ได้ (union)
MATCH_MONTHS = True

# =============================================================================
# IMPORTS
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# =============================================================================
# LOAD & PREPARE
# =============================================================================

def load_local(path: str):
    """โหลด ndvi_local.parquet → monthly mean per polygon
    Returns:
      df_monthly   : DataFrame(plot_id, YYYY-MM cols)
      month_map    : dict YYYY-MM → [YYYY/MM/DD cols]
      raw_df       : original per-point DataFrame (for individual-date plots)
    """
    df = pd.read_parquet(path)
    date_cols = [c for c in df.columns if c not in ("plot_id", "point_id")]

    parsed    = pd.to_datetime(date_cols, format="%Y/%m/%d")
    month_map = {}
    for col, ts in zip(date_cols, parsed):
        m = ts.strftime("%Y-%m")
        month_map.setdefault(m, []).append(col)

    months_sorted = sorted(month_map.keys())
    records = []
    for pid in sorted(df["plot_id"].unique()):
        sub = df[df["plot_id"] == pid]
        row = {"plot_id": pid}
        for m in months_sorted:
            vals = sub[month_map[m]].values.astype(np.float32).ravel()
            row[m] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
        records.append(row)

    return pd.DataFrame(records), months_sorted, month_map, df


def load_pipeline(path: str):
    """โหลด ndvi_mean.parquet → monthly mean per polygon"""
    df = pd.read_parquet(path)
    meta = {"plot_id", "point_id", "lat", "lon", "geometry"}
    month_cols = sorted(c for c in df.columns if c not in meta)

    records = []
    for pid in sorted(df["plot_id"].unique()):
        sub = df[df["plot_id"] == pid]
        row = {"plot_id": pid}
        for m in month_cols:
            vals = sub[m].values.astype(np.float32)
            row[m] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
        records.append(row)

    return pd.DataFrame(records), month_cols


def intersect_months(local_df, pipe_df, all_months, plot_ids):
    """คืน list เดือนที่ทั้งสอง version มีค่า (ทุก plot_id ต้องมีอย่างน้อยหนึ่งเพียง)."""
    matched = []
    for m in all_months:
        local_has = any(
            np.isfinite(local_df[local_df.plot_id == p][m].values[0])
            if m in local_df.columns and not local_df[local_df.plot_id == p].empty else False
            for p in plot_ids
        )
        pipe_has = any(
            np.isfinite(pipe_df[pipe_df.plot_id == p][m].values[0])
            if m in pipe_df.columns and not pipe_df[pipe_df.plot_id == p].empty else False
            for p in plot_ids
        )
        if local_has and pipe_has:
            matched.append(m)
    return matched


# =============================================================================
# PLOT: single polygon (with individual v1 dates as scatter)
# =============================================================================

def plot_polygon(pid, local_row, pipe_row, months, month_map, raw_local_df,
                 output_dir, province=""):
    dates = pd.to_datetime(months, format="%Y-%m")

    v1 = np.array([local_row.get(m, np.nan) for m in months], dtype=np.float32)
    v2 = np.array([pipe_row.get(m, np.nan)  for m in months], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(14, 4))

    # individual v1 dates as small dots (show scene-level variance)
    sub = raw_local_df[raw_local_df["plot_id"] == pid]
    for m in months:
        if m not in month_map:
            continue
        for col in month_map[m]:
            pt = pd.to_datetime(col, format="%Y/%m/%d")
            vals = sub[col].values.astype(np.float32)
            scene_mean = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
            if np.isfinite(scene_mean):
                ax.scatter(pt, scene_mean, s=18, color="#1D9E75", alpha=0.55,
                           zorder=2, marker="x")

    valid1 = np.isfinite(v1)
    valid2 = np.isfinite(v2)

    if valid1.sum() > 1:
        ax.plot(dates[valid1], v1[valid1],
                color="#0B6E4F", linewidth=1.8, marker="o", markersize=4,
                label="v1 monthly mean")
    if valid2.sum() > 1:
        ax.plot(dates[valid2], v2[valid2],
                color="#C0392B", linewidth=1.8, marker="s", markersize=4,
                linestyle="--", label="v2 pipeline mean")

    both = valid1 & valid2
    if both.sum() > 0:
        ax.fill_between(dates[both], v1[both], v2[both],
                        alpha=0.12, color="#8E44AD", label="diff region")

    legend_elements = [
        Line2D([0], [0], color="#1D9E75", alpha=0.7, lw=0, marker="x",
               markersize=6, label="v1 scene-level mean"),
        Line2D([0], [0], color="#0B6E4F", lw=2, marker="o", markersize=4,
               label="v1 monthly mean"),
        Line2D([0], [0], color="#C0392B", lw=2, marker="s", markersize=4,
               linestyle="--", label="v2 pipeline mean"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    title = f"NDVI Monthly Mean — Plot {pid}"
    if province:
        title += f"  ({province})"
    mode = "matched months only" if MATCH_MONTHS else "all months"
    title += f"  [{mode}]"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("NDVI")
    ax.set_ylim(-0.15, 1.05)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle=":", alpha=0.25)
    fig.tight_layout()

    suffix = "_matched" if MATCH_MONTHS else ""
    out_path = os.path.join(output_dir, f"compare_mean_plot{pid:03d}{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# PLOT: all polygons (subplots)
# =============================================================================

def plot_all(local_df, pipe_df, months, plot_ids, output_dir, province=""):
    dates   = pd.to_datetime(months, format="%Y-%m")
    n_plots = len(plot_ids)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, pid in zip(axes, plot_ids):
        lr = local_df[local_df.plot_id == pid].iloc[0].to_dict() if not local_df[local_df.plot_id == pid].empty else {}
        pr = pipe_df[pipe_df.plot_id   == pid].iloc[0].to_dict() if not pipe_df[pipe_df.plot_id   == pid].empty else {}

        v1 = np.array([lr.get(m, np.nan) for m in months], dtype=np.float32)
        v2 = np.array([pr.get(m, np.nan) for m in months], dtype=np.float32)
        valid1 = np.isfinite(v1)
        valid2 = np.isfinite(v2)

        if valid1.sum() > 1:
            ax.plot(dates[valid1], v1[valid1],
                    color="#0B6E4F", linewidth=1.8, marker="o", markersize=3,
                    label="v1 local")
        if valid2.sum() > 1:
            ax.plot(dates[valid2], v2[valid2],
                    color="#C0392B", linewidth=1.8, marker="s", markersize=3,
                    linestyle="--", label="v2 pipeline")

        both = valid1 & valid2
        if both.sum() > 0:
            ax.fill_between(dates[both], v1[both], v2[both],
                            alpha=0.12, color="#8E44AD")

        ax.set_title(f"Plot {pid}", fontsize=11)
        ax.set_ylabel("NDVI")
        ax.set_ylim(-0.15, 1.05)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.grid(axis="x", linestyle=":", alpha=0.25)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    axes[-1].set_xlabel("Month")

    mode = "matched months only" if MATCH_MONTHS else "all months"
    suptitle = f"NDVI Monthly Mean: v1 vs v2  [{mode}]"
    if province:
        suptitle += f"  — {province}"
    fig.suptitle(suptitle, fontsize=13, y=1.005)
    fig.tight_layout()

    suffix = "_matched" if MATCH_MONTHS else ""
    out_path = os.path.join(output_dir, f"compare_mean_all{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# REPORT: diff + scene breakdown
# =============================================================================

def report_diff(local_df, pipe_df, months, plot_ids, month_map, raw_local_df, output_dir):
    records = []
    for pid in plot_ids:
        lr = local_df[local_df.plot_id == pid].iloc[0].to_dict() if not local_df[local_df.plot_id == pid].empty else {}
        pr = pipe_df[pipe_df.plot_id   == pid].iloc[0].to_dict() if not pipe_df[pipe_df.plot_id   == pid].empty else {}
        sub = raw_local_df[raw_local_df["plot_id"] == pid]

        for m in months:
            v1, v2 = lr.get(m, np.nan), pr.get(m, np.nan)
            if not (np.isfinite(v1) and np.isfinite(v2)):
                continue

            # how many individual v1 scene dates are in this month & are non-NaN
            dates_in = month_map.get(m, [])
            scene_means = []
            for col in dates_in:
                sm = np.nanmean(sub[col].values.astype(np.float32))
                scene_means.append(round(float(sm), 4) if np.isfinite(sm) else np.nan)
            n_valid_scenes = sum(np.isfinite(x) for x in scene_means)

            records.append({
                "plot_id":       pid,
                "month":         m,
                "v1_local":      round(float(v1), 5),
                "v2_pipe":       round(float(v2), 5),
                "diff":          round(float(v2 - v1), 5),
                "abs_diff":      round(abs(float(v2 - v1)), 5),
                "v1_n_scenes":   n_valid_scenes,
                "v1_scene_vals": str(scene_means),
            })

    if not records:
        print("  ไม่มีเดือนที่เทียบได้")
        return

    df = pd.DataFrame(records).sort_values("abs_diff", ascending=False)
    suffix = "_matched" if MATCH_MONTHS else ""
    csv_path = os.path.join(output_dir, f"diff_summary{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}  ({len(df)} matched months)")

    print(f"\n  Overall  mean_abs={df.abs_diff.mean():.4f}  max_abs={df.abs_diff.max():.4f}")
    print(f"\n  Top 15 largest |v2 - v1|  (v1_n_scenes = จำนวน scene ที่ v1 รวมเข้าไป):")
    show = df[["plot_id","month","v1_local","v2_pipe","diff","v1_n_scenes","v1_scene_vals"]].head(15)
    print(show.to_string(index=False))

    print(f"\n  Per-plot summary:")
    print(df.groupby("plot_id").agg(
        n_months=("month", "count"),
        mean_abs=("abs_diff", "mean"),
        max_abs=("abs_diff", "max"),
    ).to_string())


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading v1 (ndvi_local.parquet)...")
    local_df, local_months, month_map, raw_local = load_local(LOCAL_PARQUET)
    print(f"  plots: {sorted(local_df.plot_id.unique())}, months: {len(local_months)}")

    print("Loading v2 (ndvi_mean.parquet)...")
    pipe_df, pipe_months = load_pipeline(PIPELINE_PARQUET)
    print(f"  plots: {sorted(pipe_df.plot_id.unique())}, months: {len(pipe_months)}")

    plot_ids   = sorted(set(local_df.plot_id.unique()) | set(pipe_df.plot_id.unique()))
    all_months = sorted(set(local_months) | set(pipe_months))

    if MATCH_MONTHS:
        use_months = intersect_months(local_df, pipe_df, all_months, plot_ids)
        print(f"\nMatch mode ON: {len(all_months)} total → {len(use_months)} matched months")
    else:
        use_months = all_months
        print(f"\nMatch mode OFF: using all {len(all_months)} months (union)")

    print(f"Month range: {use_months[0]} → {use_months[-1]}")

    print("\n[1] Per-polygon plots...")
    for pid in plot_ids:
        lr = local_df[local_df.plot_id == pid].iloc[0].to_dict() if not local_df[local_df.plot_id == pid].empty else {}
        pr = pipe_df[pipe_df.plot_id   == pid].iloc[0].to_dict() if not pipe_df[pipe_df.plot_id   == pid].empty else {}
        plot_polygon(pid, lr, pr, use_months, month_map, raw_local, OUTPUT_DIR, PROVINCE)

    print("\n[2] Combined subplot...")
    plot_all(local_df, pipe_df, use_months, plot_ids, OUTPUT_DIR, PROVINCE)

    print("\n[3] Diff report...")
    report_diff(local_df, pipe_df, use_months, plot_ids, month_map, raw_local, OUTPUT_DIR)

    print(f"\nDone. All outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
