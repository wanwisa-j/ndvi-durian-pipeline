#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ndvi.py
============
โหลด ndvi_local.parquet / ndvi_gee.parquet / points_meta.parquet
แล้วสร้าง PNG ดังนี้:

  plot_<N>_points.png   — NDVI รายพอยต์ทุกจุด (เส้น local + เส้น gee ต่อ point)
  plot_<N>_mean.png     — เส้น mean/median รายแปลง local vs gee (ทุก polygon รวมในรูปเดียว)

เงื่อนไข:
  - วันที่ x-axis = วันที่จาก local เท่านั้น (GEE ใช้วันเดียวกัน)
  - สีเส้นแยกตาม source (local/gee) + plot_id
  - บันทึกเป็น PNG ความละเอียด 150 dpi
"""

# ======================================================================================
# CONFIGURATION  —  ให้ตรงกับ collect_ndvi.py
# ======================================================================================

OUTPUT_DIR = "/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare_op"
PROVINCE   = "rayong"

# ======================================================================================
# IMPORTS
# ======================================================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

# ======================================================================================
# HELPERS
# ======================================================================================

def load_parquets(output_dir: str):
    local_df = pd.read_parquet(os.path.join(output_dir, "ndvi_local.parquet"))
    gee_df   = pd.read_parquet(os.path.join(output_dir, "ndvi_gee.parquet"))
    meta_df  = pd.read_parquet(os.path.join(output_dir, "points_meta.parquet"))
    return local_df, gee_df, meta_df


def get_date_columns(df: pd.DataFrame):
    """คืน list ของ column ที่เป็น date (ไม่ใช่ plot_id / point_id)"""
    return sorted([c for c in df.columns if c not in ('plot_id', 'point_id')])


def parse_dates(date_cols: list):
    """แปลง date strings → pd.DatetimeIndex"""
    return pd.to_datetime(date_cols, format='%Y/%m/%d')


def wide_to_matrix(df: pd.DataFrame, plot_id: int, date_cols: list) -> np.ndarray:
    """
    คืน matrix shape (n_points, n_dates) สำหรับ plot_id นั้น
    """
    sub = df[df['plot_id'] == plot_id].sort_values('point_id')
    if sub.empty:
        return np.empty((0, len(date_cols)))
    return sub[date_cols].values.astype(np.float32)


# ======================================================================================
# PLOT 1: per-polygon point-level NDVI
# ======================================================================================

def plot_polygon_points(plot_id: int,
                        local_mat: np.ndarray,
                        gee_mat: np.ndarray,
                        dates: pd.DatetimeIndex,
                        output_dir: str,
                        province: str = ""):
    """
    local_mat / gee_mat: shape (n_points, n_dates)
    เส้น local = เขียว (alpha ต่ำ), เส้น gee = น้ำเงิน (alpha ต่ำ)
    เส้น mean local = เขียวเข้ม, เส้น mean gee = น้ำเงินเข้ม
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    n_local = local_mat.shape[0]
    n_gee   = gee_mat.shape[0]

    # --- point-level lines ---
    for pt in range(n_local):
        y = local_mat[pt]
        valid = np.isfinite(y)
        if valid.sum() > 1:
            ax.plot(dates[valid], y[valid],
                    color='#1D9E75', alpha=0.12, linewidth=0.6)

    for pt in range(n_gee):
        y = gee_mat[pt]
        valid = np.isfinite(y)
        if valid.sum() > 1:
            ax.plot(dates[valid], y[valid],
                    color='#378ADD', alpha=0.12, linewidth=0.6, linestyle='--')

    # --- mean line ---
    with np.errstate(all='ignore'):
        local_mean = np.nanmean(local_mat, axis=0)
        gee_mean   = np.nanmean(gee_mat,   axis=0)

    valid_l = np.isfinite(local_mean)
    valid_g = np.isfinite(gee_mean)

    if valid_l.sum() > 1:
        ax.plot(dates[valid_l], local_mean[valid_l],
                color='#0B6E4F', linewidth=2.0, marker='o', markersize=3,
                label=f'Local mean (n={n_local} pts)')
    if valid_g.sum() > 1:
        ax.plot(dates[valid_g], gee_mean[valid_g],
                color='#1B55A8', linewidth=2.0, marker='s', markersize=3,
                linestyle='--', label=f'GEE mean (n={n_gee} pts)')

    # dummy handles for legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1D9E75', alpha=0.4, lw=1, label='Local points'),
        Line2D([0], [0], color='#378ADD', alpha=0.4, lw=1, linestyle='--', label='GEE points'),
        Line2D([0], [0], color='#0B6E4F', lw=2, marker='o', markersize=4, label=f'Local mean'),
        Line2D([0], [0], color='#1B55A8', lw=2, marker='s', markersize=4,
               linestyle='--', label=f'GEE mean'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

    title = f"NDVI Point-level — Plot {plot_id}"
    if province:
        title += f"  ({province.capitalize()})"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("NDVI")
    ax.set_ylim(-0.1, 1.0)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.grid(axis='x', linestyle=':', alpha=0.25)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"plot_{plot_id:03d}_points.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ======================================================================================
# PLOT 2: summary — mean/median per polygon (all polygons in one figure)
# ======================================================================================

def plot_summary_all_polygons(local_df, gee_df, date_cols, dates, plot_ids,
                               output_dir, province="", stat="mean"):
    """
    layout: nrows = จำนวน polygon, แต่ละ row = 1 polygon
    แต่ละ subplot: local (เส้นทึบ) vs gee (เส้นประ) ใน axes เดียวกัน
    """
    n_plots = len(plot_ids)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    stat_fn = np.nanmean if stat == "mean" else np.nanmedian

    for i, pid in enumerate(plot_ids):
        ax = axes[i]

        local_mat = wide_to_matrix(local_df, pid, date_cols)
        gee_mat   = wide_to_matrix(gee_df,   pid, date_cols)

        local_agg = stat_fn(local_mat, axis=0) if local_mat.shape[0] > 0 else np.full(len(date_cols), np.nan)
        gee_agg   = stat_fn(gee_mat,   axis=0) if gee_mat.shape[0]   > 0 else np.full(len(date_cols), np.nan)

        valid_l = np.isfinite(local_agg)
        valid_g = np.isfinite(gee_agg)

        if valid_l.sum() > 1:
            ax.plot(dates[valid_l], local_agg[valid_l],
                    color='#0B6E4F', linewidth=1.8, marker='o', markersize=3,
                    linestyle='-', label='Local S2')

        if valid_g.sum() > 1:
            ax.plot(dates[valid_g], gee_agg[valid_g],
                    color='#1B55A8', linewidth=1.8, marker='s', markersize=3,
                    linestyle='--', label='GEE S2')

        n_local_pts = local_mat.shape[0]
        n_gee_pts   = gee_mat.shape[0]
        ax.set_title(f"Plot {pid}  —  Local (n={n_local_pts} pts) vs GEE (n={n_gee_pts} pts)", fontsize=11)
        ax.set_ylabel("NDVI")
        ax.set_ylim(-0.1, 1.0)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.grid(axis='x', linestyle=':', alpha=0.25)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    axes[-1].set_xlabel("Date")

    title = f"NDVI {stat.capitalize()} — Local vs GEE"
    if province:
        title += f"  ({province.capitalize()})"
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"summary_{stat}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ======================================================================================
# PLOT 3: per-polygon side-by-side scatter (local vs gee) — optional extra
# ======================================================================================

def plot_scatter_local_vs_gee(local_df: pd.DataFrame,
                               gee_df: pd.DataFrame,
                               plot_ids: list,
                               date_cols: list,
                               output_dir: str):
    """
    Scatter plot local NDVI (x) vs GEE NDVI (y) per polygon
    จุดคือ (point, date) pairs ที่ทั้งสอง source มีค่า
    """
    n = len(plot_ids)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for i, pid in enumerate(plot_ids):
        ax  = axes[i // ncols][i % ncols]
        lm  = wide_to_matrix(local_df, pid, date_cols).ravel()
        gm  = wide_to_matrix(gee_df,   pid, date_cols).ravel()
        valid = np.isfinite(lm) & np.isfinite(gm)
        if valid.sum() > 0:
            ax.scatter(lm[valid], gm[valid], s=4, alpha=0.3, color='#6B2FA0')
            # 1:1 line
            mn = min(lm[valid].min(), gm[valid].min())
            mx = max(lm[valid].max(), gm[valid].max())
            ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1, label='1:1')
        ax.set_xlabel("Local NDVI")
        ax.set_ylabel("GEE NDVI")
        ax.set_title(f"Plot {pid}  (n={valid.sum():,})")
        ax.legend(fontsize=8)

    # hide empty axes
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Scatter: Local vs GEE NDVI per polygon", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "scatter_local_vs_gee.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path

def report_large_diffs(local_df, gee_df, meta_df, date_cols, plot_ids,
                        output_dir, epsg, threshold=0.15, top_n=20):
    """
    เพิ่ม meta_df และ epsg เพื่อแปลง x_proj/y_proj → lat/lon
    """
    from pyproj import Transformer
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    records = []
    for pid in plot_ids:
        local_sub = local_df[local_df['plot_id'] == pid].sort_values('point_id')
        gee_sub   = gee_df[gee_df['plot_id']     == pid].sort_values('point_id')
        meta_sub  = meta_df[meta_df['plot_id']   == pid].sort_values('point_id')

        local_mat = local_sub[date_cols].values.astype(np.float32)
        gee_mat   = gee_sub[date_cols].values.astype(np.float32)
        point_ids = local_sub['point_id'].values

        # แปลง x_proj/y_proj → lon/lat ทีเดียวทั้ง polygon
        lons, lats = transformer.transform(
            meta_sub['x_proj'].values,
            meta_sub['y_proj'].values
        )
        # map point_id → (lon, lat)
        pt_to_lon = dict(zip(meta_sub['point_id'].values, lons))
        pt_to_lat = dict(zip(meta_sub['point_id'].values, lats))

        diff = np.abs(local_mat - gee_mat)
        rows_idx, cols_idx = np.where(np.isfinite(diff) & (diff >= threshold))

        for r, c in zip(rows_idx, cols_idx):
            pt = int(point_ids[r])
            records.append({
                'plot_id':    pid,
                'point_id':   pt,
                'date':       date_cols[c],
                'lat':        round(pt_to_lat[pt], 6),
                'lon':        round(pt_to_lon[pt], 6),
                'local_ndvi': float(local_mat[r, c]),
                'gee_ndvi':   float(gee_mat[r, c]),
                'abs_diff':   float(diff[r, c]),
            })

    if not records:
        print(f"  ไม่มี diff >= {threshold}")
        return None

    diff_df = (pd.DataFrame(records)
               .sort_values('abs_diff', ascending=False)
               .reset_index(drop=True))

    csv_path = os.path.join(output_dir, "large_diffs.csv")
    diff_df.to_csv(csv_path, index=False)
    print(f"\n  บันทึก {len(diff_df):,} rows → {csv_path}")

    print(f"\n  Top {top_n} largest |local - gee| (threshold={threshold}):")
    print(f"  {'plot':>4}  {'point':>6}  {'date':>12}  {'lat':>10}  {'lon':>11}  {'local':>7}  {'gee':>7}  {'|diff|':>7}")
    print("  " + "-" * 72)
    for _, row in diff_df.head(top_n).iterrows():
        print(
            f"  {int(row.plot_id):>4}  "
            f"{int(row.point_id):>6}  "
            f"{row.date:>12}  "
            f"{row.lat:>10.6f}  "
            f"{row.lon:>11.6f}  "
            f"{row.local_ndvi:>7.4f}  "
            f"{row.gee_ndvi:>7.4f}  "
            f"{row.abs_diff:>7.4f}"
        )

    print(f"\n  Summary per plot (diff >= {threshold}):")
    summary = (diff_df.groupby('plot_id')
               .agg(n_pairs=('abs_diff', 'count'),
                    mean_diff=('abs_diff', 'mean'),
                    max_diff=('abs_diff', 'max'))
               .reset_index())
    print(summary.to_string(index=False))

    return diff_df

def report_unmatched_dates(local_df, gee_df, date_cols, plot_ids, output_dir):
    """
    หาวันที่ที่มีค่าแค่ source เดียว (อีก source เป็น NaN ทั้ง polygon)
    - local_only : local มีค่า แต่ gee NaN ทุก point ใน polygon นั้น
    - gee_only   : gee มีค่า แต่ local NaN ทุก point ใน polygon นั้น
    """
    records = []

    for pid in plot_ids:
        local_sub = local_df[local_df['plot_id'] == pid][date_cols].values.astype(np.float32)
        gee_sub   = gee_df[gee_df['plot_id']     == pid][date_cols].values.astype(np.float32)

        for ci, date in enumerate(date_cols):
            local_col = local_sub[:, ci]
            gee_col   = gee_sub[:, ci]

            local_has = np.any(np.isfinite(local_col))
            gee_has   = np.any(np.isfinite(gee_col))

            if local_has and not gee_has:
                status = 'local_only'
                local_valid = int(np.isfinite(local_col).sum())
                gee_valid   = 0
            elif gee_has and not local_has:
                status = 'gee_only'
                local_valid = 0
                gee_valid   = int(np.isfinite(gee_col).sum())
            else:
                continue   # matched (both have or both NaN)

            records.append({
                'plot_id':     pid,
                'date':        date,
                'status':      status,
                'local_valid_pts': local_valid,
                'gee_valid_pts':   gee_valid,
                'local_mean':  float(np.nanmean(local_col)) if local_valid > 0 else np.nan,
                'gee_mean':    float(np.nanmean(gee_col))   if gee_valid   > 0 else np.nan,
            })

    if not records:
        print("  ไม่มี unmatched dates")
        return None

    um_df = (pd.DataFrame(records)
             .sort_values(['plot_id', 'date'])
             .reset_index(drop=True))

    # บันทึก CSV
    csv_path = os.path.join(output_dir, "unmatched_dates.csv")
    um_df.to_csv(csv_path, index=False)
    print(f"\n  บันทึก {len(um_df):,} rows → {csv_path}")

    # print แยกตาม status
    for status in ['local_only', 'gee_only']:
        sub = um_df[um_df['status'] == status]
        label = 'LOCAL only (GEE = NaN)' if status == 'local_only' else 'GEE only (Local = NaN)'
        print(f"\n  [{label}]  {len(sub)} dates")
        if sub.empty:
            continue
        print(f"  {'plot':>4}  {'date':>12}  {'local_pts':>9}  {'gee_pts':>8}  {'local_mean':>10}  {'gee_mean':>9}")
        print("  " + "-" * 60)
        for _, row in sub.iterrows():
            lm = f"{row.local_mean:.4f}" if np.isfinite(row.local_mean) else "   NaN"
            gm = f"{row.gee_mean:.4f}"   if np.isfinite(row.gee_mean)   else "   NaN"
            print(
                f"  {int(row.plot_id):>4}  "
                f"{row.date:>12}  "
                f"{int(row.local_valid_pts):>9}  "
                f"{int(row.gee_valid_pts):>8}  "
                f"{lm:>10}  "
                f"{gm:>9}"
            )

    # summary
    print(f"\n  Summary unmatched per plot:")
    summary = (um_df.groupby(['plot_id', 'status'])
               .size()
               .unstack(fill_value=0)
               .reset_index())
    print(summary.to_string(index=False))

    return um_df
    
def filter_matched_dates_only(local_df, gee_df, date_cols, plot_ids, output_dir):
    """
    กรองเอาเฉพาะวันที่ที่ทั้ง local และ gee มีค่า (ทุก polygon)
    คือวันที่อย่างน้อย 1 point มีค่าในทั้งสอง source พร้อมกัน
    บันทึก ndvi_local_matched.parquet / ndvi_gee_matched.parquet
    """
    matched_dates = []

    for date in date_cols:
        all_matched = True
        for pid in plot_ids:
            local_col = local_df[local_df['plot_id'] == pid][date].values.astype(np.float32)
            gee_col   = gee_df[gee_df['plot_id']     == pid][date].values.astype(np.float32)

            local_has = np.any(np.isfinite(local_col))
            gee_has   = np.any(np.isfinite(gee_col))

            # ถ้า polygon ใดใด unmatched → ตัดวันนี้ออก
            if local_has != gee_has:
                all_matched = False
                break

        if all_matched:
            matched_dates.append(date)

    n_removed = len(date_cols) - len(matched_dates)
    print(f"\n  Total dates       : {len(date_cols)}")
    print(f"  Matched dates     : {len(matched_dates)}")
    print(f"  Removed (unmatched): {n_removed}")

    if not matched_dates:
        print("  ไม่มี matched dates เลย")
        return None, None, []

    # keep columns เฉพาะ matched dates
    keep_cols = ['plot_id', 'point_id'] + matched_dates
    local_matched = local_df[keep_cols].copy()
    gee_matched   = gee_df[keep_cols].copy()

    local_path = os.path.join(output_dir, "ndvi_local_matched.parquet")
    gee_path   = os.path.join(output_dir, "ndvi_gee_matched.parquet")
    local_matched.to_parquet(local_path, index=False)
    gee_matched.to_parquet(gee_path,     index=False)
    print(f"\n  Saved: {local_path}")
    print(f"  Saved: {gee_path}")

    # print วันที่ถูกตัดออก
    removed_dates = [d for d in date_cols if d not in matched_dates]
    if removed_dates:
        print(f"\n  Removed dates ({len(removed_dates)}):")
        for d in removed_dates:
            # บอกด้วยว่า source ไหนที่หายไปในแต่ละ plot
            detail = []
            for pid in plot_ids:
                local_col = local_df[local_df['plot_id'] == pid][d].values.astype(np.float32)
                gee_col   = gee_df[gee_df['plot_id']     == pid][d].values.astype(np.float32)
                local_has = np.any(np.isfinite(local_col))
                gee_has   = np.any(np.isfinite(gee_col))
                if local_has and not gee_has:
                    detail.append(f"plot{pid}=local_only")
                elif gee_has and not local_has:
                    detail.append(f"plot{pid}=gee_only")
            print(f"    {d}  →  {', '.join(detail)}")

    return local_matched, gee_matched, matched_dates
# ======================================================================================
# MAIN
# ======================================================================================

def main():
    print("Loading parquets...")
    local_df, gee_df, meta_df = load_parquets(OUTPUT_DIR)

    date_cols = get_date_columns(local_df)
    dates     = parse_dates(date_cols)
    plot_ids  = sorted(local_df['plot_id'].unique().tolist())

    print(f"  Polygons : {len(plot_ids)}")
    print(f"  Dates    : {len(date_cols)}")
    print(f"  Points   : {len(meta_df)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Plot 1: per-polygon point lines ----
    print("\n[1] Per-polygon point-level plots...")
    for pid in plot_ids:
        local_mat = wide_to_matrix(local_df, pid, date_cols)
        gee_mat   = wide_to_matrix(gee_df,   pid, date_cols)
        plot_polygon_points(
            plot_id=pid,
            local_mat=local_mat,
            gee_mat=gee_mat,
            dates=dates,
            output_dir=OUTPUT_DIR,
            province=PROVINCE,
        )

    # ---- Plot 2: summary mean ----
    print("\n[2] Summary mean plot (all polygons)...")
    plot_summary_all_polygons(
        local_df=local_df,
        gee_df=gee_df,
        date_cols=date_cols,
        dates=dates,
        plot_ids=plot_ids,
        output_dir=OUTPUT_DIR,
        province=PROVINCE,
        stat='mean',
    )

    # ---- Plot 3: summary median ----
    print("\n[3] Summary median plot (all polygons)...")
    plot_summary_all_polygons(
        local_df=local_df,
        gee_df=gee_df,
        date_cols=date_cols,
        dates=dates,
        plot_ids=plot_ids,
        output_dir=OUTPUT_DIR,
        province=PROVINCE,
        stat='median',
    )

    # ---- Plot 4: scatter local vs gee ----
    print("\n[4] Scatter local vs GEE...")
    plot_scatter_local_vs_gee(
        local_df=local_df,
        gee_df=gee_df,
        plot_ids=plot_ids,
        date_cols=date_cols,
        output_dir=OUTPUT_DIR,
    )

    print(f"\nAll plots saved to: {OUTPUT_DIR}")

    # ---- Diff report ----
    EPSG = 32647   # ← ใส่ค่าตรงๆ หรืออ่านจาก best_ts เหมือน collect_ndvi.py

    print("\n[5] Large diff report...")
    report_large_diffs(
        local_df=local_df,
        gee_df=gee_df,
        meta_df=meta_df,       # เพิ่ม
        date_cols=date_cols,
        plot_ids=plot_ids,
        output_dir=OUTPUT_DIR,
        epsg=EPSG,             # เพิ่ม
        threshold=0.15,
        top_n=20,
    )

    # ---- Unmatched dates ----
    print("\n[6] Unmatched dates report...")
    report_unmatched_dates(
        local_df=local_df,
        gee_df=gee_df,
        date_cols=date_cols,
        plot_ids=plot_ids,
        output_dir=OUTPUT_DIR,
    )

if __name__ == "__main__":
    main()