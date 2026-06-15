#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_angkanap_v2.py
======================
เทียบ ndvi_max จาก angkanap pipeline กับ ndvi_max ของ v2 pipeline (wanwisa)

angkanap source:
  /fs2/angkanap/15_Durian/08_model/00_index_points/ndvi/create_20260521/rayong/
  - 28,654 ไฟล์: rayong_segment_ndvi_{tile}_{plot_id}.parquet
  - cols: x, y, YYYY/MM/DD..., geometry  (individual S2 dates, per-point rows)
  - ไม่มี plot_id column → อ่านจากชื่อไฟล์

v2 source:
  /fs2/wanwisa/durian/outputs/rayong/ndvi_max.parquet
  - cols: plot_id, point_id, lat, lon, YYYY-MM..., geometry  (per-point monthly max)

ผลลัพธ์:
  scatter_angkanap_vs_v2.png      — polygon-level scatter per month-group
  timeseries_sample.png          — time-series กราฟสำหรับ plot_ids ที่เลือก
  diff_summary_angkanap_v2.csv   — per-polygon per-month diff
  stats_summary.txt              — overall stats

วิธีรัน:
  python3 compare_angkanap_v2.py
"""

# =============================================================================
# CONFIG
# =============================================================================

ANGKANAP_DIR     = "/fs2/angkanap/15_Durian/08_model/00_index_points/ndvi/create_20260521/rayong"
V2_PARQUET       = "/fs2/wanwisa/durian/outputs/rayong/ndvi_max.parquet"
OUTPUT_DIR       = "/fs2/wanwisa/durian/test_polygon_gee_vs_server/compare/angkanap_vs_v2"
N_JOBS           = 8        # parallel workers for reading angkanap files
SAMPLE_PLOT_IDS  = None     # None = auto-pick 6 plots; or list e.g. [100, 500, 1000]

# =============================================================================
# IMPORTS
# =============================================================================

import os, re, glob, time, warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# =============================================================================
# STEP 1: Read & aggregate angkanap files → polygon-level monthly max
# =============================================================================

def _process_file(path: str) -> pd.DataFrame | None:
    """
    อ่านหนึ่งไฟล์ → คำนวณ per-point monthly max → nanmean ข้ามpoints → 1 row per plot.
    Returns DataFrame: plot_id + monthly cols (YYYY-MM)
    """
    m = re.search(r'_(\d+)\.parquet$', path)
    if not m:
        return None
    plot_id = int(m.group(1))

    df = pd.read_parquet(path)
    date_cols = [c for c in df.columns if re.fullmatch(r'\d{4}/\d{2}/\d{2}', c)]
    if not date_cols:
        return None

    # group dates → YYYY-MM
    parsed    = pd.to_datetime(date_cols, format="%Y/%m/%d")
    month_map = {}
    for col, ts in zip(date_cols, parsed):
        mm = ts.strftime("%Y-%m")
        month_map.setdefault(mm, []).append(col)

    mat = df[date_cols].values.astype(np.float32)  # (n_pts, n_dates)

    row = {"plot_id": plot_id}
    for mm, cols in month_map.items():
        idx = [date_cols.index(c) for c in cols]
        sub = mat[:, idx]  # (n_pts, n_dates_in_month)
        # per-point max across dates in month
        pt_max = np.nanmax(sub, axis=1)  # (n_pts,)
        valid  = pt_max[np.isfinite(pt_max)]
        row[mm] = float(np.nanmean(valid)) if len(valid) > 0 else np.nan

    return pd.DataFrame([row])


def load_angkanap(src_dir: str, n_jobs: int = 8) -> pd.DataFrame:
    """อ่านทุกไฟล์แบบ parallel → aggregate ข้าม tiles (plot_id เดียวกัน → max)"""
    files  = sorted(glob.glob(os.path.join(src_dir, "*.parquet")))
    n      = len(files)
    print(f"  {n} files, {n_jobs} workers...")

    results = []
    done    = 0
    t0      = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_process_file, f): f for f in files}
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                results.append(r)
            done += 1
            if done % 2000 == 0:
                el = time.perf_counter() - t0
                print(f"    {done}/{n}  {el:.0f}s elapsed")

    elapsed = time.perf_counter() - t0
    print(f"  Read complete: {elapsed:.0f}s")

    combined = pd.concat(results, ignore_index=True)
    month_cols = sorted(c for c in combined.columns if c != "plot_id")

    # same plot_id can appear in multiple tiles → take max (best signal)
    n_before = len(combined)
    combined = (
        combined.groupby("plot_id", sort=False)[month_cols]
        .max()         # element-wise max across tiles
        .reset_index()
    )
    n_after = len(combined)
    if n_before != n_after:
        print(f"  Dedup tile overlap: {n_before} → {n_after} rows ({n_before-n_after} removed)")

    print(f"  angkanap: {len(combined)} plots, {len(month_cols)} months ({month_cols[0]} → {month_cols[-1]})")
    return combined


# =============================================================================
# STEP 2: Load v2 ndvi_max → polygon-level monthly max (nanmean of pt maxes)
# =============================================================================

def load_v2_max(path: str) -> pd.DataFrame:
    meta = {"plot_id", "point_id", "lat", "lon", "geometry"}
    df   = pd.read_parquet(path)
    mcols = sorted(c for c in df.columns if c not in meta)

    print(f"  v2: {df.plot_id.nunique()} plots, {len(mcols)} months ({mcols[0]} → {mcols[-1]})")

    # nanmean of per-point monthly max → polygon-level
    agg = (
        df.groupby("plot_id")[mcols]
        .mean()          # nanmean across points (NaN ignored by pandas default)
        .reset_index()
    )
    return agg


# =============================================================================
# STEP 3: Merge on matched months
# =============================================================================

def merge_matched(ang: pd.DataFrame, v2: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    ang_months = set(c for c in ang.columns if c != "plot_id")
    v2_months  = set(c for c in v2.columns  if c != "plot_id")
    shared     = sorted(ang_months & v2_months)
    print(f"\n  angkanap months: {len(ang_months)}  v2 months: {len(v2_months)}  shared: {len(shared)}")
    if shared:
        print(f"  shared range: {shared[0]} → {shared[-1]}")

    ang_sub = ang[["plot_id"] + shared].rename(columns={m: f"ang_{m}" for m in shared})
    v2_sub  = v2[["plot_id"] + shared].rename(columns={m: f"v2_{m}"  for m in shared})

    merged = ang_sub.merge(v2_sub, on="plot_id", how="inner")
    print(f"  merged: {len(merged)} plots")
    return merged, shared


# =============================================================================
# STEP 4: Build long-format diff table
# =============================================================================

def build_diff(merged: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    records = []
    for m in months:
        sub = merged[["plot_id", f"ang_{m}", f"v2_{m}"]].dropna()
        sub = sub.rename(columns={f"ang_{m}": "ang", f"v2_{m}": "v2"})
        sub["month"]    = m
        sub["diff"]     = (sub["v2"] - sub["ang"]).round(5)
        sub["abs_diff"] = sub["diff"].abs().round(5)
        records.append(sub)
    return pd.concat(records, ignore_index=True)


# =============================================================================
# STEP 5: Plots
# =============================================================================

def plot_scatter(diff_df: pd.DataFrame, months: list[str], output_dir: str):
    """Scatter: angkanap vs v2 monthly max per polygon, coloured by month-group."""
    n      = len(months)
    ncols  = min(n, 4)
    nrows  = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, m in enumerate(months):
        ax  = axes[i // ncols][i % ncols]
        sub = diff_df[diff_df["month"] == m].dropna()
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        ax.scatter(sub["ang"], sub["v2"], s=2, alpha=0.15, color="#2C3E50")
        mn = min(sub["ang"].min(), sub["v2"].min())
        mx = max(sub["ang"].max(), sub["v2"].max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="1:1")
        mae = sub["abs_diff"].mean()
        ax.set_title(f"{m}  (n={len(sub):,}  MAE={mae:.3f})", fontsize=9)
        ax.set_xlabel("angkanap max")
        ax.set_ylabel("v2 max")

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Scatter: angkanap vs v2 polygon-level monthly NDVI max\n(Rayong)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "scatter_angkanap_vs_v2.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_timeseries(ang: pd.DataFrame, v2: pd.DataFrame, months: list[str],
                    plot_ids: list[int], output_dir: str):
    """Time-series กราฟสำหรับ sample plot_ids."""
    dates  = pd.to_datetime(months, format="%Y-%m")
    n      = len(plot_ids)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, pid in zip(axes, plot_ids):
        ang_row = ang[ang.plot_id == pid]
        v2_row  = v2[v2.plot_id  == pid]

        a_vals = np.array([
            float(ang_row[m].values[0]) if m in ang_row.columns and not ang_row.empty else np.nan
            for m in months
        ], dtype=np.float32)
        v_vals = np.array([
            float(v2_row[m].values[0]) if m in v2_row.columns and not v2_row.empty else np.nan
            for m in months
        ], dtype=np.float32)

        va = np.isfinite(a_vals)
        vv = np.isfinite(v_vals)

        if va.sum() > 1:
            ax.plot(dates[va], a_vals[va], color="#27AE60", lw=1.8,
                    marker="o", markersize=4, label="angkanap max")
        if vv.sum() > 1:
            ax.plot(dates[vv], v_vals[vv], color="#E74C3C", lw=1.8,
                    marker="s", markersize=4, linestyle="--", label="v2 max")

        both = va & vv
        if both.sum() > 0:
            ax.fill_between(dates[both], a_vals[both], v_vals[both],
                            alpha=0.12, color="#8E44AD")

        ax.set_title(f"Plot {pid}", fontsize=11)
        ax.set_ylabel("NDVI max")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.grid(axis="x", linestyle=":", alpha=0.25)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
    axes[-1].set_xlabel("Month")

    fig.suptitle("NDVI max: angkanap vs v2  (Rayong — sample polygons)", fontsize=13, y=1.005)
    fig.tight_layout()
    out = os.path.join(output_dir, "timeseries_sample.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# =============================================================================
# STEP 6: Summary stats
# =============================================================================

def summarize(diff_df: pd.DataFrame, output_dir: str):
    overall_mae  = diff_df["abs_diff"].mean()
    overall_bias = diff_df["diff"].mean()
    overall_max  = diff_df["abs_diff"].max()
    n_pairs      = len(diff_df)

    by_month = (
        diff_df.groupby("month")
        .agg(n=("abs_diff","count"), mae=("abs_diff","mean"),
             bias=("diff","mean"), max_abs=("abs_diff","max"))
        .reset_index()
    )

    lines = [
        "=== angkanap vs v2 NDVI max comparison (Rayong) ===",
        f"  n polygon-month pairs : {n_pairs:,}",
        f"  Overall MAE           : {overall_mae:.4f}",
        f"  Overall bias (v2-ang) : {overall_bias:+.4f}",
        f"  Max abs diff          : {overall_max:.4f}",
        "",
        "  Per-month:",
    ]
    lines.append(f"  {'month':8}  {'n':>7}  {'MAE':>7}  {'bias':>7}  {'max':>7}")
    lines.append("  " + "-" * 44)
    for _, r in by_month.iterrows():
        lines.append(f"  {r.month:8}  {int(r.n):>7,}  {r.mae:>7.4f}  {r.bias:>+7.4f}  {r.max_abs:>7.4f}")

    txt = "\n".join(lines)
    print("\n" + txt)

    txt_path = os.path.join(output_dir, "stats_summary.txt")
    with open(txt_path, "w") as f:
        f.write(txt + "\n")
    print(f"\n  Saved: {txt_path}")

    csv_path = os.path.join(output_dir, "diff_summary_angkanap_v2.csv")
    diff_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1] Loading angkanap files (parallel)...")
    ang = load_angkanap(ANGKANAP_DIR, n_jobs=N_JOBS)

    print("\n[2] Loading v2 ndvi_max...")
    v2_raw = load_v2_max(V2_PARQUET)

    print("\n[3] Merging on matched months...")
    merged, months = merge_matched(ang, v2_raw)

    if not months:
        print("ERROR: ไม่มีเดือนที่ตรงกัน — ตรวจสอบ date formats")
        return

    # reformat v2 for time-series plot (polygon-level)
    v2_poly = v2_raw[["plot_id"] + months]

    print("\n[4] Building diff table...")
    diff_df = build_diff(merged, months)
    print(f"  {len(diff_df):,} polygon-month pairs")

    print("\n[5] Scatter plots...")
    plot_scatter(diff_df, months, OUTPUT_DIR)

    print("\n[6] Time-series sample plots...")
    global SAMPLE_PLOT_IDS
    if SAMPLE_PLOT_IDS is None:
        common_pids = sorted(set(ang.plot_id) & set(v2_raw.plot_id))
        # pick plots spread across ID range
        step = max(1, len(common_pids) // 6)
        SAMPLE_PLOT_IDS = [common_pids[i] for i in range(0, min(6 * step, len(common_pids)), step)]
    print(f"  Sample plot_ids: {SAMPLE_PLOT_IDS}")
    plot_timeseries(ang[["plot_id"] + months], v2_poly, months, SAMPLE_PLOT_IDS, OUTPUT_DIR)

    print("\n[7] Summary stats...")
    summarize(diff_df, OUTPUT_DIR)

    print(f"\nDone → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
