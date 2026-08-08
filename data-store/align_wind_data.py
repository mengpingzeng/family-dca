#!/usr/bin/env python3
"""
Wind 数据对齐器

将 Wind PE 数据和现有 merged PE 合并为统一时序：
    - 2026-04-16 之前 → 使用 Wind 数据（历史更长）
    - 2026-04-16 之后 → 使用现有数据（更新更及时）
    - 重叠段保留对比字段

输出：
    data-store/parquet/aligned_source/{code}.parquet
        date, pe_aligned, pe_source, pe_ttm_wind, pe_ttm_ours,
        pe_pct_aligned, (pb_dj, bond_yield, fed_* 等现有字段直接继承)

用法：
    python data-store/align_wind_data.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
WIND_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "wind_source")
MERGED_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "merged")
OUT_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "aligned_source")

ALIGN_DATE = pd.Timestamp("2026-04-16")


def calc_percentile(series: pd.Series) -> pd.Series:
    ranked = series.rank(pct=True)
    return (ranked * 100).round(2)


def align_one(code: str) -> pd.DataFrame:
    wind_path = os.path.join(WIND_DIR, f"{code}.parquet")
    merged_path = os.path.join(MERGED_DIR, f"{code}.parquet")

    if not os.path.exists(wind_path):
        return pd.DataFrame()
    if not os.path.exists(merged_path):
        return pd.DataFrame()

    wind = pd.read_parquet(wind_path)
    merged = pd.read_parquet(merged_path)

    wind["date"] = pd.to_datetime(wind["date"])
    merged["date"] = pd.to_datetime(merged["date"])

    # Split: Wind 取其 early + overlap, merged 取其 recent
    wind_part = wind[wind["date"] <= ALIGN_DATE].copy()
    merged_part = merged[merged["date"] > ALIGN_DATE].copy()

    wind_part["pe_source"] = "wind"
    merged_part["pe_source"] = "ours"

    # PE value columns for alignment
    pe_col = "pe_ttm_csi" if "pe_ttm_csi" in merged.columns else "pe_ttm_dj"
    wind_part["pe_aligned"] = wind_part["pe_ttm_wind"]
    wind_part["pe_ttm_ours"] = np.nan
    merged_part["pe_aligned"] = merged_part[pe_col]
    merged_part["pe_ttm_wind"] = np.nan
    merged_part["pe_ttm_ours"] = merged_part[pe_col]

    # Other columns from merged (PB, bond_yield, FED) — only in the recent part
    for col in merged.columns:
        if col not in ["date", "pe_ttm_csi", "pe_ttm_dj"]:
            if col not in wind_part.columns:
                wind_part[col] = np.nan

    # Concatenate
    result = pd.concat([wind_part, merged_part], ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)

    # Compute aligned PE percentile
    result["pe_pct_aligned"] = calc_percentile(result["pe_aligned"])

    # Clean up: keep only relevant columns
    keep_cols = ["date", "pe_aligned", "pe_source", "pe_ttm_wind", "pe_ttm_ours", "pe_pct_aligned"]
    for col in ["pb_dj", "bond_yield", "fed_csi", "fed_dj"]:
        if col in result.columns:
            keep_cols.append(col)
    return result[keep_cols]


def main():
    print(f"\nWind 数据对齐 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs(OUT_DIR, exist_ok=True)

    codes = sorted([f.replace(".parquet", "") for f in os.listdir(WIND_DIR) if f.endswith(".parquet")])

    success = 0
    for code in codes:
        df = align_one(code)
        if df.empty:
            print(f"  [SKIP] {code}")
            continue
        out_path = os.path.join(OUT_DIR, f"{code}.parquet")
        df.to_parquet(out_path, index=False)
        wind_rows = len(df[df["pe_source"] == "wind"])
        ours_rows = len(df[df["pe_source"] == "ours"])
        print(f"  {code:8s} {len(df):>5d}条 (Wind:{wind_rows}+现有:{ours_rows}) "
              f"{df.date.min().date()} ~ {df.date.max().date()} "
              f"PE {df.pe_aligned.min():.2f}~{df.pe_aligned.max():.2f}")
        success += 1

    print(f"\n{'='*60}")
    print(f"结果: 对齐 {success} 个指数 → {OUT_DIR}/")


if __name__ == "__main__":
    main()
