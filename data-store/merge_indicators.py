#!/usr/bin/env python3
"""
指标合并器

将 PE、PB、国债收益率合并为统一 Parquet，预计算 FED 和 PE 百分位，
供前端直接消费。

输入:
    parquet/index_pe/{code}.parquet   (pe_ttm_dj 或 pe_ttm_csi)
    parquet/index_pb/{code}.parquet   (pb_dj)
    parquet/bond_yield/cn_10y_bond.parquet (bond_yield)

输出:
    parquet/merged/{code}.parquet
        date, pe_ttm_dj, pe_ttm_csi, pb_dj, bond_yield,
        fed_dj, fed_csi, pe_pct_dj, pe_pct_csi

用法:
    python merge_indicators.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PE_DJ_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "index_pe_dj")
PE_CSI_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "index_pe_csi")
PB_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "index_pb")
PRICE_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "index_price")
WIND_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "wind_source")
BOND_PATH = os.path.join(PROJECT_DIR, "data-store", "parquet", "bond_yield", "cn_10y_bond.parquet")
BOND_US_PATH = os.path.join(PROJECT_DIR, "data-store", "parquet", "bond_yield", "us_10y_bond.parquet")
MERGED_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "merged")

# 市场分类：决定用中国还是美国国债
US_MARKET_CODES = {
    "SPX500", "NDX100",      # 美股
    "HSI", "HSTECH",          # 港股（港币挂钩美元）
    "930930", "930931",       # 港股通（港股）
    "931573", "930915",       # 港股通科技/高股息（港股）
}

INDEX_NAMES = {
    "000300": "沪深300", "000905": "中证500", "000852": "中证1000",
    "000016": "上证50", "000688": "科创50", "000510": "中证A500",
    "399006": "创业板指", "399330": "深证100",
    "000015": "上证红利", "000922": "中证红利", "930955": "红利低波100",
    "930915": "港股通高股息", "930930": "港股综合", "930931": "港股通50",
    "931573": "港股通科技", "930939": "中证质量成长",
    "HSI": "恒生指数", "HSTECH": "恒生科技",
    "NDX100": "纳斯达克100", "SPX500": "标普500",
}


def calc_percentile(series: pd.Series) -> pd.Series:
    """对序列计算每个值在其历史分布中的百分位（0~100）。"""
    ranked = series.rank(pct=True)
    return (ranked * 100).round(2)


def merge_one(code: str, bond_df: pd.DataFrame, use_us: bool = False) -> pd.DataFrame:
    """合并单个指数的 PE + PB + 国债，计算 FED 和百分位。"""
    dj_path = os.path.join(PE_DJ_DIR, f"{code}.parquet")
    csi_path = os.path.join(PE_CSI_DIR, f"{code}.parquet")
    pb_path = os.path.join(PB_DIR, f"{code}.parquet")
    wind_path = os.path.join(WIND_DIR, f"{code}.parquet")

    # 从三个 PE 来源独立读入
    dfs = []
    source_count = 0

    # 蛋卷 PE
    if os.path.exists(dj_path):
        dj = pd.read_parquet(dj_path)
        if not dj.empty:
            dj["date"] = pd.to_datetime(dj["date"])
            dj = dj[["date", "pe_ttm_dj"]].dropna(subset=["pe_ttm_dj"])
            if not dj.empty:
                dfs.append(dj)
                source_count += 1

    # 中证 PE
    if os.path.exists(csi_path):
        csi = pd.read_parquet(csi_path)
        if not csi.empty:
            csi["date"] = pd.to_datetime(csi["date"])
            csi = csi[["date", "pe_ttm_csi"]].dropna(subset=["pe_ttm_csi"])
            if not csi.empty:
                dfs.append(csi)
                source_count += 1

    # Wind PE
    if os.path.exists(wind_path):
        wind = pd.read_parquet(wind_path)
        if not wind.empty:
            wind["date"] = pd.to_datetime(wind["date"])
            wind = wind.rename(columns={"pe_ttm_wind": "pe_ttm_wind"})
            wind = wind[["date", "pe_ttm_wind"]].dropna(subset=["pe_ttm_wind"])
            if not wind.empty:
                dfs.append(wind)
                source_count += 1

    if source_count == 0:
        return pd.DataFrame()

    # 按 date 合并所有来源
    df = dfs[0]
    for other in dfs[1:]:
        df = df.merge(other, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True)

    # 读指数价格
    price_path = os.path.join(PRICE_DIR, f"{code}.parquet")
    if os.path.exists(price_path):
        pr = pd.read_parquet(price_path)
        pr["date"] = pd.to_datetime(pr["date"])
        df = df.merge(pr[["date", "index_price"]], on="date", how="left")
        df["index_price"] = df["index_price"].ffill()

    # 读 PB，前向填充使周频数据对齐日频日期
    if os.path.exists(pb_path):
        pb = pd.read_parquet(pb_path)
        pb["date"] = pd.to_datetime(pb["date"])
        pb = pb.sort_values("date")
        df = df.merge(pb, on="date", how="left")
        df["pb_dj"] = df["pb_dj"].ffill()

    # 合国债收益率，前向填充
    if not bond_df.empty:
        df = df.merge(bond_df, on="date", how="left")
        df["bond_yield"] = df["bond_yield"].ffill()

    bond_label = "us" if use_us else "cn"
    if "bond_yield" in df.columns:
        df["bond_source"] = bond_label

    # 计算 FED（股票收益率 - 国债收益率）
    # FED > 0 表示股票相对债券便宜
    if "pe_ttm_csi" in df.columns:
        df["fed_csi"] = np.where(
            (df["pe_ttm_csi"] > 0) & df["bond_yield"].notna(),
            (1.0 / df["pe_ttm_csi"]) * 100 - df["bond_yield"],
            np.nan,
        )
        df["pe_pct_csi"] = calc_percentile(df["pe_ttm_csi"])
        df["fed_pct_csi"] = calc_percentile(df["fed_csi"])

    if "pe_ttm_dj" in df.columns:
        df["fed_dj"] = np.where(
            (df["pe_ttm_dj"] > 0) & df["bond_yield"].notna(),
            (1.0 / df["pe_ttm_dj"]) * 100 - df["bond_yield"],
            np.nan,
        )
        df["pe_pct_dj"] = calc_percentile(df["pe_ttm_dj"])
        df["fed_pct_dj"] = calc_percentile(df["fed_dj"])

    if "pe_ttm_wind" in df.columns:
        df["pe_pct_wind"] = calc_percentile(df["pe_ttm_wind"])

    # PB 百分位
    if "pb_dj" in df.columns:
        df["pb_pct_dj"] = calc_percentile(df["pb_dj"])

    return df


def main():
    print(f"\n指标合并 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs(MERGED_DIR, exist_ok=True)

    # 读国债收益率
    bond_cn = pd.DataFrame()
    bond_us = pd.DataFrame()
    if os.path.exists(BOND_PATH):
        bond_cn = pd.read_parquet(BOND_PATH)
        bond_cn["date"] = pd.to_datetime(bond_cn["date"])
        bond_cn = bond_cn.rename(columns={"bond_yield_cn": "bond_yield"})
        print(f"中国10Y国债: {len(bond_cn)}条, {bond_cn['date'].min().date()}~{bond_cn['date'].max().date()}")
    if os.path.exists(BOND_US_PATH):
        bond_us = pd.read_parquet(BOND_US_PATH)
        bond_us["date"] = pd.to_datetime(bond_us["date"])
        bond_us = bond_us.rename(columns={"bond_yield_us": "bond_yield"})
        print(f"美国10Y国债: {len(bond_us)}条, {bond_us['date'].min().date()}~{bond_us['date'].max().date()}")
    if bond_cn.empty and bond_us.empty:
        print("[WARN] 国债收益率文件不存在，FED 将为空")

    # 收集所有指数代码
    codes = set()
    for d in [PE_DJ_DIR, PE_CSI_DIR, PB_DIR]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".parquet"):
                    codes.add(f.replace(".parquet", ""))

    success = 0
    for code in sorted(codes):
        name = INDEX_NAMES.get(code, code)
        use_us = code in US_MARKET_CODES
        bond_df = bond_us if use_us else bond_cn
        df = merge_one(code, bond_df, use_us=use_us)
        if df.empty:
            continue

        out_path = os.path.join(MERGED_DIR, f"{code}.parquet")
        df.to_parquet(out_path, index=False)

        cols = df.columns.tolist()
        print(f"  {code:8s} {name:10s} {len(df):>5d}条 {df['date'].min().date()}~{df['date'].max().date()}  cols={cols}")
        success += 1

    print(f"\n{'='*60}")
    print(f"结果: 合并 {success} 个指数 → {MERGED_DIR}/")


if __name__ == "__main__":
    main()
