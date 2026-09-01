#!/usr/bin/env python3
"""
新版 Wind 数据合并构建器
=========================

从 wind_new/ 目录读取 14 个指数 xlsx (PE/PB/股息率/收盘价) + 美国国债 CSV，
合并债券收益率计算 FED，并按各指数自身历史年限计算滚动百分位。

数据流:
    wind_new/{code}.xlsx + bond_yield (cn/us)
      -> PE=市盈率TTM, PB=市净率LF, 收盘价, FED=1/PE*100 - bond
      -> 滚动百分位 pe_pct/pb_pct/fed_pct (窗口 = min(10, floor(年限)))
      -> parquet/wind_new_merged/{code}.parquet

窗口规则: 10 年为主；不足 10 年的指数用 floor(已积累年限) 年。
  - 港股通50 930931 -> 9年, 港股综合 930930 -> 8年, 科创50/恒生科技 -> 6年

用法: python data-store/build_wind_new_merged.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WIND_NEW_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new"
CN_BOND_PATH = PROJECT_DIR / "data-store" / "parquet" / "bond_yield" / "cn_10y_bond.parquet"
US_BOND_CSV = WIND_NEW_DIR / "美国_联邦基金目标利率.csv"
OUTPUT_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_WINDOW = 10

# code(输出) -> xlsx 前缀, 名称, 债券口径
INDICES = [
    {"code": "000015", "xlsx": "000015.SH",   "name": "上证红利",   "bond": "cn"},
    {"code": "000016", "xlsx": "000016.SH",   "name": "上证50",     "bond": "cn"},
    {"code": "000300", "xlsx": "000300.SH",   "name": "沪深300",    "bond": "cn"},
    {"code": "000688", "xlsx": "000688.SH",   "name": "科创50",     "bond": "cn", "window": 3},
    {"code": "000852", "xlsx": "000852.SH",   "name": "中证1000",   "bond": "cn"},
    {"code": "000905", "xlsx": "000905.SH",   "name": "中证500",    "bond": "cn"},
    {"code": "399006", "xlsx": "399006.SZ",   "name": "创业板指",   "bond": "cn"},
    {"code": "399330", "xlsx": "399330.SZ",   "name": "深证100",    "bond": "cn"},
    {"code": "930930", "xlsx": "930930.CSI",  "name": "港股综合",   "bond": "us", "window": 5},
    {"code": "930931", "xlsx": "930931.CSI",  "name": "港股通50",   "bond": "us", "window": 5},
    {"code": "HSI",    "xlsx": "HSI.HI",      "name": "恒生指数",   "bond": "us"},
    {"code": "HSTECH", "xlsx": "HSTECH.HI",   "name": "恒生科技",   "bond": "us", "window": 3},
    {"code": "NDX100", "xlsx": "NDX.GI",      "name": "纳斯达克100","bond": "us"},
    {"code": "SPX500", "xlsx": "SPX.GI",      "name": "标普500",    "bond": "us"},
]


def rolling_pct(series, window_rows, min_samples=None):
    """滚动百分位。窗口内需要至少 min_samples 个有效值才输出。"""
    n = len(series)
    result = np.full(n, np.nan)
    clean = ~np.isnan(series)
    if min_samples is None:
        min_samples = window_rows
    for i in range(n):
        if not clean[i]:
            continue
        start = max(0, i - window_rows)
        wc = clean[start:i + 1]
        if wc.sum() < min_samples:
            continue
        w = series[start:i + 1][wc]
        result[i] = (w <= series[i]).sum() / len(w)
    return result


def load_us_bond():
    """解析新版 CSV 中的美国 10 年国债收益率 (1953-2026)."""
    df = pd.read_csv(US_BOND_CSV, encoding="gbk")
    df = df.drop(index=[0, 1])
    df = df[df["指标名称"] != "数据来源：Wind"]
    df["date"] = pd.to_datetime(df["指标名称"], errors="coerce")
    df = df[df["date"].notna()]
    df["bond_yield"] = pd.to_numeric(df["美国:国债收益率:10年"], errors="coerce")
    df = df[["date", "bond_yield"]].dropna().sort_values("date").reset_index(drop=True)
    return df


def load_cn_bond():
    df = pd.read_parquet(CN_BOND_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"bond_yield_cn": "bond_yield"})[["date", "bond_yield"]]
    return df.sort_values("date").reset_index(drop=True)


def build_one(info, cn_bond, us_bond):
    code, name, bond_src = info["code"], info["name"], info["bond"]
    xlsx_path = WIND_NEW_DIR / f"{info['xlsx']}-历史PEPB-20260816.xlsx"
    if not xlsx_path.exists():
        print(f"  [SKIP] 文件不存在: {xlsx_path.name}")
        return None

    raw = pd.read_excel(xlsx_path)
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["交易日期"]),
        "price": pd.to_numeric(raw["收盘价"], errors="coerce"),
        "pe": pd.to_numeric(raw["市盈率TTM"], errors="coerce"),
        "pb": pd.to_numeric(raw["市净率LF"], errors="coerce"),
    })
    df = df.sort_values("date").reset_index(drop=True)

    # 对齐债券收益率 (backward fill)
    bond = cn_bond if bond_src == "cn" else us_bond
    df = pd.merge_asof(df, bond, on="date", direction="backward")
    df["bond_yield"] = df["bond_yield"].ffill()

    # FED = 盈利收益率 - 债券收益率
    df["fed"] = np.where(
        (df["pe"] > 0) & (df["bond_yield"].notna()),
        (1.0 / df["pe"]) * 100 - df["bond_yield"],
        np.nan,
    )

    # 确定窗口年限: 显式指定优先, 否则 min(10, floor(有效PE年限))
    pe_valid = df[df["pe"].notna()]
    total_years = (pe_valid["date"].max() - pe_valid["date"].min()).days / 365.25
    if info.get("window"):
        window = int(info["window"])
    else:
        window = min(MAX_WINDOW, max(1, int(total_years)))
    rpy = len(pe_valid) / max(total_years, 1)  # 每年行数 (周频 ~52)
    wr = int(window * rpy)
    min_samples = max(20, wr)

    pe_arr = df["pe"].values.astype(float)
    pb_arr = df["pb"].values.astype(float)
    fed_arr = df["fed"].values.astype(float)
    df["pe_pct"] = rolling_pct(pe_arr, wr, min_samples)
    df["pb_pct"] = rolling_pct(pb_arr, wr, min_samples)
    df["fed_pct"] = rolling_pct(fed_arr, wr, min_samples)
    df["window"] = window

    tradable = df[df["pe_pct"].notna()]
    first_trade = tradable["date"].iloc[0].date() if len(tradable) else "N/A"
    print(f"  OK | {code:8s} {name:6s} 窗口={window}yr | {len(df)}行 PE {df.pe.min():.2f}~{df.pe.max():.2f} "
          f"| 可交易 {len(tradable)} 行, 起点 {first_trade} | bond={bond_src}")

    out = OUTPUT_DIR / f"{code}.parquet"
    df[["date", "price", "pe", "pb", "fed", "pe_pct", "pb_pct", "fed_pct", "window"]].to_parquet(out, index=False)
    return df


def main():
    print(f"\n新版 Wind 合并数据构建\n{'='*60}")
    cn_bond = load_cn_bond()
    us_bond = load_us_bond()
    print(f"债券: 中债 {cn_bond.date.min().date()}~{cn_bond.date.max().date()} | "
          f"美债 {us_bond.date.min().date()}~{us_bond.date.max().date()}")

    for info in INDICES:
        build_one(info, cn_bond, us_bond)

    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
