#!/usr/bin/env python3
"""
构建 akshare 回测用合并数据

将 akshare PE/PB + 国债收益率 + 指数价格合并为统一的回测 DataFrame。
输出到 parquet/aks_merged/，每指数一个文件。

列结构:
  date, price, pe, pb, fed,
  pe_pct_w3/w5/w10, pb_pct_w3/w5/w10, fed_pct_w3/w5/w10

用法:
  python data-store/build_aks_merged.py            # 全部构建
  python data-store/build_aks_merged.py --check     # 验证
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
AKS_PE_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "aks", "pe")
AKS_PB_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "aks", "pb")
BOND_PATH = os.path.join(PROJECT_DIR, "data-store", "parquet", "bond_yield", "cn_10y_bond.parquet")
PRICE_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "index_price_aks")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "aks_merged")

INDICES = [
    {"code": "000300", "name": "沪深300", "pe_col": "pe_wgt"},
    {"code": "000016", "name": "上证50",  "pe_col": "pe_wgt"},
    {"code": "000905", "name": "中证500", "pe_col": "pe_med"},
]

WINDOW_YEARS = [3, 5, 10]

# ============================================================================
# 滚动百分位（向量化）
# ============================================================================

def rolling_pct(series: np.ndarray, window_rows: int, min_samples: int = 20) -> np.ndarray:
    n = len(series)
    result = np.full(n, np.nan)
    clean = ~np.isnan(series)
    for i in range(n):
        if not clean[i]:
            continue
        start = max(0, i - window_rows)
        wc = clean[start:i+1]
        if wc.sum() < min_samples:
            continue
        w = series[start:i+1][wc]
        result[i] = (w <= series[i]).sum() / len(w)
    return result

# ============================================================================
# 构建单指数合并数据
# ============================================================================

def build_one(code: str, name: str, pe_col: str) -> pd.DataFrame:
    """合并单个指数的 PE + PB + FED + 价格 + 滚动百分位。"""
    # 读 PE
    pe_path = os.path.join(AKS_PE_DIR, f"{code}.parquet")
    if not os.path.exists(pe_path):
        print(f"  [SKIP] PE 文件不存在: {pe_path}")
        return pd.DataFrame()
    pe = pd.read_parquet(pe_path)
    pe["date"] = pd.to_datetime(pe["date"])
    pe = pe.sort_values("date")[["date", pe_col]].rename(columns={pe_col: "pe"})

    # 读 PB
    pb_path = os.path.join(AKS_PB_DIR, f"{code}.parquet")
    pb = pd.read_parquet(pb_path)
    pb["date"] = pd.to_datetime(pb["date"])
    pb = pb.sort_values("date")[["date", "pb_wgt"]].rename(columns={"pb_wgt": "pb"})

    # 读债券
    bond = pd.read_parquet(BOND_PATH)
    bond["date"] = pd.to_datetime(bond["date"])
    bond = bond.sort_values("date")[["date", "bond_yield_cn"]].rename(
        columns={"bond_yield_cn": "bond_yield"}
    )

    # 读价格
    price_path = os.path.join(PRICE_DIR, f"{code}.parquet")
    if not os.path.exists(price_path):
        print(f"  [SKIP] 价格文件不存在: {price_path}")
        return pd.DataFrame()
    price = pd.read_parquet(price_path)
    price["date"] = pd.to_datetime(price["date"])
    price_col = "index_open" if "index_open" in price.columns else "index_price"
    price = price.sort_values("date")[["date", price_col]].rename(
        columns={price_col: "price"}
    )

    # 用 merge_asof 对齐（PE/PB/bond 以价格日期为索引）
    for src, cols in [(pe, ["pe"]), (pb, ["pb"]), (bond, ["bond_yield"])]:
        src_sorted = src.sort_values("date")
        src_sorted = src_sorted[["date"] + cols].dropna(subset=cols)
        if src_sorted.empty:
            continue
        price = pd.merge_asof(price, src_sorted, on="date", direction="backward")
        for c in cols:
            if c in price.columns:
                price[c] = price[c].ffill()

    # 计算 FED
    price["fed"] = np.where(
        (price["pe"] > 0) & (price["bond_yield"].notna()),
        (1.0 / price["pe"]) * 100 - price["bond_yield"],
        np.nan,
    )

    # 只保留有 PE 的行
    price = price.dropna(subset=["pe"]).reset_index(drop=True)
    if len(price) < 100:
        print(f"  [SKIP] 有效数据不足: {len(price)} 行")
        return pd.DataFrame()

    # 计算日期跨度
    total_days = (price["date"].max() - price["date"].min()).days
    total_years = total_days / 365.25
    rpy = len(price) / max(total_years, 1)  # rows per year

    # 计算各窗口滚动百分位
    for w in WINDOW_YEARS:
        wr = int(w * rpy)
        min_samples = max(20, wr)  # 完整窗口: 必须积累满 wr 行才开始计算

        # PE 百分位
        pe_arr = price["pe"].values.astype(float)
        price[f"pe_pct_w{w}"] = rolling_pct(pe_arr, wr, min_samples)

        # PB 百分位
        if "pb" in price.columns and price["pb"].notna().sum() > 50:
            pb_arr = price["pb"].values.astype(float)
            price[f"pb_pct_w{w}"] = rolling_pct(pb_arr, wr, min_samples)

        # FED 百分位
        if "fed" in price.columns and price["fed"].notna().sum() > 50:
            fed_arr = price["fed"].values.astype(float)
            price[f"fed_pct_w{w}"] = rolling_pct(fed_arr, wr, min_samples)

    # 保留需要的列
    keep_cols = ["date", "price", "pe", "pb", "fed"]
    for w in WINDOW_YEARS:
        for col in [f"pe_pct_w{w}", f"pb_pct_w{w}", f"fed_pct_w{w}"]:
            if col in price.columns:
                keep_cols.append(col)
    price = price[keep_cols]

    # 保存
    out_path = os.path.join(OUTPUT_DIR, f"{code}.parquet")
    price.to_parquet(out_path, index=False)

    # 打印
    tradable = {}
    for w in WINDOW_YEARS:
        pct_col = f"pe_pct_w{w}"
        first_valid = price[price[pct_col].notna()].iloc[0] if pct_col in price.columns else None
        tradable[w] = str(first_valid["date"].date()) if first_valid is not None else "N/A"

    print(f"  OK | {len(price)}条 | {price.date.min().date()} ~ {price.date.max().date()} | "
          f"可交易: 3yr={tradable[3]} 5yr={tradable[5]} 10yr={tradable[10]}")

    return price


def main():
    print(f"\nakshare 回测数据构建 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx in INDICES:
        print(f"[BUILD] {idx['code']} {idx['name']} (PE={idx['pe_col']}) ...", end=" ", flush=True)
        df = build_one(idx["code"], idx["name"], idx["pe_col"])
        if df.empty:
            print("FAIL")
        else:
            pass

    print(f"\n输出目录: {OUTPUT_DIR}")


def check():
    print("akshare 回测数据检查\n")
    for idx in INDICES:
        path = os.path.join(OUTPUT_DIR, f"{idx['code']}.parquet")
        if not os.path.exists(path):
            print(f"  {idx['code']} {idx['name']}: 未构建")
            continue
        df = pd.read_parquet(path)
        cols = df.columns.tolist()
        print(f"  {idx['code']} {idx['name']}: {len(df)}条, "
              f"{df.date.min().date()}~{df.date.max().date()}, "
              f"cols={[c for c in cols if 'pct' in c or c in ['pe','pb','fed','price']]}")
        for w in WINDOW_YEARS:
            pct_col = f"pe_pct_w{w}"
            if pct_col in df.columns:
                valid = df[df[pct_col].notna()]
                if len(valid) > 0:
                    print(f"    窗口{w}yr: {len(valid)}个有效交易日, "
                          f"{valid.date.min().date()}起可交易")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="akshare 回测数据构建器")
    parser.add_argument("--check", action="store_true", help="验证数据")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        main()
