#!/usr/bin/env python3
"""
推送数据构建器 — 从蛋卷 PE/PB + 国债 + 指数价格 构建推送信号所需数据。

输出: wind_new_search/push/data/{code}.parquet
列: date, price, pe, pb, fed, pe_pct, pb_pct, fed_pct, window
（与 wind_new_merged 同构, 可直接喂给 engine）

数据流:
  index_pe_dj/{code}.parquet   (蛋卷 PE, 周频)
  index_pb/{code}.parquet      (蛋卷 PB, 周频)
  bond_yield/{cn|us}_10y_bond.parquet (国债, 日频)
  index_price/{code}.parquet 或 index_price_aks/{code}.parquet (价格, 日频)
    -> merge_asof 对齐 -> 计算 pe_pct/pb_pct/fed_pct -> 输出

用法: python wind_new_search/push/build_push_data.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

PUSH_DIR = PROJECT_DIR / "wind_new_search" / "push"
DATA_DIR = PUSH_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

PE_DIR = PROJECT_DIR / "data-store" / "parquet" / "index_pe_dj"
PB_DIR = PROJECT_DIR / "data-store" / "parquet" / "index_pb"
BOND_DIR = PROJECT_DIR / "data-store" / "parquet" / "bond_yield"
PRICE_DIRS = [
    PROJECT_DIR / "data-store" / "parquet" / "index_price",
    PROJECT_DIR / "data-store" / "parquet" / "index_price_aks",
]

MAX_WINDOW = 10


def load_config():
    with open(PUSH_DIR / "config.json") as f:
        return json.load(f)


def rolling_pct(series, window_rows, min_samples=None):
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


def load_bond(src):
    if src == "cn":
        df = pd.read_parquet(BOND_DIR / "cn_10y_bond.parquet")
        df = df.rename(columns={"bond_yield_cn": "bond_yield"})
    else:
        df = pd.read_parquet(BOND_DIR / "us_10y_bond.parquet")
        df = df.rename(columns={c: "bond_yield" for c in df.columns if "yield" in c})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "bond_yield"]].sort_values("date").reset_index(drop=True)


def find_price(code):
    for d in PRICE_DIRS:
        p = d / f"{code}.parquet"
        if p.exists():
            return p
    return None


def build_one(code, info):
    pe_path = PE_DIR / f"{code}.parquet"
    pb_path = PB_DIR / f"{code}.parquet"
    price_path = find_price(code)
    if not pe_path.exists() or not price_path.exists():
        print(f"  [SKIP] {code}: PE或价格缺失")
        return None

    pe = pd.read_parquet(pe_path)
    pe["date"] = pd.to_datetime(pe["date"])
    pe = pe.sort_values("date")[["date", "pe_ttm_dj"]].rename(columns={"pe_ttm_dj": "pe"})
    pe = pe.dropna(subset=["pe"]).reset_index(drop=True)

    pb = pd.DataFrame()
    if pb_path.exists():
        pb = pd.read_parquet(pb_path)
        pb["date"] = pd.to_datetime(pb["date"])
        pb = pb.sort_values("date")[["date", "pb_dj"]].rename(columns={"pb_dj": "pb"})
        pb = pb.dropna(subset=["pb"]).reset_index(drop=True)

    # 周频对齐 PE/PB 合并
    pe = pe.set_index("date")
    if len(pb):
        pb = pb.set_index("date")
        merged_wk = pe.join(pb, how="outer").sort_index().reset_index()
        merged_wk["pe"] = merged_wk["pe"].ffill().bfill()
        merged_wk["pb"] = merged_wk["pb"].ffill().bfill()
    else:
        merged_wk = pe.reset_index()
        merged_wk["pb"] = np.nan

    # FED = 1/PE*100 - bond
    bond = load_bond(info.get("bond", "cn"))
    merged_wk = pd.merge_asof(merged_wk, bond, on="date", direction="backward")
    merged_wk["bond_yield"] = merged_wk["bond_yield"].ffill()
    merged_wk["fed"] = np.where(
        (merged_wk["pe"] > 0) & (merged_wk["bond_yield"].notna()),
        (1.0 / merged_wk["pe"]) * 100 - merged_wk["bond_yield"],
        np.nan,
    )

    # 滚动百分位 (周频, 窗口 = min(10, floor(有效PE年限)))
    pe_valid = merged_wk[merged_wk["pe"].notna()]
    total_years = (pe_valid["date"].max() - pe_valid["date"].min()).days / 365.25 if len(pe_valid) else 0
    window = min(MAX_WINDOW, max(1, int(total_years)))
    rpy = len(pe_valid) / max(total_years, 1)
    wr = int(window * rpy)
    # 蛋卷历史多为 10 年整, 严格满窗(min_samples=wr)会让前段全部无信号;
    # 放宽到窗口的 80%: 最新一天仍满窗精确, 历史回溯平滑可用。
    min_samples = max(20, int(wr * 0.8))

    merged_wk["pe_pct"] = rolling_pct(merged_wk["pe"].values.astype(float), wr, min_samples)
    merged_wk["pb_pct"] = rolling_pct(merged_wk["pb"].values.astype(float), wr, min_samples)
    merged_wk["fed_pct"] = rolling_pct(merged_wk["fed"].values.astype(float), wr, min_samples)
    merged_wk["window"] = window

    # 价格对齐到日频
    price = pd.read_parquet(price_path)
    price["date"] = pd.to_datetime(price["date"])
    price_col = "index_price" if "index_price" in price.columns else "close"
    price = price.sort_values("date")[["date", price_col]].rename(columns={price_col: "price"})

    out = pd.merge_asof(price, merged_wk, on="date", direction="backward")
    for c in ["pe", "pb", "fed", "pe_pct", "pb_pct", "fed_pct"]:
        out[c] = out[c].ffill()
    out = out.dropna(subset=["pe", "pe_pct"]).reset_index(drop=True)

    cols = ["date", "price", "pe", "pb", "fed", "pe_pct", "pb_pct", "fed_pct", "window"]
    out = out[cols]
    out.to_parquet(DATA_DIR / f"{code}.parquet", index=False)

    tradable = out[out["pe_pct"].notna()]
    print(f"  OK | {code:8s} {info['name']:6s} 窗口={window}yr | {len(out)}行 "
          f"| {out['date'].min().date()} ~ {out['date'].max().date()} "
          f"| 可交易 {len(tradable)} 行")
    return out


def main():
    cfg = load_config()
    print(f"\n推送数据构建 ({len(cfg['indices'])} 指数)\n{'='*60}")
    for code, info in cfg["indices"].items():
        build_one(code, info)
    print(f"\n输出: {DATA_DIR}")


if __name__ == "__main__":
    main()
