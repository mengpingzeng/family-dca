#!/usr/bin/env python3
"""
构建蛋卷版回测数据（5 年滚动窗口）

蛋卷 PE 为周频(pe_ttm_dj)，在周频上直接计算 5 年滚动百分位，
再 merge_asof 对齐到日频价格。

输出: aks_merged_dj/{code}.parquet (date, price, pe, pe_pct_w5)

用法: python data-store/build_danjuan_merged.py
"""
import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DJ_PE_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "index_pe_dj")
PRICE_DIRS = [
    os.path.join(PROJECT_DIR, "data-store", "parquet", "index_price_aks"),
    os.path.join(PROJECT_DIR, "data-store", "parquet", "index_price"),
]
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "aks_merged_dj")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_YEARS = [5]
WEEKS_PER_YEAR = 52  # 蛋卷周频

DJ_INDICES = [
    {"code": "000015", "name": "上证红利"},
    {"code": "000016", "name": "上证50"},
    {"code": "000300", "name": "沪深300"},
    {"code": "000688", "name": "科创50"},
    {"code": "000852", "name": "中证1000"},
    {"code": "000905", "name": "中证500"},
    {"code": "000922", "name": "中证红利"},
    {"code": "399006", "name": "创业板指"},
    {"code": "399330", "name": "深证100"},
    {"code": "HSI", "name": "恒生指数"},
    {"code": "HSTECH", "name": "恒生科技"},
    {"code": "NDX100", "name": "纳斯达克100"},
    {"code": "SPX500", "name": "标普500"},
]


def rolling_pct(series: np.ndarray, window_rows: int, min_samples: int = 20) -> np.ndarray:
    """滚动窗口百分位（在周频上计算）。第 (window_years+1) 年首周才能算。"""
    n = len(series)
    result = np.full(n, np.nan)
    clean = ~np.isnan(series)
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


def find_price_path(code: str):
    for d in PRICE_DIRS:
        p = os.path.join(d, f"{code}.parquet")
        if os.path.exists(p):
            return p
    return None


def build_one(code: str, name: str) -> pd.DataFrame:
    pe_path = os.path.join(DJ_PE_DIR, f"{code}.parquet")
    if not os.path.exists(pe_path):
        print(f"  [SKIP] 蛋卷 PE 不存在: {code}")
        return pd.DataFrame()

    price_path = find_price_path(code)
    if not price_path:
        print(f"  [SKIP] 价格不存在: {code}")
        return pd.DataFrame()

    # 蛋卷 PE (周频)
    pe = pd.read_parquet(pe_path)
    pe["date"] = pd.to_datetime(pe["date"])
    pe = pe.sort_values("date")[["date", "pe_ttm_dj"]].rename(columns={"pe_ttm_dj": "pe"})
    pe = pe.dropna(subset=["pe"]).reset_index(drop=True)

    # 在周频上直接计算 5 年滚动百分位
    for w in WINDOW_YEARS:
        wr = int(w * WEEKS_PER_YEAR)
        min_samples = wr  # 满 5 年才能算
        pe_arr = pe["pe"].values.astype(float)
        pe[f"pe_pct_w{w}"] = rolling_pct(pe_arr, wr, min_samples)

    # 价格 (日频)
    price = pd.read_parquet(price_path)
    price["date"] = pd.to_datetime(price["date"])
    price_col = "index_price" if "index_price" in price.columns else "close"
    price = price.sort_values("date")[["date", price_col]].rename(columns={price_col: "price"})

    # merge_asof 对齐: 日频价格为主, 周频 pe + pe_pct backward fill
    pe_sorted = pe.sort_values("date")
    merged = pd.merge_asof(price, pe_sorted, on="date", direction="backward")
    merged["pe"] = merged["pe"].ffill()
    for w in WINDOW_YEARS:
        merged[f"pe_pct_w{w}"] = merged[f"pe_pct_w{w}"].ffill()
    merged = merged.dropna(subset=["pe"]).reset_index(drop=True)

    if len(merged) < 50:
        print(f"  [SKIP] 有效数据不足: {len(merged)} 行")
        return pd.DataFrame()

    keep_cols = ["date", "price", "pe"]
    for w in WINDOW_YEARS:
        for col in [f"pe_pct_w{w}"]:
            if col in merged.columns:
                keep_cols.append(col)
    merged = merged[keep_cols]

    out_path = os.path.join(OUTPUT_DIR, f"{code}.parquet")
    merged.to_parquet(out_path, index=False)

    pct_col = "pe_pct_w5"
    valid = merged[merged[pct_col].notna()]
    first_valid = str(valid["date"].iloc[0].date()) if len(valid) > 0 else "N/A(不足5年)"
    print(f"  OK | {len(merged)}条 | {merged.date.min().date()} ~ {merged.date.max().date()} | "
          f"可交易 {len(valid)} 天, 起点 {first_valid}")

    return merged


def main():
    print(f"\n蛋卷版回测数据构建")
    print(f"窗口: {WINDOW_YEARS} 年滚动(周频) | 第 6 年首周可交易\n")

    for idx in DJ_INDICES:
        print(f"[BUILD] {idx['code']} {idx['name']} ...", flush=True)
        build_one(idx["code"], idx["name"])

    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
