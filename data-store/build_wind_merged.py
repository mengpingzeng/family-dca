#!/usr/bin/env python3
"""
构建 wind 版回测数据

数据流: wind_source/{code}.parquet (pe_ttm_wind) + index_price_aks/index_price (价格)
     → 对齐 → 计算 10 年滚动窗口 PE 百分位 pe_pct_w10 → aks_merged_wind/{code}.parquet

与 akshare 版口径一致: 10 年滚动窗口, 第 11 年首日才能算第一个百分位。

用法: python data-store/build_wind_merged.py
"""
import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
WIND_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "wind_source")
PRICE_DIRS = [
    os.path.join(PROJECT_DIR, "data-store", "parquet", "index_price_aks"),
    os.path.join(PROJECT_DIR, "data-store", "parquet", "index_price"),
]
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "aks_merged_wind")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_YEARS = [10]

WIND_INDICES = [
    {"code": "000015", "name": "上证红利"},
    {"code": "000016", "name": "上证50"},
    {"code": "000300", "name": "沪深300"},
    {"code": "000688", "name": "科创50"},
    {"code": "000852", "name": "中证1000"},
    {"code": "000905", "name": "中证500"},
    {"code": "399006", "name": "创业板指"},
    {"code": "399330", "name": "深证100"},
    {"code": "930930", "name": "港股综合"},
    {"code": "930931", "name": "港股通50"},
    {"code": "HSI", "name": "恒生指数"},
    {"code": "HSTECH", "name": "恒生科技"},
    {"code": "NDX100", "name": "纳斯达克100"},
    {"code": "SPX500", "name": "标普500"},
]


def rolling_pct(series: np.ndarray, window_rows: int, min_samples: int = 20) -> np.ndarray:
    """10 年滚动窗口百分位。第 11 年首日才能算第一个值。"""
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
    wind_path = os.path.join(WIND_DIR, f"{code}.parquet")
    if not os.path.exists(wind_path):
        print(f"  [SKIP] wind PE 不存在: {code}")
        return pd.DataFrame()

    price_path = find_price_path(code)
    if not price_path:
        print(f"  [SKIP] 价格不存在: {code}")
        return pd.DataFrame()

    # wind PE
    pe = pd.read_parquet(wind_path)
    pe["date"] = pd.to_datetime(pe["date"])
    pe = pe.sort_values("date")[["date", "pe_ttm_wind"]].rename(columns={"pe_ttm_wind": "pe"})

    # 价格
    price = pd.read_parquet(price_path)
    price["date"] = pd.to_datetime(price["date"])
    price_col = "index_price" if "index_price" in price.columns else "close"
    price = price.sort_values("date")[["date", price_col]].rename(columns={price_col: "price"})

    # 对齐: 价格日期为主, PE backward fill
    pe_sorted = pe.sort_values("date")[["date", "pe"]].dropna(subset=["pe"])
    merged = pd.merge_asof(price, pe_sorted, on="date", direction="backward")
    merged["pe"] = merged["pe"].ffill()
    merged = merged.dropna(subset=["pe"]).reset_index(drop=True)

    if len(merged) < 100:
        print(f"  [SKIP] 有效数据不足: {len(merged)} 行")
        return pd.DataFrame()

    # 计算 10 年滚动窗口百分位
    total_days = (merged["date"].max() - merged["date"].min()).days
    total_years = total_days / 365.25
    rpy = len(merged) / max(total_years, 1)

    for w in WINDOW_YEARS:
        wr = int(w * rpy)
        min_samples = max(20, wr)
        pe_arr = merged["pe"].values.astype(float)
        merged[f"pe_pct_w{w}"] = rolling_pct(pe_arr, wr, min_samples)

    keep_cols = ["date", "price", "pe"]
    for w in WINDOW_YEARS:
        for col in [f"pe_pct_w{w}"]:
            if col in merged.columns:
                keep_cols.append(col)
    merged = merged[keep_cols]

    out_path = os.path.join(OUTPUT_DIR, f"{code}.parquet")
    merged.to_parquet(out_path, index=False)

    pct_col = "pe_pct_w10"
    valid = merged[merged[pct_col].notna()]
    first_valid = str(valid["date"].iloc[0].date()) if len(valid) > 0 else "N/A(不足10年)"
    print(f"  OK | {len(merged)}条 | {merged.date.min().date()} ~ {merged.date.max().date()} | "
          f"可交易 {len(valid)} 天, 起点 {first_valid}")

    return merged


def main():
    print(f"\nwind 版回测数据构建")
    print(f"窗口: {WINDOW_YEARS} 年滚动 | 第 11 年首日可交易\n")

    for idx in WIND_INDICES:
        print(f"[BUILD] {idx['code']} {idx['name']} ...", flush=True)
        build_one(idx["code"], idx["name"])

    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
