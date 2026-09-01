#!/usr/bin/env python3
"""
生成三数据源 PE 值 + PE 百分位 对比数据（统一周频 + 统一 5 年窗口）

对三源都有的指数(沪深300/上证50/中证500/上证红利/中证1000)：
  - PE 值: akshare(pe) / wind(pe_ttm_wind) / 蛋卷(pe_ttm_dj)
  - PE 百分位: 统一 5 年滚动窗口
  - 频率: akshare/wind 日频重采样为周频，与蛋卷(周频)对齐

输出: output/pe_comparison.json
"""
import os, json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
AK_MERGED = os.path.join(PROJECT_DIR, "data-store", "parquet", "aks_merged")
WIND_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "wind_source")
DJ_PE_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "index_pe_dj")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW = 5
WEEKS_PER_YEAR = 52

COMPARE_INDICES = [
    {"code": "000300", "name": "沪深300"},
    {"code": "000016", "name": "上证50"},
    {"code": "000905", "name": "中证500"},
    {"code": "000015", "name": "上证红利"},
    {"code": "000852", "name": "中证1000"},
]


def rolling_pct(series: np.ndarray, window_rows: int, min_samples: int) -> np.ndarray:
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


def to_weekly(df: pd.DataFrame, value_cols) -> pd.DataFrame:
    """重采样到周频，对齐到周日（W-SUN，与蛋卷周频锚点一致）。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    weekly = df[value_cols].resample("W-SUN").last()
    weekly = weekly.dropna(how="all").reset_index()
    return weekly


def main():
    result = {"window": WINDOW, "freq": "weekly", "indices": []}

    for idx in COMPARE_INDICES:
        code = idx["code"]
        name = idx["name"]
        print(f"[{code}] {name} ...", flush=True)
        entry = {"code": code, "name": name, "sources": {}}

        # 1. akshare (读日频 pe + pe_pct_w5 → 周频)
        ak_path = os.path.join(AK_MERGED, f"{code}.parquet")
        if os.path.exists(ak_path):
            df = pd.read_parquet(ak_path)
            pct_col = f"pe_pct_w{WINDOW}"
            wk = to_weekly(df[["date", "pe", pct_col]], ["pe", pct_col])
            entry["sources"]["akshare"] = {
                "pe": [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in zip(wk["date"], wk["pe"]) if pd.notna(v)],
                "pe_pct": [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in zip(wk["date"], wk[pct_col]) if pd.notna(v)],
            }

        # 2. wind (日频 pe_ttm_wind → 算 5yr 百分位 → 周频)
        wind_path = os.path.join(WIND_DIR, f"{code}.parquet")
        if os.path.exists(wind_path):
            df = pd.read_parquet(wind_path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["pe_ttm_wind"]).sort_values("date").reset_index(drop=True)
            # 日频 5 年窗口百分位
            total_days = (df["date"].max() - df["date"].min()).days
            rpy = len(df) / max(total_days / 365.25, 1)
            df = df.copy()
            df["pe_pct"] = rolling_pct(df["pe_ttm_wind"].values.astype(float), int(WINDOW * rpy), int(WINDOW * rpy))
            wk = to_weekly(df[["date", "pe_ttm_wind", "pe_pct"]], ["pe_ttm_wind", "pe_pct"])
            entry["sources"]["wind"] = {
                "pe": [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in zip(wk["date"], wk["pe_ttm_wind"]) if pd.notna(v)],
                "pe_pct": [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in zip(wk["date"], wk["pe_pct"]) if pd.notna(v)],
            }

        # 3. 蛋卷 (周频 pe_ttm_dj → 算 5yr 百分位 → 对齐到周日)
        dj_path = os.path.join(DJ_PE_DIR, f"{code}.parquet")
        if os.path.exists(dj_path):
            df = pd.read_parquet(dj_path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["pe_ttm_dj"]).sort_values("date").reset_index(drop=True)
            df = df.copy()
            df["pe_pct"] = rolling_pct(df["pe_ttm_dj"].values.astype(float), int(WINDOW * WEEKS_PER_YEAR), int(WINDOW * WEEKS_PER_YEAR))
            wk = to_weekly(df[["date", "pe_ttm_dj", "pe_pct"]], ["pe_ttm_dj", "pe_pct"])
            entry["sources"]["dj"] = {
                "pe": [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in zip(wk["date"], wk["pe_ttm_dj"]) if pd.notna(v)],
                "pe_pct": [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in zip(wk["date"], wk["pe_pct"]) if pd.notna(v)],
            }

        # 裁剪到三源共同时间区间
        pe_starts = [src["pe"][0][0] for src in entry["sources"].values() if src.get("pe")]
        pct_starts = [src["pe_pct"][0][0] for src in entry["sources"].values() if src.get("pe_pct")]
        pe_start = max(pe_starts) if pe_starts else None
        pct_start = max(pct_starts) if pct_starts else None
        for sk, src in entry["sources"].items():
            if pe_start:
                src["pe"] = [[d, v] for d, v in src.get("pe", []) if d >= pe_start]
            if pct_start:
                src["pe_pct"] = [[d, v] for d, v in src.get("pe_pct", []) if d >= pct_start]

        result["indices"].append(entry)

    out_path = os.path.join(OUTPUT_DIR, "pe_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False)

    for idx in result["indices"]:
        srcs = list(idx["sources"].keys())
        print(f"  {idx['name']}: 源={srcs} ", end="")
        for s in srcs:
            print(f"{s}(PE {len(idx['sources'][s]['pe'])}点, pct {len(idx['sources'][s]['pe_pct'])}点) ", end="")
        print()

    print(f"\n输出: {out_path}")


if __name__ == "__main__":
    main()
