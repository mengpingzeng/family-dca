#!/usr/bin/env python3
"""
生成 Wind vs 蛋卷 PE 值对比数据（各自原生频率，完整历史，不做重采样/裁剪/百分位）

  - Wind: wind_source/{code}.parquet 的 pe_ttm_wind，日频，完整历史
  - 蛋卷: index_pe_dj/{code}.parquet 的 pe_ttm_dj，周频，完整历史

输出: output/pe_wind_dj.json
"""
import os, json
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
WIND_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "wind_source")
DJ_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "index_pe_dj")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDEX_ORDER = [
    ("000300", "沪深300"),
    ("000016", "上证50"),
    ("000905", "中证500"),
    ("000852", "中证1000"),
    ("000688", "科创50"),
    ("399006", "创业板指"),
    ("399330", "深证100"),
    ("000015", "上证红利"),
    ("HSI", "恒生指数"),
    ("HSTECH", "恒生科技"),
    ("NDX100", "纳斯达克100"),
    ("SPX500", "标普500"),
]


def main():
    result = {"indices": []}

    for code, name in INDEX_ORDER:
        wind_path = os.path.join(WIND_DIR, f"{code}.parquet")
        dj_path = os.path.join(DJ_DIR, f"{code}.parquet")
        if not (os.path.exists(wind_path) and os.path.exists(dj_path)):
            continue

        entry = {"code": code, "name": name, "sources": {}}

        df = pd.read_parquet(wind_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["pe_ttm_wind"]).sort_values("date")
        entry["sources"]["wind"] = {
            "freq": "daily",
            "pe": [[d.strftime("%Y-%m-%d"), round(float(v), 2)]
                   for d, v in zip(df["date"], df["pe_ttm_wind"])],
        }

        df = pd.read_parquet(dj_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["pe_ttm_dj"]).sort_values("date")
        entry["sources"]["dj"] = {
            "freq": "weekly",
            "pe": [[d.strftime("%Y-%m-%d"), round(float(v), 2)]
                   for d, v in zip(df["date"], df["pe_ttm_dj"])],
        }

        result["indices"].append(entry)
        print(f"[{code}] {name}: wind {len(entry['sources']['wind']['pe'])}点(日频), dj {len(entry['sources']['dj']['pe'])}点(周频)")

    out_path = os.path.join(OUTPUT_DIR, "pe_wind_dj.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"\n输出: {out_path}")


if __name__ == "__main__":
    main()
