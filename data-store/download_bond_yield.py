#!/usr/bin/env python3
"""
国债收益率下载器（中国 + 美国 10年期）

从东方财富数据中心获取中债/美债 10Y 收益率历史数据（日频）。

输出：
    bond_yield/cn_10y_bond.parquet  (中国10Y，EMM00166469)
    bond_yield/us_10y_bond.parquet  (美国10Y，EMG00001310)

用法:
    python download_bond_yield.py                  # 下载中美 10Y
    python download_bond_yield.py --check           # 验证接口
"""

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests

API_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://data.eastmoney.com/",
}
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 2.0

BOND_CONFIGS = {
    "cn": {
        "column": "EMM00166469",
        "filename": "cn_10y_bond.parquet",
        "label": "中国10Y",
        "output_col": "bond_yield_cn",
    },
    "us": {
        "column": "EMG00001310",
        "filename": "us_10y_bond.parquet",
        "label": "美国10Y",
        "output_col": "bond_yield_us",
    },
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "data-store", "parquet", "bond_yield")


def fetch_all(yield_column: str) -> pd.DataFrame:
    """分页获取全部国债收益率数据。"""
    all_rows = []
    page = 1
    total_pages = None

    base_params = {
        "reportName": "RPTA_WEB_TREASURYYIELD",
        "columns": f"SOLAR_DATE,{yield_column}",
        "pageSize": 500,
        "sortColumns": "SOLAR_DATE",
        "sortTypes": 1,
    }

    while True:
        params = {**base_params, "pageNumber": page}
        try:
            r = requests.get(API_URL, params=params, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARN] 第{page}页请求失败: {e}", file=sys.stderr)
            break

        data = r.json()
        result = data.get("result")
        if not result:
            break
        if total_pages is None:
            total_pages = result.get("pages", 0)

        for item in result.get("data", []):
            val = item.get(yield_column)
            if val is None:
                continue
            all_rows.append({
                "date": pd.to_datetime(item["SOLAR_DATE"]).date(),
                "bond_yield": float(val),
            })

        page += 1
        if page > total_pages:
            break
        time.sleep(REQUEST_DELAY)

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows).sort_values("date").reset_index(drop=True)


def download_one(config: dict, output_dir: str):
    """下载单个国债品种。"""
    print(f"\n[{config['label']}] 开始下载...")
    df = fetch_all(config["column"])
    if df.empty:
        print(f"  [FAIL] 无数据")
        return False

    df = df.rename(columns={"bond_yield": config["output_col"]})
    out_path = os.path.join(output_dir, config["filename"])
    df.to_parquet(out_path, index=False)
    print(f"  [OK] {len(df)}条 | {df['date'].min()} ~ {df['date'].max()} | "
          f"{df[config['output_col']].min():.4f}% ~ {df[config['output_col']].max():.4f}% | "
          f"最新 {df.iloc[-1]['date']} = {df.iloc[-1][config['output_col']]:.4f}%")
    return True


def check():
    """验证接口连通性。"""
    print("国债收益率接口检查\n")
    base_params = {
        "reportName": "RPTA_WEB_TREASURYYIELD",
        "columns": "SOLAR_DATE,EMM00166469,EMG00001310",
        "pageSize": 5,
        "pageNumber": 1,
        "sortColumns": "SOLAR_DATE",
        "sortTypes": -1,
    }
    try:
        r = requests.get(API_URL, params=base_params, headers=REQUEST_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", {})
        print(f"[OK] 共 {result.get('count', 0)} 条记录")
        for item in result.get("data", []):
            cn = item.get("EMM00166469")
            us = item.get("EMG00001310")
            print(f"  {item['SOLAR_DATE'][:10]}  CN={cn:.4f}%  US={us:.4f}%")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="国债收益率下载器（中/美 10Y）")
    parser.add_argument("--check", action="store_true", help="验证接口连通性")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    output_dir = args.output or DEFAULT_OUTPUT
    os.makedirs(output_dir, exist_ok=True)

    if args.check:
        check()
        return

    print(f"\n国债收益率下载 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for key, cfg in BOND_CONFIGS.items():
        download_one(cfg, output_dir)


if __name__ == "__main__":
    main()
