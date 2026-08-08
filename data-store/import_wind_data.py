#!/usr/bin/env python3
"""
Wind 数据导入器

从邮件附件 .xlsx 解析 Wind PE TTM 日频数据 → parquet/wind_source/

输入：
    /tmp/指数数据.xlsx       (13 sheets, 每 sheet 一个指数)
    /tmp/红利指数[000015.SH].xlsx (1 sheet, 上证红利)

输出：
    data-store/parquet/wind_source/{code}.parquet  (14 个文件)
        col: date, pe_ttm_wind

用法：
    python data-store/import_wind_data.py
"""

import os
import sys
from datetime import datetime

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "wind_source")

# xlsx 中文名 → 系统标准代码
SHEET_NAME_MAP = {
    "上证50":     "000016",
    "标普500":    "SPX500",
    "沪深300":    "000300",
    "纳斯达克100": "NDX100",
    "恒生指数":    "HSI",
    "深证100":    "399330",
    "港股通50":    "930931",
    "中证500":    "000905",
    "港股综合":    "930930",
    "中证1000":   "000852",
    "科创50":     "000688",
    "恒生科技":    "HSTECH",
    "创业板指":    "399006",
    "红利指数":    "000015",
}

INDEX_NAMES = {
    "000016": "上证50", "SPX500": "标普500", "000300": "沪深300",
    "NDX100": "纳斯达克100", "HSI": "恒生指数", "399330": "深证100",
    "930931": "港股通50", "000905": "中证500", "930930": "港股综合",
    "000852": "中证1000", "000688": "科创50", "HSTECH": "恒生科技",
    "399006": "创业板指", "000015": "上证红利",
}

FILES = [
    ("/tmp/指数数据.xlsx", SHEET_NAME_MAP),
    ("/tmp/红利指数[000015.SH].xlsx", {"sheet1": "000015"}),
]


def parse_sheet(filepath: str, sheet: str, code: str) -> pd.DataFrame:
    """解析单个 sheet，返回 (date, pe_ttm_wind) DataFrame。"""
    df = pd.read_excel(filepath, sheet_name=sheet, header=None, engine="openpyxl")
    dates = pd.to_datetime(df.iloc[1:, 0], errors="coerce")
    pe_vals = pd.to_numeric(df.iloc[1:, 1], errors="coerce")
    result = pd.DataFrame({
        "date": dates,
        "pe_ttm_wind": pe_vals,
    }).dropna().sort_values("date").reset_index(drop=True)
    result["date"] = result["date"].dt.date
    return result


def main():
    print(f"\nWind 数据导入 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = 0
    for filepath, name_map in FILES:
        if not os.path.exists(filepath):
            print(f"[SKIP] 文件不存在: {filepath}")
            continue
        try:
            xl = pd.ExcelFile(filepath, engine="openpyxl")
        except Exception as e:
            print(f"[ERROR] 无法打开 {filepath}: {e}")
            continue

        for sheet in xl.sheet_names:
            if sheet not in name_map:
                print(f"[SKIP] 未知 sheet: {sheet}")
                continue
            code = name_map[sheet]
            name = INDEX_NAMES.get(code, code)

            df = parse_sheet(filepath, sheet, code)
            if df.empty:
                print(f"[SKIP] {code} {name} — 无数据")
                continue

            out_path = os.path.join(OUTPUT_DIR, f"{code}.parquet")
            df.to_parquet(out_path, index=False)
            print(f"  {code:8s} {name:10s} {len(df):>5d}条 {df.date.min()} ~ {df.date.max()}  PE {df.pe_ttm_wind.min():.2f}~{df.pe_ttm_wind.max():.2f}")
            success += 1

    print(f"\n{'='*60}")
    print(f"结果: 导入 {success} 个指数 → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
