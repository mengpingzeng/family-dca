#!/usr/bin/env python3
"""
蛋卷基金 PE TTM 历史数据下载器

从蛋卷基金 API 获取指数 PE TTM 历史数据（周频、等权PE），保存为 Parquet 文件。

数据在项目中的位置：
    ┌─────────────────────────────────────────────────┐
    │  数据采集阶段 （当前脚本）                         │
    │  ├── download_index_pe.py     → 中证系指数PE      │
    │  ├── download_missing_index.py → 非中证系指数PE    │
    │  ├── download_danjuan_pe.py   → 蛋卷PE（本周）    │  ← 你在这里
    │  └── download_etf_linked_fund.py → 基金净值       │
    ├─────────────────────────────────────────────────┤
    │  回测引擎 backtest/backtest.py                    │
    ├─────────────────────────────────────────────────┤
    │  输出 output/                                     │
    └─────────────────────────────────────────────────┘

用法:
    python download_danjuan_pe.py                  # 下载全部12个指数
    python download_danjuan_pe.py --check           # 仅验证接口连通性
    python download_danjuan_pe.py --list            # 列出所有指数代码
    python download_danjuan_pe.py --code 000300     # 下载指定指数
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import requests

# ============================================================================
# 指数配置
# ============================================================================

DANJUAN_INDEXES = [
    # A股 - 上交所
    {"code": "000300", "name": "沪深300",   "dj_code": "SH000300", "dj_available": True},
    {"code": "000905", "name": "中证500",   "dj_code": "SH000905", "dj_available": True},
    {"code": "000852", "name": "中证1000",  "dj_code": "SH000852", "dj_available": True},
    {"code": "000016", "name": "上证50",    "dj_code": "SH000016", "dj_available": True},
    {"code": "000688", "name": "科创50",    "dj_code": "SH000688", "dj_available": True},
    {"code": "000015", "name": "上证红利",  "dj_code": "SH000015", "dj_available": True},
    {"code": "000922", "name": "中证红利",  "dj_code": "SH000922", "dj_available": True},
    # A股 - 深交所
    {"code": "399006", "name": "创业板指",  "dj_code": "SZ399006", "dj_available": True},
    {"code": "399330", "name": "深证100",   "dj_code": "SZ399330", "dj_available": True},
    # 港股
    {"code": "HSI",    "name": "恒生指数",  "dj_code": "HKHSI",    "dj_available": True},
    {"code": "HSTECH", "name": "恒生科技",  "dj_code": "HKHSTECH", "dj_available": True},
    # 美股
    {"code": "NDX100", "name": "纳斯达克100", "dj_code": "NDX",    "dj_available": True},
    {"code": "SPX500", "name": "标普500",     "dj_code": "SP500",  "dj_available": True},
    # 蛋卷无PE数据，需中证官方PE兜底
    {"code": "930930", "name": "港股综合",  "dj_code": None,       "dj_available": False},
    {"code": "930931", "name": "港股通50",  "dj_code": None,       "dj_available": False},
]

DJ_API_BASE = "https://danjuanfunds.com/djapi/index_eva/pe_history"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_TIMEOUT = 15

# 默认输出目录（相对于脚本所在目录的上一级，即项目根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # /mnt/data/zmp/


# ============================================================================
# 核心下载函数
# ============================================================================

def fetch_danjuan_pe(danjuan_code: str) -> pd.DataFrame:
    """
    从蛋卷基金获取单个指数的 PE TTM 历史数据（周频）。

    Args:
        danjuan_code: 蛋卷格式代码，如 "SH000300"

    Returns:
        DataFrame(date, pe_ttm_dj)，按日期升序排列。空DataFrame表示无数据。
    """
    url = f"{DJ_API_BASE}/{danjuan_code}"

    try:
        r = requests.get(
            url,
            params={"day": "all"},
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] 请求失败: {e}", file=sys.stderr)
        return pd.DataFrame()

    try:
        data = r.json()
    except ValueError:
        print(f"  [ERROR] 响应非JSON格式", file=sys.stderr)
        return pd.DataFrame()

    if "data" not in data or not data["data"]:
        return pd.DataFrame()

    pe_items = data["data"].get("index_eva_pe_growths", [])
    if not pe_items:
        return pd.DataFrame()

    df = pd.DataFrame(pe_items)
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    df["pe_ttm_dj"] = pd.to_numeric(df["pe"], errors="coerce")
    df = df[["date", "pe_ttm_dj"]].dropna(subset=["pe_ttm_dj"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================================================================
# 下载流程
# ============================================================================

def download_index(index_info: dict, output_dir: str) -> bool:
    """下载单个指数PE数据并保存为Parquet。返回True表示成功。"""
    code = index_info["code"]
    name = index_info["name"]
    dj_code = index_info["dj_code"]

    if not index_info["dj_available"]:
        print(f"[SKIP] {code} {name} — 蛋卷无此指数PE，需中证官方PE兜底")
        return False

    print(f"[FETCH] {code} {name} ({dj_code}) ...", end=" ", flush=True)
    df = fetch_danjuan_pe(dj_code)

    if df.empty:
        print("无数据")
        return False

    out_path = os.path.join(output_dir, f"{code}.parquet")
    df.to_parquet(out_path, index=False)

    date_min = df["date"].min()
    date_max = df["date"].max()
    pe_min = df["pe_ttm_dj"].min()
    pe_max = df["pe_ttm_dj"].max()
    pe_latest = df.iloc[-1]["pe_ttm_dj"]

    print(f"OK | {len(df)}条 | {date_min} ~ {date_max} | "
          f"PE {pe_min:.2f}~{pe_max:.2f} | 最新 {pe_latest:.4f}")

    return True


def download_all(output_dir: str):
    """下载全部有蛋卷数据的指数PE。"""
    available = [idx for idx in DANJUAN_INDEXES if idx["dj_available"]]
    skipped = [idx for idx in DANJUAN_INDEXES if not idx["dj_available"]]

    print(f"\n蛋卷PE数据下载 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"共 {len(DANJUAN_INDEXES)} 个指数，其中蛋卷可用 {len(available)} 个，"
          f"需兜底 {len(skipped)} 个\n")

    success = 0
    fail = 0

    for idx in available:
        if download_index(idx, output_dir):
            success += 1
        else:
            fail += 1

    for idx in skipped:
        print(f"[SKIP] {idx['code']} {idx['name']} — 蛋卷无此指数PE，需中证官方PE兜底")

    print(f"\n{'='*60}")
    print(f"结果: 成功 {success}, 失败 {fail}, 跳过 {len(skipped)}")
    print(f"输出目录: {output_dir}")


def check():
    """验证接口连通性，仅下载沪深300一条记录作抽样检查。"""
    print("蛋卷PE接口连通性检查\n")
    idx = DANJUAN_INDEXES[0]  # 沪深300
    print(f"测试指数: {idx['name']} ({idx['dj_code']})")

    df = fetch_danjuan_pe(idx["dj_code"])
    if df.empty:
        print("[FAIL] 接口不可用或返回空数据")
        sys.exit(1)

    print(f"[OK] 接口正常，获取到 {len(df)} 条记录")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  PE范围:   {df['pe_ttm_dj'].min():.2f} ~ {df['pe_ttm_dj'].max():.2f}")
    print(f"  最新PE:   {df.iloc[-1]['pe_ttm_dj']:.4f} ({df.iloc[-1]['date']})")


def list_indices():
    """列出所有指数代码及蛋卷可用状态。"""
    print(f"{'指数代码':<8} {'指数名称':<10} {'蛋卷代码':<10} {'蛋卷可用'}")
    print("-" * 42)
    for idx in DANJUAN_INDEXES:
        available = "是" if idx["dj_available"] else "否（需兜底）"
        dj_code = idx["dj_code"] or "-"
        print(f"{idx['code']:<8} {idx['name']:<10} {dj_code:<10} {available}")


# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="蛋卷基金 PE TTM 历史数据下载器"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="验证接口连通性（仅下载沪深300抽样检查）"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有指数代码及蛋卷可用状态"
    )
    parser.add_argument(
        "--code", type=str, default=None,
        help="下载指定指数代码（如 000300）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出目录（默认 data-store/parquet/index_pe/）"
    )
    args = parser.parse_args()

    output_dir = args.output or os.path.join(PROJECT_DIR, "data-store", "parquet", "index_pe_dj")
    os.makedirs(output_dir, exist_ok=True)

    if args.check:
        check()
    elif args.list:
        list_indices()
    elif args.code:
        matched = [i for i in DANJUAN_INDEXES if i["code"] == args.code]
        if not matched:
            print(f"[ERROR] 未知指数代码: {args.code}", file=sys.stderr)
            print("使用 --list 查看可用指数代码", file=sys.stderr)
            sys.exit(1)
        idx = matched[0]
        if not idx["dj_available"]:
            print(f"[SKIP] {idx['code']} {idx['name']} — 蛋卷无此指数PE，需中证官方PE兜底")
            sys.exit(1)
        success = download_index(idx, output_dir)
        sys.exit(0 if success else 1)
    else:
        download_all(output_dir)


if __name__ == "__main__":
    main()
