#!/usr/bin/env python3
"""
中证指数官网 PE TTM 历史数据下载器

从 csindex.com.cn 获取指数 PE TTM 数据（整体法加权PE），保存为 Parquet 文件。

数据源说明：
    - 主力源：oss-ch.csindex.com.cn 的 indicator.xls，每日 PE2（计算用股本 P/E）
    - 备选源：csindex index-perf JSON API（仅供手动切换，防 WAF 反爬）

数据在项目中的位置：
    ┌─────────────────────────────────────────────────┐
    │  数据采集阶段                                     │
    │  ├── download_index_pe.py     → 中证系指数PE ★   │
    │  ├── download_danjuan_pe.py   → 蛋卷PE（等权）    │
    │  ├── download_missing_index.py → 非中证系指数PE   │
    │  └── download_etf_linked_fund.py → 基金净值       │
    ├─────────────────────────────────────────────────┤
    │  回测引擎 backtest/backtest.py                    │
    └─────────────────────────────────────────────────┘

用法:
    python download_index_pe.py                  # 下载全部16个指数
    python download_index_pe.py --check           # 验证接口连通性
    python download_index_pe.py --list            # 列出所有指数
    python download_index_pe.py --code 000510     # 下载指定指数
"""

import argparse
import os
import sys
import time
from datetime import datetime
from io import BytesIO

import pandas as pd
import requests

# ============================================================================
# 指数配置
# ============================================================================

CSI_INDEXES = [
    # 宽基
    {"code": "000300", "name": "沪深300"},
    {"code": "000905", "name": "中证500"},
    {"code": "000852", "name": "中证1000"},
    {"code": "000016", "name": "上证50"},
    {"code": "000688", "name": "科创50"},
    {"code": "000510", "name": "中证A500"},
    {"code": "399006", "name": "创业板指"},
    {"code": "399330", "name": "深证100"},
    # 红利
    {"code": "000015", "name": "上证红利"},
    {"code": "000922", "name": "中证红利"},
    {"code": "930955", "name": "红利低波100"},
    # 港股通
    {"code": "930915", "name": "港股通高股息",   "alt_code": "930914"},
    {"code": "930930", "name": "港股综合"},
    {"code": "930931", "name": "港股通50"},
    {"code": "931573", "name": "港股通科技"},
    # 质量成长
    {"code": "930939", "name": "中证质量成长"},
]

INDICATOR_BASE = (
    "https://oss-ch.csindex.com.cn/static/html/csindex/"
    "public/uploads/file/autofile/indicator"
)
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 3.0  # seconds between requests to avoid WAF rate limiting

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


# ============================================================================
# 核心下载 — indicator.xls（当前主力源）
# ============================================================================

def fetch_csi_pe_indicator(index_code: str, alt_code: str = None) -> pd.DataFrame:
    """
    从中证指数 OSS indicator.xls 获取 PE TTM 数据（日频，近~20 日）。

    indicator.xls 列结构：
        日期 / 指数代码 / … / PE1(总股本) / PE2(计算用股本) / 股息率1 / 股息率2

    我们取 PE2（计算用股本 P/E）作为 pe_ttm_csi。

    Args:
        index_code: 指数代码，如 "000300"
        alt_code:   备选代码，如 930915(CNY) 回报 404 时改用 930914(HKD)

    Returns:
        DataFrame(date, pe_ttm_csi)，按日期升序。
    """
    codes_to_try = [index_code]
    if alt_code:
        codes_to_try.append(alt_code)

    for code in codes_to_try:
        url = f"{INDICATOR_BASE}/{code}indicator.xls"
        try:
            r = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
        except requests.RequestException:
            continue

        try:
            df = pd.read_excel(BytesIO(r.content), engine="xlrd")
        except Exception:
            continue

        if df.shape[0] < 2 or df.shape[1] < 8:
            continue

        # 列0=日期(YYYYMMDD), 列6=PE1, 列7=PE2
        rows = []
        for _, row in df.iloc[1:].iterrows():
            try:
                date_str = str(int(row.iloc[0]))
                pe_val = float(row.iloc[7])
                rows.append({
                    "date": pd.to_datetime(date_str, format="%Y%m%d").date(),
                    "pe_ttm_csi": pe_val,
                })
            except (ValueError, TypeError):
                continue

        if not rows:
            continue

        result = pd.DataFrame(rows)
        result = result.sort_values("date").reset_index(drop=True)
        return result

    return pd.DataFrame()


# ============================================================================
# 备选源 — index-perf JSON API（含全量历史，需防反爬）
# ============================================================================

def fetch_csi_pe_perf(index_code: str) -> pd.DataFrame:
    """
    从中证指数 index-perf JSON 接口获取 PE TTM 历史数据（日频，2018至今）。

    注意：此接口有 WAF 防护，高并发会触发 403 封禁。
    作为 indicator.xls 的备选，不要在生产中连续调用。
    """
    url = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
    params = {
        "indexCode": index_code,
        "startDate": "20180101",
        "endDate": "20260805",
    }
    try:
        r = requests.get(
            url,
            params=params,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.csindex.com.cn/",
            },
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame()

    try:
        payload = r.json()
    except ValueError:
        return pd.DataFrame()

    items = payload.get("data") or []
    if not items:
        return pd.DataFrame()

    rows = []
    for item in items:
        peg = item.get("peg")
        if peg is None:
            continue
        rows.append({
            "date": pd.to_datetime(item["tradeDate"], format="%Y%m%d").date(),
            "pe_ttm_csi": float(peg),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values("date").reset_index(drop=True)


# ============================================================================
# 统一下载入口
# ============================================================================

def fetch_csi_pe(index_info: dict) -> pd.DataFrame:
    """
    统一下载入口：优先 index-perf（全量历史），失败回退 indicator.xls。
    """
    code = index_info["code"]
    alt_code = index_info.get("alt_code")

    df = fetch_csi_pe_perf(code)
    if not df.empty:
        return df

    return fetch_csi_pe_indicator(code, alt_code=alt_code)


# ============================================================================
# 下载流程
# ============================================================================

def download_index(index_info: dict, output_dir: str) -> bool:
    """下载单个指数PE数据并保存为Parquet。返回True表示成功。"""
    code = index_info["code"]
    name = index_info["name"]

    print(f"[FETCH] {code} {name} ...", end=" ", flush=True)
    df = fetch_csi_pe(index_info)

    if df.empty:
        print("无数据")
        return False

    out_path = os.path.join(output_dir, f"{code}.parquet")
    df.to_parquet(out_path, index=False)

    date_min = df["date"].min()
    date_max = df["date"].max()
    pe_min = df["pe_ttm_csi"].min()
    pe_max = df["pe_ttm_csi"].max()
    pe_latest = df.iloc[-1]["pe_ttm_csi"]

    print(f"OK | {len(df):>4d}条 | {date_min} ~ {date_max} | "
          f"PE {pe_min:.2f}~{pe_max:.2f} | 最新 {pe_latest:.2f}")

    return True


def download_all(output_dir: str):
    """下载全部中证指数PE。"""
    print(f"\n中证指数PE下载 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"共 {len(CSI_INDEXES)} 个指数\n")

    success = 0
    fail = 0

    for i, idx in enumerate(CSI_INDEXES):
        if download_index(idx, output_dir):
            success += 1
        else:
            fail += 1
        if i < len(CSI_INDEXES) - 1:
            time.sleep(REQUEST_DELAY)

    print(f"\n{'='*60}")
    print(f"结果: 成功 {success}, 失败 {fail}")
    print(f"输出目录: {output_dir}")


def check():
    """验证接口连通性。"""
    print("中证指数官网接口连通性检查\n")
    idx = CSI_INDEXES[0]  # 沪深300
    print(f"测试指数: {idx['name']} ({idx['code']})")

    df = fetch_csi_pe(idx)
    if df.empty:
        print("[FAIL] 接口不可用或返回空数据")
        sys.exit(1)

    print(f"[OK] 接口正常，获取到 {len(df)} 条记录")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  PE范围:   {df['pe_ttm_csi'].min():.2f} ~ {df['pe_ttm_csi'].max():.2f}")
    print(f"  最新PE:   {df.iloc[-1]['pe_ttm_csi']:.2f} ({df.iloc[-1]['date']})")


def list_indices():
    """列出所有中证指数代码。"""
    print(f"{'指数代码':<8} {'指数名称':<12} {'备选代码':<8}")
    print("-" * 30)
    for idx in CSI_INDEXES:
        alt = idx.get("alt_code", "-")
        print(f"{idx['code']:<8} {idx['name']:<12} {alt:<8}")
    print(f"\n共 {len(CSI_INDEXES)} 个指数")


# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="中证指数官网 PE TTM 数据下载器"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="验证接口连通性（下载沪深300抽样检查）"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有指数代码"
    )
    parser.add_argument(
        "--code", type=str, default=None,
        help="下载指定指数代码（如 000510）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出目录（默认 data-store/parquet/index_pe_csi/）"
    )
    args = parser.parse_args()

    output_dir = args.output or os.path.join(PROJECT_DIR, "data-store", "parquet", "index_pe_csi")
    os.makedirs(output_dir, exist_ok=True)

    if args.check:
        check()
    elif args.list:
        list_indices()
    elif args.code:
        matched = [i for i in CSI_INDEXES if i["code"] == args.code]
        if not matched:
            print(f"[ERROR] 未知指数代码: {args.code}", file=sys.stderr)
            print("使用 --list 查看可用指数代码", file=sys.stderr)
            sys.exit(1)
        success = download_index(matched[0], output_dir)
        sys.exit(0 if success else 1)
    else:
        download_all(output_dir)


if __name__ == "__main__":
    main()
