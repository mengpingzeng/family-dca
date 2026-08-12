#!/usr/bin/env python3
"""
akshare PE/PB/FED 数据下载器（方式1：乐咕乐股原始数据）

从 legulegu.com API 获取指数 PE/PB/FED 数据，保存为 Parquet。

与现有 download_danjuan_pe.py 等完全独立，输出到 parquet/aks/。

数据源：legulegu.com（乐咕乐股），即 akshare 的 stock_index_pe_lg() 等函数的底层 API。

覆盖范围：
  PE: 上证50、沪深300、中证500、中证1000 (6列PE口名)
  PB: 上证50、沪深300、中证500、中证1000 (3列PB口名)
  全市场: A股整体PE/PB中位数
  FED: 沪深300 股债利差

用法:
    python download_akshare_pe.py                  # 下载全部
    python download_akshare_pe.py --check           # 验证接口
    python download_akshare_pe.py --list            # 列出指数
    python download_akshare_pe.py --code 000300     # 下载指定指数
"""

import argparse
import os
import sys
import time
from datetime import datetime
from hashlib import md5

import pandas as pd
import requests
from lxml import etree

# ============================================================================
# 指数配置
# ============================================================================

LEGU_INDEXES = [
    {"code": "000016", "name": "上证50",   "legu_code": "000016.SH"},
    {"code": "000300", "name": "沪深300",  "legu_code": "000300.SH"},
    {"code": "000905", "name": "中证500",  "legu_code": "000905.SH"},
    {"code": "000852", "name": "中证1000", "legu_code": "000852.SH"},
]

LEGU_REFERER = "https://legulegu.com/stockdata/sz50-ttm-lyr"
FED_REFERER = "https://legulegu.com/stockdata/equity-bond-spread"

PE_API = "https://legulegu.com/api/stockdata/index-basic-pe"
PB_API = "https://legulegu.com/api/stockdata/index-basic-pb"
FED_API = "https://legulegu.com/api/stockdata/equity-bond-spread"

# 市场 API 必须在 www. 子域名下调用，否则返回空
MARKET_REFERER = "https://www.legulegu.com/stockdata/a-ttm-lyr"
MARKET_PE_API = "https://www.legulegu.com/api/stock-data/market-ttm-lyr"
MARKET_PB_API = "https://www.legulegu.com/api/stock-data/market-index-pb"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 1.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# ============================================================================
# 工具函数
# ============================================================================

def _get_token():
    """生成 legulegu API 的 token（今日日期的 MD5）。"""
    return md5(datetime.now().date().isoformat().encode()).hexdigest()


def _init_session(referer):
    """访问 legulegu 页面，获取 CSRF cookie 和 token。"""
    s = requests.Session()
    s.headers.update(HEADERS)
    r = s.get(referer, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    root = etree.HTML(r.text)
    meta = root.xpath('//meta[@name="_csrf"]')
    if meta:
        s.headers["X-CSRF-Token"] = meta[0].get("content")
    return s

# ============================================================================
# 数据获取
# ============================================================================

def fetch_index_pe(legu_code: str) -> pd.DataFrame:
    """获取指数 PE 数据（6列PE口名）。"""
    session = _init_session(LEGU_REFERER)
    url = PE_API
    params = {"token": _get_token(), "indexCode": legu_code}
    r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json().get("data", [])

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(columns={
        "ttmPe": "pe_ew",
        "addTtmPe": "pe_wgt",
        "middleTtmPe": "pe_med",
        "lyrPe": "pe_ew_lyr",
        "addLyrPe": "pe_wgt_lyr",
        "middleLyrPe": "pe_med_lyr",
    })
    cols = ["date", "pe_ew", "pe_wgt", "pe_med", "pe_ew_lyr", "pe_wgt_lyr", "pe_med_lyr"]
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("date").reset_index(drop=True)


def fetch_index_pb(legu_code: str) -> pd.DataFrame:
    """获取指数 PB 数据（3列PB口名）。"""
    session = _init_session(LEGU_REFERER)
    url = PB_API
    params = {"token": _get_token(), "indexCode": legu_code}
    r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json().get("data", [])

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(columns={
        "pb": "pb_ew",
        "addPb": "pb_wgt",
        "middlePb": "pb_med",
    })
    cols = ["date", "pb_ew", "pb_wgt", "pb_med"]
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("date").reset_index(drop=True)


def fetch_market_pe() -> pd.DataFrame:
    """获取全 A 股等权/中位数 PE TTM 数据。"""
    session = _init_session(MARKET_REFERER)
    url = MARKET_PE_API
    params = {"marketId": "5", "token": _get_token()}
    r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json().get("data", [])

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(columns={
        "averagePETTM": "pe_ew",
        "middlePETTM": "pe_med",
        "averagePELYR": "pe_ew_lyr",
        "middlePELYR": "pe_med_lyr",
    })
    cols = ["date", "pe_ew", "pe_med", "pe_ew_lyr", "pe_med_lyr", "close"]
    df = df[[c for c in cols if c in df.columns]]
    df["type"] = "market_ttm"
    return df.sort_values("date").reset_index(drop=True)


def fetch_market_pb() -> pd.DataFrame:
    """获取全 A 股等权/中位数 PB 数据。"""
    session = _init_session(MARKET_REFERER)
    url = MARKET_PB_API
    params = {"marketId": "ALL", "token": _get_token()}
    r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json().get("data", [])

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(columns={
        "equalWeightAveragePB": "pb_ew",
        "middlePB": "pb_med",
        "weightingAveragePB": "pb_wgt",
    })
    cols = ["date", "pb_ew", "pb_med", "pb_wgt", "close"]
    df = df[[c for c in cols if c in df.columns]]
    df["type"] = "market_pb"
    return df.sort_values("date").reset_index(drop=True)


def fetch_fed() -> pd.DataFrame:
    """获取沪深300 FED 股债利差数据。"""
    session = _init_session(FED_REFERER)
    url = FED_API
    params = {"token": _get_token(), "code": "000300.SH"}
    r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json().get("data", [])

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(columns={
        "peSpread": "fed",
        "peSpreadAverage": "fed_ma",
        "close": "csi300_close",
    })
    cols = ["date", "fed", "fed_ma", "csi300_close"]
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("date").reset_index(drop=True)

# ============================================================================
# 下载流程
# ============================================================================

def _save_and_report(df, out_path, name, value_col, col_name):
    """保存 DataFrame 并打印摘要。"""
    if df.empty:
        print("  无数据")
        return False
    df.to_parquet(out_path, index=False)
    series = df[value_col]
    print(f"  OK | {len(df):>5d}条 | {df['date'].min()} ~ {df['date'].max()} | "
          f"{col_name} {series.min():.2f}~{series.max():.2f} | 最新 {series.iloc[-1]:.2f}")
    return True


def download_all(output_dir: str):
    """下载全部 akshare 数据。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nakshare 数据下载（legulegu API）— {now}")
    print(f"输出目录: {output_dir}\n")

    pe_dir = os.path.join(output_dir, "pe")
    pb_dir = os.path.join(output_dir, "pb")
    market_dir = os.path.join(output_dir, "market")
    fed_dir = os.path.join(output_dir, "fed")
    for d in [pe_dir, pb_dir, market_dir, fed_dir]:
        os.makedirs(d, exist_ok=True)

    # --- 指数 PE ---
    print("=" * 50)
    print("  指数 PE（等权/加权/中位数 TTM + 静态）")
    for idx in LEGU_INDEXES:
        print(f"[FETCH] {idx['code']} {idx['name']} ...", end="", flush=True)
        try:
            df = fetch_index_pe(idx["legu_code"])
            _save_and_report(df, os.path.join(pe_dir, f"{idx['code']}.parquet"),
                             idx["name"], "pe_ew", "PE(TTM等权)")
        except Exception as e:
            print(f"  [FAIL] {e}")
        time.sleep(REQUEST_DELAY)

    # --- 指数 PB ---
    print("\n" + "=" * 50)
    print("  指数 PB（等权/加权/中位数）")
    for idx in LEGU_INDEXES:
        print(f"[FETCH] {idx['code']} {idx['name']} ...", end="", flush=True)
        try:
            df = fetch_index_pb(idx["legu_code"])
            _save_and_report(df, os.path.join(pb_dir, f"{idx['code']}.parquet"),
                             idx["name"], "pb_ew", "PB(等权)")
        except Exception as e:
            print(f"  [FAIL] {e}")
        time.sleep(REQUEST_DELAY)

    # --- 全市场 PE ---
    print("\n" + "=" * 50)
    print("  全市场 PE")
    print("[FETCH] 全A股PE ...", end="", flush=True)
    try:
        df = fetch_market_pe()
        _save_and_report(df, os.path.join(market_dir, "market_pe.parquet"),
                         "全A股PE", "pe_ew", "PE(等权)")
    except Exception as e:
        print(f"  [FAIL] {e}")
    time.sleep(REQUEST_DELAY)

    # --- 全市场 PB ---
    print("\n  全市场 PB")
    print("[FETCH] 全A股PB ...", end="", flush=True)
    try:
        df = fetch_market_pb()
        _save_and_report(df, os.path.join(market_dir, "market_pb.parquet"),
                         "全A股PB", "pb_ew", "PB(等权)")
    except Exception as e:
        print(f"  [FAIL] {e}")
    time.sleep(REQUEST_DELAY)

    # --- FED ---
    print("\n" + "=" * 50)
    print("  FED 股债利差")
    print("[FETCH] 沪深300股债利差 ...", end="", flush=True)
    try:
        df = fetch_fed()
        _save_and_report(df, os.path.join(fed_dir, "csi300_fed.parquet"),
                         "FED", "fed", "FED")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print(f"\n{'='*50}")
    print("下载完成")


def check():
    """验证接口连通性。"""
    print("legulegu API 接口连通性检查\n")
    print("1. 指数PE接口 ...", end=" ", flush=True)
    df = fetch_index_pe("000300.SH")
    if df.empty:
        print("[FAIL]")
        sys.exit(1)
    print(f"[OK] {len(df)}条, {df['date'].min()}~{df['date'].max()}")

    print("2. 指数PB接口 ...", end=" ", flush=True)
    df = fetch_index_pb("000300.SH")
    if df.empty:
        print("[FAIL]")
        sys.exit(1)
    print(f"[OK] {len(df)}条")

    print("3. 全市场PE接口 ...", end=" ", flush=True)
    df = fetch_market_pe()
    if df.empty:
        print("[FAIL]")
        sys.exit(1)
    print(f"[OK] {len(df)}条")

    print("4. 全市场PB接口 ...", end=" ", flush=True)
    df = fetch_market_pb()
    if df.empty:
        print("[FAIL]")
        sys.exit(1)
    print(f"[OK] {len(df)}条")

    print("5. FED接口 ...", end=" ", flush=True)
    df = fetch_fed()
    if df.empty:
        print("[FAIL]")
        sys.exit(1)
    print(f"[OK] {len(df)}条")

    print("\n全部接口正常")


def list_indices():
    """列出所有指数。"""
    print(f"{'指数代码':<8} {'指数名称':<10} {'legu_code'}")
    print("-" * 32)
    for idx in LEGU_INDEXES:
        print(f"{idx['code']:<8} {idx['name']:<10} {idx['legu_code']}")
    print(f"\n共 {len(LEGU_INDEXES)} 个指数 + 全市场PE/PB + FED")

# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="akshare PE/PB/FED 数据下载器（legulegu API）"
    )
    parser.add_argument("--check", action="store_true", help="验证接口连通性")
    parser.add_argument("--list", action="store_true", help="列出所有指数")
    parser.add_argument("--code", type=str, default=None, help="下载指定指数代码")
    parser.add_argument("--output", type=str, default=None,
                        help="输出根目录（默认 data-store/parquet/aks/）")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(PROJECT_DIR, "data-store", "parquet", "aks")

    if args.check:
        check()
    elif args.list:
        list_indices()
    elif args.code:
        matched = [i for i in LEGU_INDEXES if i["code"] == args.code]
        if not matched:
            print(f"[ERROR] 未知指数代码: {args.code}", file=sys.stderr)
            print("使用 --list 查看可用指数代码", file=sys.stderr)
            sys.exit(1)
        idx = matched[0]
        pe_dir = os.path.join(output_dir, "pe")
        pb_dir = os.path.join(output_dir, "pb")
        os.makedirs(pe_dir, exist_ok=True)
        os.makedirs(pb_dir, exist_ok=True)
        print(f"下载 {idx['code']} {idx['name']} ...")
        df_pe = fetch_index_pe(idx["legu_code"])
        _save_and_report(df_pe, os.path.join(pe_dir, f"{idx['code']}.parquet"),
                         idx["name"], "pe_ew", "PE")
        df_pb = fetch_index_pb(idx["legu_code"])
        _save_and_report(df_pb, os.path.join(pb_dir, f"{idx['code']}.parquet"),
                         idx["name"], "pb_ew", "PB")
    else:
        download_all(output_dir)


if __name__ == "__main__":
    main()
