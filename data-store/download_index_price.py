#!/usr/bin/env python3
"""指数价格数据下载器 — 从 CSI index-perf 拉取日频收盘价"""

import argparse, os, sys, time
from datetime import datetime
import pandas as pd, requests

API_URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.csindex.com.cn/"}
TIMEOUT = 30; DELAY = 3.0

DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(DIR)
OUT = os.path.join(PROJ, "data-store", "parquet", "index_price")

# 中证制编的指数（CSI API 有价格）
CSI_CODES = [
    "000300","000905","000852","000016","000688","000510",
    "399006","399330","000015","000922","930955","930915",
    "930930","930931","931573","930939",
]

def fetch_price(code):
    r = requests.get(API_URL, params={
        "indexCode": code, "startDate": "20180101", "endDate": "20260806"},
        headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    items = r.json().get("data", [])
    rows = []
    for it in items:
        if it.get("close") is not None:
            rows.append({
                "date": pd.to_datetime(it["tradeDate"], format="%Y%m%d").date(),
                "index_price": float(it["close"]),
                "index_open": float(it["open"]) if it.get("open") is not None else None,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def download_one(code, name):
    print(f"[FETCH] {code} {name} ...", end=" ", flush=True)
    try:
        df = fetch_price(code)
        if df.empty:
            print("无数据"); return False
        out_path = os.path.join(OUT, f"{code}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"OK | {len(df)}条 | {df.date.min()}~{df.date.max()} | price {df.index_price.min():.1f}~{df.index_price.max():.1f} | latest {df.iloc[-1].index_price:.2f}")
        return True
    except Exception as e:
        print(f"ERROR {e}"); return False

NAMES = {"000300":"沪深300","000905":"中证500","000852":"中证1000","000016":"上证50",
    "000688":"科创50","000510":"中证A500","399006":"创业板指","399330":"深证100",
    "000015":"上证红利","000922":"中证红利","930955":"红利低波100","930915":"港股通高股息",
    "930930":"港股综合","930931":"港股通50","931573":"港股通科技","930939":"中证质量成长"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.check:
        df = fetch_price("000300")
        print(f"Check 000300: {len(df)} rows, {df.date.min()}~{df.date.max()}" if not df.empty else "FAIL")
        return
    print(f"\n指数价格下载 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ok = sum(download_one(c, NAMES.get(c,c)) for c in CSI_CODES for _ in [time.sleep(DELAY)])
    print(f"\n结果: {ok}/{len(CSI_CODES)} => {OUT}/")

if __name__ == "__main__":
    main()
