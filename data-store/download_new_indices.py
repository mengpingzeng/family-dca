#!/usr/bin/env python3
"""
下载 5 个新指数 PE/PB/价格数据 (via akshare)
输出到 data-store/parquet/aks/ 和 data-store/parquet/index_price_aks/
"""
import os, sys, time
sys.path.insert(0, '/usr/local/python3.11/lib/python3.11/site-packages')

import akshare as ak
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PE_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "aks", "pe")
PB_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "aks", "pb")
PRICE_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "index_price_aks")

for d in [PE_DIR, PB_DIR, PRICE_DIR]:
    os.makedirs(d, exist_ok=True)

NEW_INDICES = [
    {"code": "000852", "name": "中证1000", "price_code": "sh000852",  "pe_col": "pe_med"},
    {"code": "000015", "name": "上证红利", "price_code": "sh000015",  "pe_col": "pe_wgt"},
    {"code": "399324", "name": "深证红利", "price_code": "sz399324",  "pe_col": "pe_wgt"},
    {"code": "399673", "name": "创业板50", "price_code": "sz399673",  "pe_col": "pe_med"},
    {"code": "000906", "name": "中证800",  "price_code": "sh000906",  "pe_col": "pe_wgt"},
]

def fetch_with_retry(fn, name, max_retry=8, sleep=3, **kw):
    for attempt in range(1, max_retry + 1):
        try:
            df = fn(**kw)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            if attempt == max_retry:
                raise
            print(f"    retry {attempt}/{max_retry}: {type(e).__name__}", flush=True)
            time.sleep(sleep)
    raise RuntimeError(f"{name}: all retries exhausted")

def download_pe(name):
    df = fetch_with_retry(ak.stock_index_pe_lg, f"PE-{name}", symbol=name)
    df = df.rename(columns={
        "日期": "date",
        "滚动市盈率": "pe_wgt",
        "等权滚动市盈率": "pe_ew",
        "滚动市盈率中位数": "pe_med",
        "静态市盈率": "pe_wgt_lyr",
        "等权静态市盈率": "pe_ew_lyr",
        "静态市盈率中位数": "pe_med_lyr",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    cols = ["date", "pe_ew", "pe_wgt", "pe_med", "pe_ew_lyr", "pe_wgt_lyr", "pe_med_lyr"]
    df = df[[c for c in cols if c in df.columns]].sort_values("date").reset_index(drop=True)
    return df

def download_pb(name):
    df = fetch_with_retry(ak.stock_index_pb_lg, f"PB-{name}", symbol=name)
    df = df.rename(columns={
        "日期": "date",
        "市净率": "pb_wgt",
        "等权市净率": "pb_ew",
        "市净率中位数": "pb_med",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    cols = ["date", "pb_ew", "pb_wgt", "pb_med"]
    df = df[[c for c in cols if c in df.columns]].sort_values("date").reset_index(drop=True)
    return df

def download_price(price_code):
    df = fetch_with_retry(ak.stock_zh_index_daily, f"Price-{price_code}", symbol=price_code)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    out = pd.DataFrame()
    out["date"] = df["date"]
    out["index_open"] = df.get("open", None)
    out["index_price"] = df.get("close", None)
    out = out.sort_values("date").reset_index(drop=True)
    return out

def main():
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"5 新指数数据下载 — {now}\n")

    for idx in NEW_INDICES:
        code = idx["code"]
        name = idx["name"]
        pc = idx["price_code"]
        print(f"[{code}] {name}  (price={pc})")

        try:
            df_pe = download_pe(name)
            path = os.path.join(PE_DIR, f"{code}.parquet")
            df_pe.to_parquet(path, index=False)
            print(f"  PE  OK | {len(df_pe):>5d}条 | {df_pe['date'].min()} ~ {df_pe['date'].max()}")
        except Exception as e:
            print(f"  PE  FAIL | {e}")
            continue

        try:
            df_pb = download_pb(name)
            path = os.path.join(PB_DIR, f"{code}.parquet")
            df_pb.to_parquet(path, index=False)
            print(f"  PB  OK | {len(df_pb):>5d}条 | {df_pb['date'].min()} ~ {df_pb['date'].max()}")
        except Exception as e:
            print(f"  PB  FAIL | {e}")

        try:
            df_price = download_price(pc)
            path = os.path.join(PRICE_DIR, f"{code}.parquet")
            df_price.to_parquet(path, index=False)
            print(f"  PRICE OK | {len(df_price):>5d}条 | {df_price['date'].min()} ~ {df_price['date'].max()}")
        except Exception as e:
            print(f"  PRICE FAIL | {e}")

        time.sleep(1)

    print("\n下载完成")

if __name__ == "__main__":
    main()
