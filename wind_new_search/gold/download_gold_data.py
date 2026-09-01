#!/usr/bin/env python3
"""
黄金模块数据下载（完全独立于宽基指数数据目录）

数据源:
  - 上海金 Au99.99（akshare 金交所，元/克，日频）→ gold/data/au99.99.parquet
  - 美国 10Y TIPS 实际收益率（FRED DFII10，日频）→ gold/data/tips_10y.parquet
  - 美国 10Y 名义收益率（已有 bond_yield，日频）→ 复用

输出列: date, price / date, tips10y
"""
import os, sys, time
sys.path.insert(0, '/usr/local/python3.11/lib/python3.11/site-packages')

import pandas as pd
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_with_retry(fn, name, max_retry=6, sleep=3, **kw):
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
    raise RuntimeError(f"{name}: 全部重试失败")


def download_au9999():
    import akshare as ak
    df = fetch_with_retry(ak.spot_hist_sge, "Au99.99", symbol="Au99.99")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["close"] > 0]
    out = pd.DataFrame()
    out["date"] = df["date"]
    out["price"] = df["close"]
    out = out.dropna().sort_values("date").reset_index(drop=True)
    path = os.path.join(DATA_DIR, "au99.99.parquet")
    out.to_parquet(path, index=False)
    print(f"Au99.99: {len(out)}条 {out['date'].min()} ~ {out['date'].max()} → {path}")


def download_tips10y():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"
    data = urllib.request.urlopen(url, timeout=30).read().decode()
    lines = [l for l in data.splitlines() if l.strip()]
    rows = []
    for l in lines[1:]:
        d, v = l.split(",")
        if v:
            try:
                rows.append((pd.Timestamp(d).date(), float(v)))
            except Exception:
                pass
    out = pd.DataFrame(rows, columns=["date", "tips10y"]).sort_values("date").reset_index(drop=True)
    path = os.path.join(DATA_DIR, "tips_10y.parquet")
    out.to_parquet(path, index=False)
    print(f"TIPS 10Y: {len(out)}条 {out['date'].min()} ~ {out['date'].max()} → {path}")


def main():
    download_au9999()
    download_tips10y()
    print("\n黄金数据下载完成")


if __name__ == "__main__":
    main()
