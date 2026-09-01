#!/usr/bin/env python3
"""
下载港股/美股指数价格（akshare 新浪源），供 wind 数据验证使用

输出到 data-store/parquet/index_price_aks/，格式: date, index_open, index_price

标的:
  恒生指数 HSI     → stock_hk_index_daily_sina(symbol='HSI')
  恒生科技 HSTECH  → stock_hk_index_daily_sina(symbol='HSTECH')
  纳斯达克100 NDX100 → index_us_stock_sina(symbol='.NDX')
  标普500 SPX500   → index_us_stock_sina(symbol='.INX')
"""
import os, sys, time
sys.path.insert(0, '/usr/local/python3.11/lib/python3.11/site-packages')

import akshare as ak
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "index_price_aks")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = [
    {"code": "HSI", "name": "恒生指数", "api": "hk", "symbol": "HSI"},
    {"code": "HSTECH", "name": "恒生科技", "api": "hk", "symbol": "HSTECH"},
    {"code": "NDX100", "name": "纳斯达克100", "api": "us", "symbol": ".NDX"},
    {"code": "SPX500", "name": "标普500", "api": "us", "symbol": ".INX"},
]


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


def main():
    for t in TARGETS:
        print(f"[{t['code']}] {t['name']} ({t['symbol']}) ...", end=" ", flush=True)
        try:
            if t["api"] == "hk":
                df = fetch_with_retry(ak.stock_hk_index_daily_sina, t["name"], symbol=t["symbol"])
            else:
                df = fetch_with_retry(ak.index_us_stock_sina, t["name"], symbol=t["symbol"])

            df["date"] = pd.to_datetime(df["date"]).dt.date
            out = pd.DataFrame()
            out["date"] = df["date"]
            out["index_open"] = df.get("open", None)
            out["index_price"] = df.get("close", None)
            out = out.dropna(subset=["index_price"]).sort_values("date").reset_index(drop=True)

            path = os.path.join(OUT_DIR, f"{t['code']}.parquet")
            out.to_parquet(path, index=False)
            print(f"OK | {len(out)}条 | {out['date'].min()} ~ {out['date'].max()}")
        except Exception as e:
            print(f"FAIL | {type(e).__name__}: {str(e)[:80]}")
        time.sleep(1)

    print("\n下载完成")


if __name__ == "__main__":
    main()
