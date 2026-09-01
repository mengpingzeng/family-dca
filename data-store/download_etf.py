#!/usr/bin/env python3
"""
下载 12 只 ETF 后复权日频价格。

后复权价 = 不复权市价 × (累计净值 / 单位净值)

  - 不复权市价: fund_etf_hist_sina  (close)
  - 累计净值/单位净值: fund_etf_fund_info_em  (东方财富, 连续口径, 同时处理分红与份额折算)

该复权因子同时消除「分红除权」和「份额折算」造成的价格跳空, 得到连续的总回报市价序列
(市价口径, 保留溢价/折价)。已验证与东方财富 fund_etf_hist_em(adjust='hfq') 仅差一个常数
基数因子, 对 XIRR 无影响。

用法: python data-store/download_etf.py
输出: data-store/parquet/etf/{code}.parquet  (列: date, close 不复权, hfq 后复权)
"""
import time
from pathlib import Path

import akshare as ak
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "data-store" / "parquet" / "etf"

# 指数 code -> (ETF代码, ETF名称, 市场 sh/sz)
ETF_MAP = {
    "000300": ("510300", "华泰柏瑞沪深300ETF", "sh"),
    "000905": ("510500", "南方中证500ETF", "sh"),
    "000015": ("510880", "华泰柏瑞红利ETF", "sh"),
    "000016": ("510050", "华夏上证50ETF", "sh"),
    "399330": ("159901", "易方达深证100ETF", "sz"),
    "399006": ("159915", "易方达创业板ETF", "sz"),
    "000688": ("588000", "华夏科创50ETF", "sh"),
    "000852": ("512100", "南方中证1000ETF", "sh"),
    "HSI":    ("159920", "华夏恒生ETF", "sz"),
    "HSTECH": ("513180", "华夏恒生科技ETF", "sh"),
    "SPX500": ("513500", "博时标普500ETF", "sh"),
    "NDX100": ("513100", "国泰纳指100ETF", "sh"),
}


def fetch_one(etf_code, mkt):
    sym = f"{mkt}{etf_code}"

    # 不复权市价 (新浪)
    price = ak.fund_etf_hist_sina(symbol=sym)
    price = price.rename(columns={"date": "date", "close": "close"})
    price["date"] = pd.to_datetime(price["date"]).astype("datetime64[ns]")
    price = price[["date", "close"]].sort_values("date").reset_index(drop=True)

    # 净值 (东方财富): 累计净值 + 单位净值
    nav = ak.fund_etf_fund_info_em(fund=etf_code, start_date="20000101", end_date="20261231")
    nav = nav.rename(columns={"净值日期": "date", "单位净值": "nav", "累计净值": "cum_nav"})
    nav["date"] = pd.to_datetime(nav["date"]).astype("datetime64[ns]")
    nav = nav[["date", "nav", "cum_nav"]].sort_values("date")

    # 对齐净值到市价日期, 计算复权因子
    merged = pd.merge_asof(price, nav, on="date", direction="backward")
    merged["factor"] = merged["cum_nav"] / merged["nav"]
    merged["hfq"] = merged["close"] * merged["factor"]
    return merged[["date", "close", "hfq"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx_code, (etf_code, etf_name, mkt) in ETF_MAP.items():
        try:
            df = fetch_one(etf_code, mkt)
            df.to_parquet(OUT_DIR / f"{etf_code}.parquet", index=False)
            print(f"[OK] {idx_code} -> {etf_code} {etf_name}: "
                  f"{df['date'].min().date()} ~ {df['date'].max().date()} ({len(df)} 行)")
        except Exception as e:
            print(f"[失败] {idx_code} {etf_code} {etf_name}: {type(e).__name__}: {e}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
