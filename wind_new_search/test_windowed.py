#!/usr/bin/env python3
"""
窗口分层回测 — 每个指数用它能满足的最长窗口, 只做这一档回测。

规则:
  - 10yr: PE/PB 起点 ≤ 2005-03, 交易起点固定 2015-03-06
  - 5yr : 各自 PE/PB 起点 + 5 年
  - 3yr : 各自 PE/PB 起点 + 3 年

策略: 训练集最优 (B: PB主/FED≤55%闸/卖PE B10/15/30/70 S85/95)
用法: python wind_new_search/test_windowed.py
输出: wind_new_search/output/test_windowed.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import run_backtest, prep_df

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
OUTPUT_DIR = PROJECT_DIR / "wind_new_search" / "output"

FIXED_START = "2015-03-06"  # 10yr 档统一交易起点

# code -> (window, start_mode)  start_mode: 'fixed' 或 'own'
# 10yr 档: 固定 2015-03-06 起, 起步时用已有历史(封顶10年)
ASSIGNMENT = {
    "000300": {"window": 10, "start": "fixed"},
    "000905": {"window": 10, "start": "fixed"},
    "000015": {"window": 10, "start": "fixed"},
    "000016": {"window": 10, "start": "fixed"},
    "399330": {"window": 10, "start": "fixed"},
    "HSI":    {"window": 10, "start": "fixed"},
    "SPX500": {"window": 10, "start": "fixed"},
    "399006": {"window": 5, "start": "own"},
    "NDX100": {"window": 5, "start": "own"},
    "000852": {"window": 3, "start": "own"},
    "930931": {"window": 3, "start": "own"},
    "930930": {"window": 3, "start": "own"},
    "000688": {"window": 3, "start": "own"},
    "HSTECH": {"window": 3, "start": "own"},
}

NAMES = {
    "000300": "沪深300", "000905": "中证500", "000015": "上证红利", "000016": "上证50",
    "399330": "深证100", "HSI": "恒生指数", "SPX500": "标普500", "399006": "创业板指",
    "NDX100": "纳斯达克100", "000852": "中证1000", "930931": "港股通50", "930930": "港股综合",
    "000688": "科创50", "HSTECH": "恒生科技",
}

OPTIMAL_PARAMS = {
    "buy_signal": "PB", "buy_gate": "FED", "buy_gate_cap": 0.55,
    "sell_signal": "PE", "sell_gate": None, "sell_gate_floor": None,
    "buy_floor": 0.10, "buy_low": 0.15, "buy_mid": 0.30, "buy_high": 0.70,
    "sell_heavy": 0.85, "sell_extreme": 0.95,
}


def rolling_pct(series, window_rows, min_samples):
    n = len(series)
    result = np.full(n, np.nan)
    clean = ~np.isnan(series)
    for i in range(n):
        if not clean[i]:
            continue
        start = max(0, i - window_rows)
        wc = clean[start:i + 1]
        if wc.sum() < min_samples:
            continue
        w = series[start:i + 1][wc]
        result[i] = (w <= series[i]).sum() / len(w)
    return result


def build_windowed_df(code, w=None, start_mode=None, start_date=None, uniform10yr=False):
    """加载原始数据, 按指定窗口重算百分位并过滤到交易起点, 返回 (df, cfg).

    start_date: 可选全局交易起点 (如 '2007-10-16'), 传入时覆盖 fixed/own 的默认起点;
                未传入时行为与旧版完全一致。

    uniform10yr: 为 True 时对所有指数统一用「最多10年历史、不足则有多少用多少」规则
                 (窗口封顶10年, 最少20样本即开始交易), 忽略各指数原 10/5/3 窗口分配。
    """
    cfg = dict(ASSIGNMENT[code])
    if uniform10yr:
        w = 10
        start_mode = "fixed"
        cfg["window"] = 10
        cfg["start"] = "fixed"
    else:
        w = w or cfg["window"]
        start_mode = start_mode or cfg["start"]
    p = MERGED_DIR / f"{code}.parquet"
    if not p.exists():
        return None, cfg
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    pe_valid = df[df["pe"].notna()]
    total_years = (pe_valid["date"].max() - pe_valid["date"].min()).days / 365.25
    rpy = len(pe_valid) / max(total_years, 1)
    wr = int(w * rpy)
    if start_mode == "fixed":
        # 固定起点: 起步时用已有历史(封顶窗口), 不需要满窗口
        min_samples = 20
    else:
        min_samples = max(20, wr)
    df["pe_pct"] = rolling_pct(df["pe"].values.astype(float), wr, min_samples)
    df["pb_pct"] = rolling_pct(df["pb"].values.astype(float), wr, min_samples)
    df["fed_pct"] = rolling_pct(df["fed"].values.astype(float), wr, min_samples)

    pe_start = pe_valid["date"].min()
    if start_date is not None:
        start_date = pd.Timestamp(start_date)
    elif start_mode == "fixed":
        start_date = pd.Timestamp(FIXED_START)
    else:
        start_date = pe_start + pd.DateOffset(years=w)

    df = df[df["date"] >= start_date].reset_index(drop=True)
    return df, cfg


def main():
    print(f"窗口分层回测 — 策略 B: PB主/FED≤55%闸/卖PE B10/15/30/70 S85/95\n")
    results = []
    for code in ASSIGNMENT:
        df, cfg = build_windowed_df(code)
        if df is None:
            print(f"  [SKIP] {code} 无数据")
            continue
        w = cfg["window"]
        bt_df = prep_df(df)
        r = run_backtest(bt_df, OPTIMAL_PARAMS)

        tradable = bt_df[bt_df["pe_pct"].notna()]
        first_trade = str(tradable["date"].iloc[0].date()) if len(tradable) else "N/A"
        start_date = bt_df["date"].min()
        results.append({
            "code": code, "name": NAMES[code], "window": w,
            "start_date": str(start_date.date()),
            "xirr": r["xirr"], "final_return": r["final_return"],
            "total_invested": r["total_invested"], "final_value": r["final_value"],
            "trades": r["trades"], "buys": r["buys"], "sells": r["sells"],
            "first_trade": first_trade,
            "trade_end": str(tradable["date"].iloc[-1].date()) if len(tradable) else None,
        })
        print(f"  {code:8s} {NAMES[code]:8s} {w}yr 起点{start_date.date()} "
              f"XIRR={r['xirr']*100:6.2f}% 回报={r['final_return']*100:6.2f}% "
              f"交易={r['trades']:3d} (买{r['buys']}/卖{r['sells']})")

    out = {"params": OPTIMAL_PARAMS, "results": results}
    with open(OUTPUT_DIR / "test_windowed.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'test_windowed.json'}")


if __name__ == "__main__":
    main()
