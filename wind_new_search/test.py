#!/usr/bin/env python3
"""
测试集应用 — 用训练集最优策略在 12 个测试指数上回测。

用法: python wind_new_search/test.py
输出: wind_new_search/output/test_results.json
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import run_backtest, prep_df

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
OUTPUT_DIR = PROJECT_DIR / "wind_new_search" / "output"

TEST_CODES = ["000015", "000016", "000852", "399006", "399330",
              "HSI", "NDX100", "SPX500", "930931", "930930", "000688", "HSTECH"]
TEST_NAMES = {
    "000015": "上证红利", "000016": "上证50", "000852": "中证1000",
    "399006": "创业板指", "399330": "深证100", "HSI": "恒生指数",
    "NDX100": "纳斯达克100", "SPX500": "标普500", "930931": "港股通50",
    "930930": "港股综合", "000688": "科创50", "HSTECH": "恒生科技",
}


def main():
    train = json.load(open(OUTPUT_DIR / "train_results.json"))
    best = train["top"][0]
    params = {k: best[k] for k in ["buy_signal", "buy_gate", "buy_gate_cap", "sell_signal",
                                   "buy_floor", "buy_low", "buy_mid", "buy_high",
                                   "sell_heavy", "sell_extreme"]}
    gate = f"{params['buy_gate']}≤{params['buy_gate_cap']:.0%}" if params["buy_gate"] else "无"
    print(f"最优策略: {params['buy_signal']}主/{gate}闸/卖{params['sell_signal']} "
          f"B{params['buy_floor']:.0%}/{params['buy_low']:.0%}/{params['buy_mid']:.0%}/{params['buy_high']:.0%} "
          f"S{params['sell_heavy']:.0%}/{params['sell_extreme']:.0%}")
    print(f"训练集 min XIRR = {best['unified_xirr']*100:.2f}%\n")

    results = []
    for code in TEST_CODES:
        p = MERGED_DIR / f"{code}.parquet"
        if not p.exists():
            print(f"  [SKIP] {code} 无数据")
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        window = int(df["window"].iloc[0])
        df = prep_df(df)
        r = run_backtest(df, params)
        t = df[df["pe_pct"].notna()]
        results.append({
            "code": code, "name": TEST_NAMES.get(code, code), "window": window,
            "xirr": r["xirr"], "final_return": r["final_return"],
            "total_invested": r["total_invested"], "final_value": r["final_value"],
            "trades": r["trades"], "buys": r["buys"], "sells": r["sells"],
            "trade_start": str(t["date"].iloc[0].date()) if len(t) else None,
            "trade_end": str(t["date"].iloc[-1].date()) if len(t) else None,
        })
        print(f"  {code:8s} {TEST_NAMES.get(code, code):8s} 窗口={window}yr "
              f"XIRR={r['xirr']*100:6.2f}% 回报={r['final_return']*100:6.2f}% "
              f"交易={r['trades']:3d} (买{r['buys']}/卖{r['sells']}) "
              f"可交易 {len(t)}行")

    out = {"params": params, "train_min_xirr": best["unified_xirr"], "results": results}
    with open(OUTPUT_DIR / "test_results.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'test_results.json'}")


if __name__ == "__main__":
    main()
