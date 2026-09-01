#!/usr/bin/env python3
"""
策略B + 本金阈值收缩 独立评估 (训练集 000300/000905)。

方案:
  策略B (PB主 / FED≤55%闸 / 卖PE / 无卖闸, B10/15/30/70, S85/95)
  base = 1000, 费用 万5 + 低消5元 (双向, 指数口径无整手)
  本金管理: threshold=20万 软收缩 + cap=30万 硬封顶
    < 20万      : 0.5x/1x/2x/3x (全档)
    20万 ~ 24万 : 1x/2x/3x     (去 0.5x)
    24万 ~ 28万 : 2x/3x        (去 1x)
    > 28万      : 仅 3x
    达到 30万    : 停止买入 (硬封顶)
  固定30万口径: 从回测区间第一天(pe_pct 首次有效)固化 30万 (闲置部分无息)

输出 (独立, 不覆盖旧文件):
  wind_new_search/output/strategy_b_threshold.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import run_backtest, prep_df
from wind_new_search.test_windowed import OPTIMAL_PARAMS

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
OUTPUT_DIR = PROJECT_DIR / "wind_new_search" / "output"

CODES = ["000300", "000905"]
NAMES = {"000300": "沪深300", "000905": "中证500"}

BASE = 1000
COMMISSION_RATE = 0.0005   # 万5
MIN_COMMISSION = 5.0       # 低消 5 元
THRESHOLD = 200_000        # 阈值 20万 (软收缩起点)
CAP = 300_000              # 本金封顶 30万
POOL = 300_000             # 固定本金池


def backtest(code, threshold, cap):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = prep_df(df)
    t = df[df["pe_pct"].notna()]
    r = run_backtest(df, OPTIMAL_PARAMS, base_amount=BASE,
                     commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                     lot_size=0, principal_threshold=threshold, principal_cap=cap, principal_pool=POOL)
    return {
        "code": code, "name": NAMES[code],
        "start_date": str(t.date.min().date()),
        "end_date": str(t.date.max().date()),
        "xirr": r["xirr"], "final_return": r["final_return"],
        "total_invested": r["total_invested"], "total_cash_in": r["total_cash_in"],
        "position_value": r["position_value"], "final_value": r["final_value"],
        "principal_final": r["principal_final"], "principal_annual": r["principal_annual"],
        "principal_return": r["principal_return"],
        "trades": r["trades"], "buys": r["buys"], "sells": r["sells"],
    }


def main():
    print("策略B + 20万阈值收缩 + 30万封顶 (训练集 000300/000905)")
    results = []
    baseline = []
    for code in CODES:
        r = backtest(code, THRESHOLD, CAP)
        b = backtest(code, None, CAP)
        results.append(r)
        baseline.append(b)
        print(f"  {r['name']}: XIRR={r['xirr']*100:6.2f}% 固定年化={r['principal_annual']*100:5.2f}% "
              f"投入={r['total_invested']:,.0f} 终值={r['principal_final']:,.0f} 买{r['buys']}/卖{r['sells']}")
        print(f"      基线(纯封顶): XIRR={b['xirr']*100:6.2f}% 固定年化={b['principal_annual']*100:5.2f}% "
              f"投入={b['total_invested']:,.0f} 终值={b['principal_final']:,.0f} 买{b['buys']}/卖{b['sells']}")

    out = {
        "title": "策略B + 本金阈值收缩",
        "params": OPTIMAL_PARAMS,
        "base_amount": BASE,
        "cost_model": {
            "commission_rate": COMMISSION_RATE,
            "min_commission": MIN_COMMISSION,
            "lot_size": 0,
            "note": "指数口径, 无整手限制; 佣金万5 低消5元, 双向",
        },
        "principal_threshold": THRESHOLD,
        "principal_cap": CAP,
        "principal_pool": POOL,
        "note": "固定30万口径: 从回测区间第一天(pe_pct首次有效)固化30万, 闲置部分无息",
        "threshold_tiers": [
            {"up_to": 200000, "min_mult": 0.5, "desc": "0.5x/1x/2x/3x"},
            {"up_to": 240000, "min_mult": 1.0, "desc": "1x/2x/3x"},
            {"up_to": 280000, "min_mult": 2.0, "desc": "2x/3x"},
            {"up_to": 300000, "min_mult": 3.0, "desc": "仅3x"},
            {"up_to": None, "min_mult": None, "desc": "达30万停止买入"},
        ],
        "results": results,
        "baseline": {
            "label": "纯硬封顶30万(无收缩)",
            "principal_threshold": None,
            "principal_cap": CAP,
            "results": baseline,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "strategy_b_threshold.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'strategy_b_threshold.json'}")


if __name__ == "__main__":
    main()
