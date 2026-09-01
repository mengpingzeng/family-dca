#!/usr/bin/env python3
"""
base 调大对比实验 (策略不变, 只调大每次买入基准金额).

宗旨: 低位多买(3x/2x), 高位尽量少买(0.5x)。base 是统一乘数, 调大 base
会把低位和高位档位等比放大, 但「低位倍数高/高位倍数低」的相对结构不变;
效果是更快投满 30万、提高资金利用率, 代价是相对高位也买得更多。

策略 (保持不变):
  策略B        : PB主 / FED<=55%闸 / 卖PE / 无卖闸, B10/15/30/70, S85/95
  固定年化最优 : PB主 / PB<=60%闸 / 卖FED / PB卖闸, B10/20/30/50, S80/95

base 取值: 1000 / 1500 / 2000
固定配置: 阈值收缩 20万 + 封顶 30万 + 费用 万5低消5 + 固定30万口径(回测区间第一天固化)

输出 (独立, 不覆盖旧文件):
  wind_new_search/output/base_scaling.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import run_backtest, prep_df

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
OUTPUT_DIR = PROJECT_DIR / "wind_new_search" / "output"

CODES = ["000300", "000905"]
NAMES = {"000300": "沪深300", "000905": "中证500"}

STRATEGIES = {
    "策略B": {
        "buy_signal": "PB", "buy_gate": "FED", "buy_gate_cap": 0.55,
        "sell_signal": "PE", "sell_gate": None, "sell_gate_floor": None,
        "buy_floor": 0.10, "buy_low": 0.15, "buy_mid": 0.30, "buy_high": 0.70,
        "sell_heavy": 0.85, "sell_extreme": 0.95,
    },
    "固定年化最优": {
        "buy_signal": "PB", "buy_gate": "PB", "buy_gate_cap": 0.6,
        "sell_signal": "FED", "sell_gate": "PB", "sell_gate_floor": 0.7,
        "buy_floor": 0.1, "buy_low": 0.2, "buy_mid": 0.3, "buy_high": 0.5,
        "sell_heavy": 0.8, "sell_extreme": 0.95,
    },
}

BASES = [1000, 1500, 2000]
COMMISSION_RATE = 0.0005   # 万5
MIN_COMMISSION = 5.0       # 低消 5 元
THRESHOLD = 200_000        # 阈值 20万 (软收缩起点)
CAP = 300_000              # 本金封顶 30万
POOL = 300_000             # 固定本金池


def backtest(code, params, base):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = prep_df(df)
    t = df[df["pe_pct"].notna()]
    r = run_backtest(df, params, base_amount=base,
                     commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                     lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP, principal_pool=POOL)
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
    print("base 调大对比 (策略不变, 阈值20万收缩 + 封顶30万)")
    combos = []
    for sname, params in STRATEGIES.items():
        for base in BASES:
            results = [backtest(code, params, base) for code in CODES]
            combos.append({"strategy": sname, "base": base, "results": results})

    # 对比表
    hdr = f"{'策略':12} {'base':>6} | {'300固定年化':>9} {'300 XIRR':>8} {'300终值':>10} {'300买/卖':>8} | {'500固定年化':>9} {'500 XIRR':>8} {'500终值':>10} {'500买/卖':>8}"
    print(hdr)
    print("-" * len(hdr))
    for c in combos:
        r300 = c["results"][0]
        r500 = c["results"][1]
        print(f"{c['strategy']:12} {c['base']:>6} | "
              f"{r300['principal_annual']*100:8.2f}% {r300['xirr']*100:7.2f}% {r300['principal_final']:>10,.0f} {r300['buys']:>3}/{r300['sells']:<3} | "
              f"{r500['principal_annual']*100:8.2f}% {r500['xirr']*100:7.2f}% {r500['principal_final']:>10,.0f} {r500['buys']:>3}/{r500['sells']:<3}")

    out = {
        "title": "base 调大对比 (策略不变)",
        "note": "固定30万口径: 从回测区间第一天(pe_pct首次有效)固化30万, 闲置部分无息",
        "cost_model": {"commission_rate": COMMISSION_RATE, "min_commission": MIN_COMMISSION, "lot_size": 0},
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "strategies": {k: v for k, v in STRATEGIES.items()},
        "bases": BASES,
        "combos": combos,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "base_scaling.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'base_scaling.json'}")


if __name__ == "__main__":
    main()
