#!/usr/bin/env python3
"""
指数曲线「均衡」细网格搜索.

目标(相对阶梯均衡基准 8/4/2/0 mid=0.25: 300=3.67%/13.35%, 500=5.45%/6.92%):
  1. 500 固定年化尽量冲高(目标 6%)
  2. 300 固定年化不削弱(>=3.6%)
  3. XIRR 不削弱(300>=13%, 500>=6.9%)

搜索: A x k x pct_max 细网格, 找满足硬约束的组合 + 帕累托前沿.
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

PARAMS = {
    "buy_signal": "PB", "buy_gate": "FED", "buy_gate_cap": 0.55,
    "sell_signal": "PE", "sell_gate": None, "sell_gate_floor": None,
    "buy_floor": 0.10, "buy_low": 0.15, "buy_mid": 0.25, "buy_high": 0.70,
    "sell_heavy": 0.85, "sell_extreme": 0.95,
}

A_GRID = [12, 14, 16, 18, 20, 22, 24, 28, 32]
K_GRID = [8, 10, 12, 14, 16, 18, 20, 22, 24]
PCT_MAX_GRID = [0.18, 0.20, 0.22, 0.25, 0.28]

BASE = 1000
CR, MC = 0.0005, 5.0
TH, CAP, POOL = 200_000, 300_000, 300_000


def bt(code, buy_curve):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = prep_df(df)
    return run_backtest(df, PARAMS, base_amount=BASE, commission_rate=CR, min_commission=MC,
                        lot_size=0, principal_threshold=TH, principal_cap=CAP, principal_pool=POOL,
                        buy_curve=buy_curve)


def main():
    rows = []
    for A in A_GRID:
        for k in K_GRID:
            for pm in PCT_MAX_GRID:
                r300 = bt("000300", {"A": A, "k": k, "pct_max": pm})
                r500 = bt("000905", {"A": A, "k": k, "pct_max": pm})
                rows.append({
                    "A": A, "k": k, "pct_max": pm,
                    "a300": r300["principal_annual"], "x300": r300["xirr"],
                    "a500": r500["principal_annual"], "x500": r500["xirr"],
                })

    # 帕累托前沿: (300年化, 500年化) 非支配点
    pareto = []
    for r in rows:
        dominated = any(o["a300"] >= r["a300"] and o["a500"] >= r["a500"]
                        and (o["a300"] > r["a300"] or o["a500"] > r["a500"]) for o in rows)
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda r: r["a300"])

    print("=== 帕累托前沿 (300年化 vs 500年化, 指数曲线) ===")
    print(f"{'A':>4} {'k':>3} {'pct_max':>7} | {'300年化':>7} {'300XIRR':>8} | {'500年化':>7} {'500XIRR':>8}")
    for r in pareto:
        print(f"{r['A']:>4} {r['k']:>3} {r['pct_max']:>7} | {r['a300']*100:6.2f}% {r['x300']*100:7.2f}% | {r['a500']*100:6.2f}% {r['x500']*100:7.2f}%")

    # 满足 XIRR 约束 + 300不削弱的组合
    print("\n=== 满足 XIRR 约束(300>=13% 且 500>=6.9%) 且 300年化>=3.6% 的组合 (按500年化排序) ===")
    ok = [r for r in rows if r["x300"] >= 0.13 and r["x500"] >= 0.069 and r["a300"] >= 0.036]
    ok.sort(key=lambda r: -r["a500"])
    if not ok:
        print("  (无满足全部约束的组合)")
        print("\n=== 放宽: 300年化>=3.4% 且 XIRR 不削弱 ===")
        ok = [r for r in rows if r["x300"] >= 0.13 and r["x500"] >= 0.069 and r["a300"] >= 0.034]
        ok.sort(key=lambda r: -r["a500"])
    print(f"{'A':>4} {'k':>3} {'pct_max':>7} | {'300年化':>7} {'300XIRR':>8} | {'500年化':>7} {'500XIRR':>8}")
    for r in ok[:15]:
        print(f"{r['A']:>4} {r['k']:>3} {r['pct_max']:>7} | {r['a300']*100:6.2f}% {r['x300']*100:7.2f}% | {r['a500']*100:6.2f}% {r['x500']*100:7.2f}%")

    # 最优均衡: 在 XIRR 不削弱前提下最大化 min(300,500)年化
    print("\n=== XIRR 不削弱前提下, 最大化 min(300年化,500年化) ===")
    ok2 = [r for r in rows if r["x300"] >= 0.13 and r["x500"] >= 0.069]
    ok2.sort(key=lambda r: -min(r["a300"], r["a500"]))
    for r in ok2[:5]:
        print(f"  A={r['A']} k={r['k']} pct_max={r['pct_max']}: 300={r['a300']*100:.2f}%/XIRR{r['x300']*100:.2f}%  500={r['a500']*100:.2f}%/XIRR{r['x500']*100:.2f}%  (min {min(r['a300'],r['a500'])*100:.2f}%)")

    with open(OUTPUT_DIR / "exp_curve_balanced.json", "w") as f:
        json.dump({"rows": rows, "pareto": pareto}, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'exp_curve_balanced.json'}")


if __name__ == "__main__":
    main()
