#!/usr/bin/env python3
"""
方向1: 指数曲线 + 中位地板买入 实验.

mult(pct) = floor + A * (e^-kpct - e^-kpct_max)/(1 - e^-kpct_max),  pct<pct_max; 否则 0.
地板让中位区间(接近 pct_max)保留最小买入力度, 试图同时服务 500(低位重仓) 与 300(中位持续买).

目标约束(相对阶梯均衡基准 300=3.67%/13.35%, 500=5.45%/6.92%):
  500 年化冲高 + 300 年化不削(>=3.6%) + XIRR 不削(300>=13%, 500>=6.9%)
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

A_GRID = [12, 14, 16]
K_GRID = [20, 24]
PCT_MAX_GRID = [0.18, 0.20, 0.25]
FLOOR_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]

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
                for fl in FLOOR_GRID:
                    r300 = bt("000300", {"A": A, "k": k, "pct_max": pm, "floor": fl})
                    r500 = bt("000905", {"A": A, "k": k, "pct_max": pm, "floor": fl})
                    rows.append({
                        "A": A, "k": k, "pct_max": pm, "floor": fl,
                        "a300": r300["principal_annual"], "x300": r300["xirr"],
                        "a500": r500["principal_annual"], "x500": r500["xirr"],
                    })

    # 满足全部约束
    print("=== 满足 300年化>=3.6% 且 500年化>=5.4% 且 XIRR(300>=13%,500>=6.9%) ===")
    ok = [r for r in rows if r["a300"] >= 0.036 and r["a500"] >= 0.054 and r["x300"] >= 0.13 and r["x500"] >= 0.069]
    ok.sort(key=lambda r: -r["a500"])
    if ok:
        for r in ok[:20]:
            print(f"  A={r['A']:>2} k={r['k']:>2} max={r['pct_max']} floor={r['floor']}: "
                  f"300={r['a300']*100:.2f}%/XIRR{r['x300']*100:.2f}%  500={r['a500']*100:.2f}%/XIRR{r['x500']*100:.2f}%")
    else:
        print("  (无满足全部约束的组合)")

    # 放宽: 300>=3.5% 且 500>=5.4% 且 XIRR 不削
    print("\n=== 放宽: 300年化>=3.5% 且 500>=5.4% 且 XIRR 不削 (按500排序) ===")
    ok2 = [r for r in rows if r["a300"] >= 0.035 and r["a500"] >= 0.054 and r["x300"] >= 0.13 and r["x500"] >= 0.069]
    ok2.sort(key=lambda r: -r["a500"])
    if ok2:
        for r in ok2[:15]:
            print(f"  A={r['A']:>2} k={r['k']:>2} max={r['pct_max']} floor={r['floor']}: "
                  f"300={r['a300']*100:.2f}%/XIRR{r['x300']*100:.2f}%  500={r['a500']*100:.2f}%/XIRR{r['x500']*100:.2f}%")
    else:
        print("  (无)")

    # XIRR 不削前提下最大化 min(300,500) 年化
    print("\n=== XIRR 不削前提下最大化 min(300年化,500年化) top10 ===")
    ok3 = [r for r in rows if r["x300"] >= 0.13 and r["x500"] >= 0.069]
    ok3.sort(key=lambda r: -min(r["a300"], r["a500"]))
    for r in ok3[:10]:
        print(f"  A={r['A']:>2} k={r['k']:>2} max={r['pct_max']} floor={r['floor']}: "
              f"300={r['a300']*100:.2f}%/XIRR{r['x300']*100:.2f}%  500={r['a500']*100:.2f}%/XIRR{r['x500']*100:.2f}%  (min {min(r['a300'],r['a500'])*100:.2f}%)")

    with open(OUTPUT_DIR / "exp_curve_floor.json", "w") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'exp_curve_floor.json'}")


if __name__ == "__main__":
    main()
