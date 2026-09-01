#!/usr/bin/env python3
"""
指数曲线买入 网格实验.

mult(pct) = A * (e^(-k*pct) - e^(-k*pct_max)) / (1 - e^(-k*pct_max))
  越便宜(昂贵度 pct 越低)买得越多, pct_max 处归零, 连续平滑无跳变.

对比阶梯式均衡策略 (8/4/2/0, mid=0.25).

固定配置: 策略B (PB主/FED<=55%闸/卖PE S85/95), base=1000,
          阈值20万收缩(pct_max 收缩) + 封顶30万, 费用万5低消5.

输出 (独立): wind_new_search/output/exp_curve.json
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

PARAMS = {
    "buy_signal": "PB", "buy_gate": "FED", "buy_gate_cap": 0.55,
    "sell_signal": "PE", "sell_gate": None, "sell_gate_floor": None,
    "buy_floor": 0.10, "buy_low": 0.15, "buy_mid": 0.25, "buy_high": 0.70,
    "sell_heavy": 0.85, "sell_extreme": 0.95,
}

A_GRID = [16, 32, 64, 128]
K_GRID = [10, 14, 18, 24]
PCT_MAX_GRID = [0.20, 0.25, 0.30]

BASE = 1000
COMMISSION_RATE = 0.0005
MIN_COMMISSION = 5.0
THRESHOLD = 200_000
CAP = 300_000
POOL = 300_000


def backtest(code, buy_curve=None, buy_mults=None, params=PARAMS):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = prep_df(df)
    r = run_backtest(df, params, base_amount=BASE,
                     commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                     lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP,
                     principal_pool=POOL, buy_curve=buy_curve, buy_mults=buy_mults)
    return {
        "code": code, "name": NAMES[code],
        "xirr": r["xirr"],
        "principal_final": r["principal_final"], "principal_annual": r["principal_annual"],
        "buys": r["buys"], "sells": r["sells"], "total_invested": r["total_invested"],
    }


def run_pair(buy_curve=None, buy_mults=None, params=PARAMS):
    return [backtest(code, buy_curve, buy_mults, params) for code in CODES]


def main():
    print("指数曲线买入网格 (策略B, base=1000, 阈值20万收缩+封顶30万)")
    # 阶梯基准
    base_pair = run_pair(buy_mults=(8, 4, 2, 0))
    b300, b500 = base_pair
    print(f"阶梯基准 8/4/2/0 mid=0.25: 300={b300['principal_annual']*100:.2f}%/XIRR{b300['xirr']*100:.2f}%  "
          f"500={b500['principal_annual']*100:.2f}%/XIRR{b500['xirr']*100:.2f}%")

    combos = []
    for pct_max in PCT_MAX_GRID:
        print(f"\n--- pct_max={pct_max} ---")
        ak_label = "A\\k"
        print(f"{ak_label:>6} | {'k=10':^22} | {'k=14':^22} | {'k=18':^22} | {'k=24':^22}")
        print(f"{'':6} | " + " | ".join(f"{'300年化/XIRR 500年化/XIRR':^22}" for _ in K_GRID))
        print("-" * 100)
        for A in A_GRID:
            row = []
            cells = []
            for k in K_GRID:
                pair = run_pair(buy_curve={"A": A, "k": k, "pct_max": pct_max})
                r300, r500 = pair
                combos.append({"A": A, "k": k, "pct_max": pct_max, "results": pair})
                cells.append(f"{r300['principal_annual']*100:4.2f}%/{r300['xirr']*100:5.2f}% "
                             f"{r500['principal_annual']*100:4.2f}%/{r500['xirr']*100:5.2f}%")
            print(f"{A:>6} | " + " | ".join(f"{c:^22}" for c in cells))

    # 按 min(300年化,500年化) 排序 top8
    print("\n=== 均衡最优 top8 (min(300年化,500年化) 降序) ===")
    combos_sorted = sorted(combos, key=lambda c: -min(c["results"][0]["principal_annual"], c["results"][1]["principal_annual"]))
    for c in combos_sorted[:8]:
        r300, r500 = c["results"]
        print(f"  A={c['A']:>3} k={c['k']:>2} pct_max={c['pct_max']}: "
              f"300={r300['principal_annual']*100:.2f}%/XIRR{r300['xirr']*100:.2f}%  500={r500['principal_annual']*100:.2f}%/XIRR{r500['xirr']*100:.2f}%  "
              f"(min {min(r300['principal_annual'],r500['principal_annual'])*100:.2f}%)")

    # 500 最优 top3
    print("\n=== 500 固定年化 top3 ===")
    combos_500 = sorted(combos, key=lambda c: -c["results"][1]["principal_annual"])
    for c in combos_500[:3]:
        r300, r500 = c["results"]
        print(f"  A={c['A']:>3} k={c['k']:>2} pct_max={c['pct_max']}: "
              f"300={r300['principal_annual']*100:.2f}%  500={r500['principal_annual']*100:.2f}%/XIRR{r500['xirr']*100:.2f}%")

    # 300 最优 top3
    print("\n=== 300 固定年化 top3 ===")
    combos_300 = sorted(combos, key=lambda c: -c["results"][0]["principal_annual"])
    for c in combos_300[:3]:
        r300, r500 = c["results"]
        print(f"  A={c['A']:>3} k={c['k']:>2} pct_max={c['pct_max']}: "
              f"300={r300['principal_annual']*100:.2f}%/XIRR{r300['xirr']*100:.2f}%  500={r500['principal_annual']*100:.2f}%")

    out = {
        "title": "指数曲线买入网格",
        "note": "固定30万口径; mult=A*(e^-kpct - e^-kpct_max)/(1-e^-kpct_max); 阈值收缩=本金越高pct_max越回收",
        "params": PARAMS, "base_amount": BASE,
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "A_grid": A_GRID, "k_grid": K_GRID, "pct_max_grid": PCT_MAX_GRID,
        "baseline": {"label": "阶梯 8/4/2/0 mid=0.25", "results": base_pair},
        "combos": combos,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "exp_curve.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'exp_curve.json'}")


if __name__ == "__main__":
    main()
