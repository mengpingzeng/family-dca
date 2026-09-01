#!/usr/bin/env python3
"""
档位方向深探: 低位继续加码 + 高位(买入上限)下移.

关键机制: 当最高档倍数为 0 时, 有效买入区间上限 = buy_mid, 所以「高位砍得更早」
等价于「下移 buy_mid」。本实验探索两个有效维度:
  1. 低位倍数加码: 5/3/1/0 -> 6/3/1/0 -> 6/4/2/0 -> 8/4/2/0 -> 10/5/2/0
  2. buy_mid 下移(更早停买): 0.30 -> 0.25 -> 0.20

策略 (保持不变, 均带 20万阈值收缩 + 30万封顶, base=1000):
  策略B        : PB主 / FED<=55%闸 / 卖PE / 无卖闸, S85/95
  固定年化最优 : PB主 / PB<=60%闸 / 卖FED / PB卖闸, S80/95

输出 (独立): wind_new_search/output/tier_explore.json
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
        "buy_floor": 0.10, "buy_low": 0.20, "buy_mid": 0.30, "buy_high": 0.50,
        "sell_heavy": 0.8, "sell_extreme": 0.95,
    },
}

MULTS_GRID = {
    "5/3/1/0":  (5, 3, 1, 0),
    "6/3/1/0":  (6, 3, 1, 0),
    "6/4/2/0":  (6, 4, 2, 0),
    "8/4/2/0":  (8, 4, 2, 0),
    "10/5/2/0": (10, 5, 2, 0),
}

MID_GRID = [0.30, 0.25, 0.20]

BASE = 1000
COMMISSION_RATE = 0.0005
MIN_COMMISSION = 5.0
THRESHOLD = 200_000
CAP = 300_000
POOL = 300_000


def backtest(code, params, mults, mid):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = prep_df(df)
    p = dict(params)
    p["buy_mid"] = mid
    r = run_backtest(df, p, base_amount=BASE,
                     commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                     lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP,
                     principal_pool=POOL, buy_mults=mults)
    return {
        "code": code, "name": NAMES[code],
        "xirr": r["xirr"],
        "principal_final": r["principal_final"], "principal_annual": r["principal_annual"],
        "buys": r["buys"], "sells": r["sells"], "total_invested": r["total_invested"],
    }


def main():
    print("档位方向深探 (base=1000, 阈值20万收缩 + 封顶30万)")
    combos = []
    for sname, params in STRATEGIES.items():
        for mname, mults in MULTS_GRID.items():
            for mid in MID_GRID:
                results = [backtest(code, params, mults, mid) for code in CODES]
                combos.append({"strategy": sname, "tier": mname, "buy_mid": mid,
                               "mults": list(mults), "results": results})

    hdr = f"{'策略':10} {'档位':9} {'mid':>5} | {'300年化':>7} {'300XIRR':>8} {'300终值':>9} {'买':>4} | {'500年化':>7} {'500XIRR':>8} {'500终值':>9} {'买':>4}"
    print(hdr)
    print("-" * len(hdr))
    for c in combos:
        r300, r500 = c["results"][0], c["results"][1]
        print(f"{c['strategy']:10} {c['tier']:>9} {c['buy_mid']:>5.2f} | "
              f"{r300['principal_annual']*100:6.2f}% {r300['xirr']*100:7.2f}% {r300['principal_final']:>9,.0f} {r300['buys']:>4} | "
              f"{r500['principal_annual']*100:6.2f}% {r500['xirr']*100:7.2f}% {r500['principal_final']:>9,.0f} {r500['buys']:>4}")

    # 每策略按 min(300年化, 500年化) 均衡排序 top3
    print("\n=== 均衡最优 (min(300年化,500年化) 降序) ===")
    for sname in STRATEGIES:
        sub = [c for c in combos if c["strategy"] == sname]
        sub.sort(key=lambda c: -min(c["results"][0]["principal_annual"], c["results"][1]["principal_annual"]))
        for c in sub[:3]:
            r300, r500 = c["results"][0], c["results"][1]
            print(f"  {sname} {c['tier']} mid={c['buy_mid']}: 300={r300['principal_annual']*100:.2f}%/XIRR{r300['xirr']*100:.2f}%  500={r500['principal_annual']*100:.2f}%/XIRR{r500['xirr']*100:.2f}%  (min {min(r300['principal_annual'],r500['principal_annual'])*100:.2f}%)")

    out = {
        "title": "档位方向深探",
        "note": "固定30万口径: 从回测区间第一天固化30万, 闲置部分无息; 最高档倍数0时买入上限=buy_mid",
        "base_amount": BASE, "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "strategies": STRATEGIES, "mults_grid": {k: list(v) for k, v in MULTS_GRID.items()},
        "mid_grid": MID_GRID, "combos": combos,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "tier_explore.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'tier_explore.json'}")


if __name__ == "__main__":
    main()
