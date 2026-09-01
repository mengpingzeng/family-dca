#!/usr/bin/env python3
"""
档位结构调整实验 (低位多买 / 高位尽量少买).

宗旨: 保持 base=1000, 通过调整「买入档位倍数」实现低位加码、高位减码:
  - 原始         : 3x / 2x / 1x / 0.5x   (高位轻仓 0.5x)
  - 砍高位       : 3x / 2x / 1x / 0x     (高位不买)
  - 低位加码+砍高位 : 4x / 2x / 1x / 0x
  - 强低位+砍高位   : 5x / 3x / 1x / 0x

策略 (保持不变, 均带 20万阈值收缩 + 30万封顶):
  策略B        : PB主 / FED<=55%闸 / 卖PE / 无卖闸, B10/15/30/70, S85/95
  固定年化最优 : PB主 / PB<=60%闸 / 卖FED / PB卖闸, B10/20/30/50, S80/95

固定配置: base=1000, 费用 万5低消5, 固定30万口径(回测区间第一天固化)

输出 (独立, 不覆盖旧文件):
  wind_new_search/output/tier_structure.json
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

# 档位倍数 [<bf, <bl, <bm, <bh] 对应倍数
TIERS = {
    "3/2/1/0.5x": (3, 2, 1, 0.5),
    "3/2/1/0x":   (3, 2, 1, 0),
    "4/2/1/0x":   (4, 2, 1, 0),
    "5/3/1/0x":   (5, 3, 1, 0),
}

BASE = 1000
COMMISSION_RATE = 0.0005
MIN_COMMISSION = 5.0
THRESHOLD = 200_000
CAP = 300_000
POOL = 300_000


def backtest(code, params, mults):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = prep_df(df)
    r = run_backtest(df, params, base_amount=BASE,
                     commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                     lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP,
                     principal_pool=POOL, buy_mults=mults)
    return {
        "code": code, "name": NAMES[code],
        "xirr": r["xirr"], "final_return": r["final_return"],
        "total_invested": r["total_invested"],
        "principal_final": r["principal_final"], "principal_annual": r["principal_annual"],
        "principal_return": r["principal_return"],
        "buys": r["buys"], "sells": r["sells"],
    }


def main():
    print("档位结构调整 (base=1000, 阈值20万收缩 + 封顶30万)")
    combos = []
    for sname, params in STRATEGIES.items():
        for tname, mults in TIERS.items():
            results = [backtest(code, params, mults) for code in CODES]
            combos.append({"strategy": sname, "tier": tname, "mults": list(mults), "results": results})

    hdr = f"{'策略':12} {'档位':11} | {'300固定年化':>9} {'300 XIRR':>8} {'300终值':>10} {'买/卖':>7} | {'500固定年化':>9} {'500 XIRR':>8} {'500终值':>10} {'买/卖':>7}"
    print(hdr)
    print("-" * len(hdr))
    for c in combos:
        r300, r500 = c["results"][0], c["results"][1]
        print(f"{c['strategy']:12} {c['tier']:>11} | "
              f"{r300['principal_annual']*100:8.2f}% {r300['xirr']*100:7.2f}% {r300['principal_final']:>10,.0f} {r300['buys']:>3}/{r300['sells']:<3} | "
              f"{r500['principal_annual']*100:8.2f}% {r500['xirr']*100:7.2f}% {r500['principal_final']:>10,.0f} {r500['buys']:>3}/{r500['sells']:<3}")

    out = {
        "title": "档位结构调整 (低位多买/高位少买)",
        "note": "固定30万口径: 从回测区间第一天(pe_pct首次有效)固化30万, 闲置部分无息",
        "cost_model": {"commission_rate": COMMISSION_RATE, "min_commission": MIN_COMMISSION, "lot_size": 0},
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "base_amount": BASE,
        "strategies": STRATEGIES,
        "tiers": {k: list(v) for k, v in TIERS.items()},
        "combos": combos,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "tier_structure.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'tier_structure.json'}")


if __name__ == "__main__":
    main()
