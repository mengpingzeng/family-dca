#!/usr/bin/env python3
"""
策略验证 — 用严格买入统一最优策略，验证 3 个新指数

固定策略: B8/12/22/40 S75/85, FED=off, PBv=off, 不限本金 + 闲置2% + min 10笔

验证标的:
  中证1000 (000852) : 10年窗口 + 5年窗口
  上证红利 (000015) : 10年窗口
  深证红利 (399324) : 10年窗口

输出: latest_validate.json (供前端总览页)
"""
import os, sys, json
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_search import run_backtest

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MERGED_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, os.pardir, os.pardir, "data-store", "parquet", "aks_merged"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_PARAMS = {
    "buy_floor": 0.08, "buy_low": 0.12, "buy_mid": 0.22, "buy_high": 0.40,
    "sell_heavy": 0.75, "sell_extreme": 0.85,
    "fed_gate": None, "pb_veto": None, "pb_sell": None,
}

BASE_AMOUNT = 500
IDLE_CASH_RATE = 0.02
MIN_TRADES = 10

VALIDATION_TARGETS = [
    {"code": "000852", "name": "中证1000", "windows": [10, 5]},
    {"code": "000015", "name": "上证红利", "windows": [10]},
    {"code": "399324", "name": "深证红利", "windows": [10]},
]


def main():
    print("=" * 60)
    print("策略验证 — 固定策略 B8/12/22/40 S75/85")
    print("=" * 60)

    summary = []
    for target in VALIDATION_TARGETS:
        code = target["code"]
        name = target["name"]
        path = os.path.join(MERGED_DIR, f"{code}.parquet")
        if not os.path.exists(path):
            print(f"[SKIP] {name}({code}): 数据不存在")
            continue

        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])

        for w in target["windows"]:
            r = run_backtest(df, FIXED_PARAMS, w, base_amount=BASE_AMOUNT,
                             idle_cash_rate=IDLE_CASH_RATE, min_trades=MIN_TRADES)
            valid = r["trades"] >= MIN_TRADES
            print(f"[{code}] {name} w{w}: XIRR={r['xirr']*100:.2f}% "
                  f"投入={r['total_invested']:.0f} 终值={r['final_value']:.0f} "
                  f"交易={r['trades']} (买{r['buys']}/卖{r['sells']})")

            summary.append({
                "code": code,
                "name": name,
                "window": w,
                "xirr": r["xirr"],
                "final_return": r["final_return"],
                "total_invested": r["total_invested"],
                "total_cash_in": r["total_cash_in"],
                "net_principal": r["net_principal"],
                "final_value": r["final_value"],
                "position_value": r.get("position_value", 0),
                "idle_cash": r.get("idle_cash", 0),
                "interest_earned": r.get("interest_earned", 0),
                "trades": r["trades"],
                "buys": r["buys"],
                "sells": r["sells"],
                "valid": valid,
            })

    out = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": FIXED_PARAMS,
        "base_amount": BASE_AMOUNT,
        "idle_cash_rate": IDLE_CASH_RATE,
        "min_trades": MIN_TRADES,
        "summary": summary,
    }

    out_path = os.path.join(OUTPUT_DIR, "latest_validate.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n结果保存: {out_path}")
    return summary


if __name__ == "__main__":
    main()
