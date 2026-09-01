#!/usr/bin/env python3
"""
蛋卷数据策略验证 — 严格买入统一最优策略 (B8/12/22/40 S75/85, 5年窗口)

蛋卷 PE 周期短，故窗口改为 5 年。

输出: latest_validate_dj.json (供前端)
"""
import os, sys, json
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_search import run_backtest

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MERGED_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, os.pardir, os.pardir, "data-store", "parquet", "aks_merged_dj"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_PARAMS = {
    "buy_floor": 0.08, "buy_low": 0.12, "buy_mid": 0.22, "buy_high": 0.40,
    "sell_heavy": 0.75, "sell_extreme": 0.85,
    "fed_gate": None, "pb_veto": None, "pb_sell": None,
}

BASE_AMOUNT = 500
IDLE_CASH_RATE = 0.02
MIN_TRADES = 10
WINDOW = 5

DJ_INDICES = [
    {"code": "000015", "name": "上证红利", "market": "A股"},
    {"code": "000016", "name": "上证50", "market": "A股"},
    {"code": "000300", "name": "沪深300", "market": "A股"},
    {"code": "000688", "name": "科创50", "market": "A股"},
    {"code": "000852", "name": "中证1000", "market": "A股"},
    {"code": "000905", "name": "中证500", "market": "A股"},
    {"code": "000922", "name": "中证红利", "market": "A股"},
    {"code": "399006", "name": "创业板指", "market": "A股"},
    {"code": "399330", "name": "深证100", "market": "A股"},
    {"code": "HSI", "name": "恒生指数", "market": "港股"},
    {"code": "HSTECH", "name": "恒生科技", "market": "港股"},
    {"code": "NDX100", "name": "纳斯达克100", "market": "美股"},
    {"code": "SPX500", "name": "标普500", "market": "美股"},
]


def main():
    print("=" * 60)
    print("蛋卷数据策略验证 — B8/12/22/40 S75/85 (5年窗口)")
    print("=" * 60)

    summary = []
    for target in DJ_INDICES:
        code = target["code"]
        name = target["name"]
        path = os.path.join(MERGED_DIR, f"{code}.parquet")
        if not os.path.exists(path):
            print(f"[SKIP] {name}({code}): 数据不存在")
            continue

        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])

        pct_col = f"pe_pct_w{WINDOW}"
        if pct_col not in df.columns:
            print(f"[SKIP] {name}({code}): 无 {WINDOW}年百分位列")
            continue
        valid = df[df[pct_col].notna()]
        if len(valid) < 100:
            print(f"[SKIP] {name}({code}): 有效交易日不足 ({len(valid)})")
            summary.append({
                "code": code, "name": name, "market": target["market"],
                "window": WINDOW, "xirr": None, "valid": False,
                "reason": "不足5年历史数据",
            })
            continue

        r = run_backtest(df, FIXED_PARAMS, WINDOW, base_amount=BASE_AMOUNT,
                         idle_cash_rate=IDLE_CASH_RATE, min_trades=MIN_TRADES)
        valid_trade = r["trades"] >= MIN_TRADES
        xirr_str = f"{r['xirr']*100:.2f}%" if valid_trade else "交易不足"
        print(f"[{code}] {name}: XIRR={xirr_str} 投入={r['total_invested']:.0f} "
              f"终值={r['final_value']:.0f} 交易={r['trades']} (买{r['buys']}/卖{r['sells']})")

        summary.append({
            "code": code,
            "name": name,
            "market": target["market"],
            "window": WINDOW,
            "xirr": r["xirr"] if valid_trade else None,
            "final_return": r["final_return"] if valid_trade else None,
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
            "valid": valid_trade,
            "reason": "" if valid_trade else f"交易不足({r['trades']}<{MIN_TRADES})",
        })

    out = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": FIXED_PARAMS,
        "base_amount": BASE_AMOUNT,
        "idle_cash_rate": IDLE_CASH_RATE,
        "min_trades": MIN_TRADES,
        "window": WINDOW,
        "source": "danjuan (pe_ttm_dj, 周频)",
        "summary": summary,
    }

    out_path = os.path.join(OUTPUT_DIR, "latest_validate_dj.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n结果保存: {out_path}")
    return summary


if __name__ == "__main__":
    main()
