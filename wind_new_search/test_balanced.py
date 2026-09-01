#!/usr/bin/env python3
"""
均衡策略 测试集验证 — 用均衡策略在 12 个测试指数上回测, 检验是否过拟合。

均衡策略 (训练集 000300/000905 上固定30万口径最优):
  主信号 PB / 买入闸门 FED<=55% / 卖 PE (S85/95 无卖闸)
  买档 8x/4x/2x/0x, buy_mid=0.25, base=1000
  口径: 阈值20万收缩 + 封顶30万 + 固定30万 + 费用万5低消5

输出 (独立): wind_new_search/output/test_balanced.json
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

BALANCED_PARAMS = {
    "buy_signal": "PB", "buy_gate": "FED", "buy_gate_cap": 0.55,
    "sell_signal": "PE", "sell_gate": None, "sell_gate_floor": None,
    "buy_floor": 0.10, "buy_low": 0.15, "buy_mid": 0.25, "buy_high": 0.70,
    "sell_heavy": 0.85, "sell_extreme": 0.95,
}
BALANCED_MULTS = (8, 4, 2, 0)
BASE = 1000
COMMISSION_RATE = 0.0005
MIN_COMMISSION = 5.0
THRESHOLD = 200_000
CAP = 300_000
POOL = 300_000

TRAIN_CODES = ["000300", "000905"]
TEST_CODES = ["000015", "000016", "000852", "399006", "399330",
              "HSI", "NDX100", "SPX500", "930931", "930930", "000688", "HSTECH"]
NAMES = {
    "000300": "沪深300", "000905": "中证500", "000015": "上证红利", "000016": "上证50",
    "000852": "中证1000", "399006": "创业板指", "399330": "深证100", "HSI": "恒生指数",
    "NDX100": "纳斯达克100", "SPX500": "标普500", "930931": "港股通50", "930930": "港股综合",
    "000688": "科创50", "HSTECH": "恒生科技",
}


def backtest(code):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    window = int(df["window"].iloc[0])
    df = prep_df(df)
    t = df[df["pe_pct"].notna()]
    start_date = t["date"].min()
    end_date = t["date"].max()
    r = run_backtest(df, BALANCED_PARAMS, base_amount=BASE,
                     commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                     lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP,
                     principal_pool=POOL, buy_mults=BALANCED_MULTS)
    # 满仓持有年化 (首个可交易日买入持有到期末)
    first_price = t["price"].iloc[0]
    last_price = t["price"].iloc[-1]
    years = (end_date - start_date).days / 365.25
    bh = (last_price / first_price) ** (1 / years) - 1 if years > 0 and first_price > 0 else 0.0
    return {
        "code": code, "name": NAMES.get(code, code), "window": window,
        "start_date": str(start_date.date()), "end_date": str(end_date.date()),
        "xirr": r["xirr"], "final_return": r["final_return"],
        "total_invested": r["total_invested"], "final_value": r["final_value"],
        "principal_final": r["principal_final"], "principal_annual": r["principal_annual"],
        "principal_return": r["principal_return"],
        "buys": r["buys"], "sells": r["sells"],
        "buy_hold_annual": round(bh, 4),
    }


def main():
    print("均衡策略 测试集验证 (8/4/2/0 mid=0.25, 阈值20万+封顶30万)\n")
    print(f"{'代码':8} {'名称':10} {'窗口':>4} | {'固定年化':>8} {'XIRR':>8} {'满仓年化':>8} {'终值':>10} {'买/卖':>8}")
    print("-" * 80)
    train_ref = []
    for code in TRAIN_CODES:
        r = backtest(code)
        train_ref.append(r)
        print(f"{code:8} {r['name']:10} {r['window']:>4} | {r['principal_annual']*100:7.2f}% {r['xirr']*100:7.2f}% {r['buy_hold_annual']*100:7.2f}% {r['principal_final']:>10,.0f} {r['buys']:>3}/{r['sells']:<3}  (训练)")
    print("-" * 80)
    results = []
    for code in TEST_CODES:
        r = backtest(code)
        results.append(r)
        print(f"{code:8} {r['name']:10} {r['window']:>4} | {r['principal_annual']*100:7.2f}% {r['xirr']*100:7.2f}% {r['buy_hold_annual']*100:7.2f}% {r['principal_final']:>10,.0f} {r['buys']:>3}/{r['sells']:<3}")

    # 汇总统计
    anns = [r["principal_annual"] for r in results]
    xirrs = [r["xirr"] for r in results]
    beats_bh = sum(1 for r in results if r["principal_annual"] > r["buy_hold_annual"])
    print("-" * 80)
    print(f"测试集 12 指数: 固定年化 均值={sum(anns)/len(anns)*100:.2f}% 中位={sorted(anns)[len(anns)//2]*100:.2f}% "
          f"min={min(anns)*100:.2f}% max={max(anns)*100:.2f}%")
    print(f"               XIRR      均值={sum(xirrs)/len(xirrs)*100:.2f}% 中位={sorted(xirrs)[len(xirrs)//2]*100:.2f}% "
          f"min={min(xirrs)*100:.2f}% max={max(xirrs)*100:.2f}%")
    print(f"跑赢满仓持有: {beats_bh}/{len(results)} 个指数")

    out = {
        "title": "均衡策略 测试集验证",
        "params": BALANCED_PARAMS, "buy_mults": list(BALANCED_MULTS), "base_amount": BASE,
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "cost_model": {"commission_rate": COMMISSION_RATE, "min_commission": MIN_COMMISSION, "lot_size": 0},
        "train_reference": train_ref, "results": results,
        "summary": {
            "n_test": len(results),
            "annual_mean": round(sum(anns) / len(anns), 4),
            "annual_median": round(sorted(anns)[len(anns) // 2], 4),
            "xirr_mean": round(sum(xirrs) / len(xirrs), 4),
            "xirr_median": round(sorted(xirrs)[len(xirrs) // 2], 4),
            "beats_buy_hold": beats_bh,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "test_balanced.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'test_balanced.json'}")


if __name__ == "__main__":
    main()
