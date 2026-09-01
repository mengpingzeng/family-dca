#!/usr/bin/env python3
"""
20周均线制动阀 (买入侧) — 前后对比回测.

在均衡策略上, 买入侧叠加 20 周收盘价简单均线 (SMA20) 制动阀:
  price < SMA20 (弱势/下跌)  -> 本次买入作废(不买)
  price >= SMA20 (企稳/转强) -> 恢复买入
  warm-up: 前 20 周无均线值, 照常交易不制动 (选 A)
  卖出侧: 完全不变

新增指标 (资金使用效率 + 回撤):
  max_drawdown  账户净值(固定30万口径 principal 曲线)最大回撤
  avg_occupied  时间加权平均净占用本金 = Σ(每周净占用本金)/周数
  occupancy     资金占用率 = avg_occupied / 30万
  efficiency    资金使用效率 = 期末净收益 / (avg_occupied × 年数)  (占用资金的年化回报)

数据保持独立: 不改动任何原始数据, 结果落盘到独立文件 output/trend_ma_brake.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import build_curve, prep_df

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
MA_WINDOW = 20

CODES = ["000300", "000905", "000015", "000016", "000852", "399006", "399330",
         "HSI", "NDX100", "SPX500", "930931", "930930", "000688", "HSTECH"]
TRAIN_CODES = {"000300", "000905"}
NAMES = {
    "000300": "沪深300", "000905": "中证500", "000015": "上证红利", "000016": "上证50",
    "000852": "中证1000", "399006": "创业板指", "399330": "深证100", "HSI": "恒生指数",
    "NDX100": "纳斯达克100", "SPX500": "标普500", "930931": "港股通50", "930930": "港股综合",
    "000688": "科创50", "HSTECH": "恒生科技",
}


def max_drawdown(series):
    """序列最大回撤(比例)."""
    peak = float("-inf")
    mdd = 0.0
    for v in series:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > mdd:
                mdd = dd
    return mdd


def run(code, ma_window):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    window = int(df["window"].iloc[0])
    df = prep_df(df)
    bt = build_curve(df, BALANCED_PARAMS, base_amount=BASE,
                     commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                     lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP,
                     principal_pool=POOL, buy_mults=BALANCED_MULTS, ma_window=ma_window)
    meta = bt["meta"]
    daily = bt["daily"]

    ft = meta.get("first_tradable")
    dts = pd.to_datetime([d["date"] for d in daily])
    if ft is not None:
        mask = np.asarray(dts >= pd.Timestamp(ft))
    else:
        mask = np.ones(len(daily), dtype=bool)

    principals = [d["principal"] for d, m in zip(daily, mask) if m and d.get("principal") is not None]
    occ = [d["cum_invested"] - d["cash"] for d, m in zip(daily, mask) if m]
    end_date = dts[-1]

    mdd = max_drawdown(principals)
    avg_occupied = sum(occ) / len(occ) if occ else 0.0
    occupancy = avg_occupied / POOL if POOL else 0.0

    if ft is not None:
        years = (end_date - pd.Timestamp(ft)).days / 365.25
    else:
        years = 0.0

    net_profit = (meta["principal_final"] or POOL) - POOL
    efficiency = net_profit / (avg_occupied * years) if avg_occupied > 0 and years > 0 else None

    return {
        "code": code, "name": NAMES.get(code, code), "window": window,
        "ma_window": ma_window,
        "principal_annual": meta["principal_annual"],
        "xirr": meta["xirr"],
        "principal_final": meta["principal_final"],
        "buys": meta["buys"], "sells": meta["sells"],
        "max_drawdown": round(mdd, 4),
        "avg_occupied": round(avg_occupied, 0),
        "occupancy": round(occupancy, 4),
        "efficiency": round(efficiency, 4) if efficiency is not None else None,
    }


def main():
    print("20周均线制动阀 (买入侧) 前后对比 — 均衡策略 8/4/2/0 mid=0.25, 30万固定口径\n")
    rows = []
    for code in CODES:
        base = run(code, None)
        braked = run(code, MA_WINDOW)
        tag = "训练" if code in TRAIN_CODES else ""
        rows.append({"code": code, "name": base["name"], "tag": tag,
                     "window": base["window"], "baseline": base, "braked": braked})

    hdr = (f"{'代码':8} {'名称':10} | {'年化(原/制动)':>22} | {'回撤(原/制动)':>22} | "
           f"{'占用率(原/制动)':>22} | {'效率(原/制动)':>20} | {'买(原/制动)':>14}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        b, k = r["baseline"], r["braked"]
        def eff(v):
            return f"{v*100:.2f}%" if v is not None else "  n/a "
        print(f"{r['code']:8} {r['name']:10} | "
              f"{b['principal_annual']*100:6.2f}%/ {k['principal_annual']*100:6.2f}% | "
              f"{b['max_drawdown']*100:6.2f}%/ {k['max_drawdown']*100:6.2f}% | "
              f"{b['occupancy']*100:6.2f}%/ {k['occupancy']*100:6.2f}% | "
              f"{eff(b['efficiency'])} / {eff(k['efficiency'])} | "
              f"{b['buys']:>4}/{k['buys']:>4}  {r['tag']}")

    # 汇总 (测试集: 排除训练两指数)
    test = [r for r in rows if r["code"] not in TRAIN_CODES]
    def summary(rows_, key):
        vals = [r[key]["principal_annual"] for r in rows_]
        mdd = [r[key]["max_drawdown"] for r in rows_]
        occ = [r[key]["occupancy"] for r in rows_]
        return {
            "annual_mean": round(sum(vals) / len(vals), 4),
            "annual_median": round(sorted(vals)[len(vals) // 2], 4),
            "mdd_mean": round(sum(mdd) / len(mdd), 4),
            "occ_mean": round(sum(occ) / len(occ), 4),
        }
    s_base = summary(test, "baseline")
    s_braked = summary(test, "braked")
    print("\n=== 测试集 12 指数汇总 (均值) ===")
    print(f"  固定年化 均值: 原 {s_base['annual_mean']*100:.2f}% -> 制动 {s_braked['annual_mean']*100:.2f}%")
    print(f"  固定年化 中位: 原 {s_base['annual_median']*100:.2f}% -> 制动 {s_braked['annual_median']*100:.2f}%")
    print(f"  最大回撤 均值: 原 {s_base['mdd_mean']*100:.2f}% -> 制动 {s_braked['mdd_mean']*100:.2f}%")
    print(f"  资金占用率均值: 原 {s_base['occ_mean']*100:.2f}% -> 制动 {s_braked['occ_mean']*100:.2f}%")

    out = {
        "title": "20周均线制动阀(买入侧) 前后对比",
        "note": "制动阀规则: price<SMA20 不买, price>=SMA20 恢复买入; warm-up 前20周不制动; 卖出侧不变",
        "ma_window": MA_WINDOW,
        "params": BALANCED_PARAMS, "buy_mults": list(BALANCED_MULTS), "base_amount": BASE,
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "cost_model": {"commission_rate": COMMISSION_RATE, "min_commission": MIN_COMMISSION, "lot_size": 0},
        "metrics": {
            "max_drawdown": "账户净值(固定30万口径 principal 曲线)最大回撤比例",
            "avg_occupied": "时间加权平均净占用本金(元) = Σ每周净占用本金/周数",
            "occupancy": "资金占用率 = avg_occupied / 30万",
            "efficiency": "资金使用效率 = 期末净收益 / (avg_occupied × 年数), 占用资金的年化回报",
        },
        "results": rows,
        "test_summary": {"baseline": s_base, "braked": s_braked},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "trend_ma_brake.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'trend_ma_brake.json'}")


if __name__ == "__main__":
    main()
