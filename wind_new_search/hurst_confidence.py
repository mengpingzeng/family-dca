#!/usr/bin/env python3
"""
赫斯特指数「信心指数」买入调制 — 扫描.

在均衡策略 (8/4/2/0 mid=0.25, 30万固定口径) 上, 买入侧叠加赫斯特指数调制:
  H = 滚动窗口(20周)内对数价格的 R/S 法赫斯特指数
  d = 趋势方向 (当前价 vs 窗口首价: 涨 +1 / 跌 -1)
  调制系数 f = clamp(1 + k·d·max(H-0.5, 0), hurst_min, hurst_max)
  最终买入倍数 = 估值档位倍数 × f   (f 恒>0, 不推翻估值决策, 只调力度)

卖出侧完全不变. 扫描 k ∈ {0.5, 1.0, 1.5, 2.0}.

指标 (占用率口径已修正为 max(0, 净占用本金), 修复负占用 bug):
  固定年化 / 最大回撤 / 资金占用率 / 资金使用效率 / 买/卖次数.

数据保持独立: 不改动原始数据与既有输出, 结果落盘 output/hurst_confidence.json.
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
HURST_WINDOW = 20
HURST_MIN = 0.3
HURST_MAX = 2.0

CODES = ["000300", "000905", "000015", "000016", "000852", "399006", "399330",
         "HSI", "NDX100", "SPX500", "930931", "930930", "000688", "HSTECH"]
TRAIN_CODES = {"000300", "000905"}
NAMES = {
    "000300": "沪深300", "000905": "中证500", "000015": "上证红利", "000016": "上证50",
    "000852": "中证1000", "399006": "创业板指", "399330": "深证100", "HSI": "恒生指数",
    "NDX100": "纳斯达克100", "SPX500": "标普500", "930931": "港股通50", "930930": "港股综合",
    "000688": "科创50", "HSTECH": "恒生科技",
}

K_GRID = [0.5, 1.0, 1.5, 2.0]


def max_drawdown(series):
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


def run_one(code, k):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    window = int(df["window"].iloc[0])
    df = prep_df(df)
    kw = dict(base_amount=BASE, commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
              lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP,
              principal_pool=POOL, buy_mults=BALANCED_MULTS)
    if k is None:
        bt = build_curve(df, BALANCED_PARAMS, **kw)
    else:
        bt = build_curve(df, BALANCED_PARAMS, hurst_window=HURST_WINDOW, hurst_k=k,
                         hurst_min=HURST_MIN, hurst_max=HURST_MAX, **kw)
    meta = bt["meta"]
    daily = bt["daily"]

    ft = meta.get("first_tradable")
    dts = pd.to_datetime([d["date"] for d in daily])
    if ft is not None:
        mask = np.asarray(dts >= pd.Timestamp(ft))
    else:
        mask = np.ones(len(daily), dtype=bool)

    principals = [d["principal"] for d, m in zip(daily, mask) if m and d.get("principal") is not None]
    occ = [max(0.0, d["cum_invested"] - d["cash"]) for d, m in zip(daily, mask) if m]
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
        "principal_annual": meta["principal_annual"], "xirr": meta["xirr"],
        "buys": meta["buys"], "sells": meta["sells"],
        "max_drawdown": round(mdd, 4),
        "avg_occupied": round(avg_occupied, 0),
        "occupancy": round(occupancy, 4),
        "efficiency": round(efficiency, 4) if efficiency is not None else None,
    }


def aggregate(results):
    ann = [r["principal_annual"] for r in results]
    mdd = [r["max_drawdown"] for r in results]
    occ = [r["occupancy"] for r in results]
    eff = [r["efficiency"] for r in results if r["efficiency"] is not None]
    return {
        "annual_mean": round(sum(ann) / len(ann), 4),
        "annual_median": round(sorted(ann)[len(ann) // 2], 4),
        "mdd_mean": round(sum(mdd) / len(mdd), 4),
        "mdd_median": round(sorted(mdd)[len(mdd) // 2], 4),
        "occ_mean": round(sum(occ) / len(occ), 4),
        "eff_mean": round(sum(eff) / len(eff), 4) if eff else None,
    }


def main():
    print("赫斯特指数「信心指数」买入调制扫描 (窗口20周, 范围0.3~2.0, 30万固定口径)\n")

    base_results = [run_one(code, None) for code in CODES]
    base_test = aggregate([r for r in base_results if r["code"] not in TRAIN_CODES])
    base_train = aggregate([r for r in base_results if r["code"] in TRAIN_CODES])

    combos = []
    for k in K_GRID:
        results = [run_one(code, k) for code in CODES]
        train = [r for r in results if r["code"] in TRAIN_CODES]
        test = [r for r in results if r["code"] not in TRAIN_CODES]
        combos.append({"k": k, "train": aggregate(train), "test": aggregate(test), "results": results})

    def fmt_eff(v):
        return f"{v*100:6.2f}%" if v is not None else "  n/a  "

    print(f"{'k':>5} | {'测试年化(均值/中位)':>24} | {'测试回撤(均值/中位)':>24} | {'占用率均值':>10} | {'效率均值':>9} | 训练年化中位")
    print("-" * 118)
    for c in combos:
        t = c["test"]; tr = c["train"]
        print(f"{c['k']:>5} | {t['annual_mean']*100:6.2f}%/{t['annual_median']*100:6.2f}% | "
              f"{t['mdd_mean']*100:6.2f}%/{t['mdd_median']*100:6.2f}% | {t['occ_mean']*100:6.2f}% | "
              f"{fmt_eff(t['eff_mean'])} | {tr['annual_median']*100:6.2f}%")
    print("-" * 118)
    print(f"基线(无调制) 测试: 年化 均值{base_test['annual_mean']*100:.2f}%/中位{base_test['annual_median']*100:.2f}% | "
          f"回撤 均值{base_test['mdd_mean']*100:.2f}%/中位{base_test['mdd_median']*100:.2f}% | "
          f"占用{base_test['occ_mean']*100:.2f}% | 效率{fmt_eff(base_test['eff_mean'])}")
    print(f"基线(无调制) 训练: 年化 均值{base_train['annual_mean']*100:.2f}%/中位{base_train['annual_median']*100:.2f}% | "
          f"回撤 均值{base_train['mdd_mean']*100:.2f}% | 占用{base_train['occ_mean']*100:.2f}%")

    # 逐指数明细 (k=1.0 vs 基线)
    print("\n=== 明细: k=1.0 vs 基线 (年化 / 回撤 / 占用率 / 买/卖) ===")
    k1 = next(c for c in combos if c["k"] == 1.0)
    base_by_code = {r["code"]: r for r in base_results}
    print(f"{'代码':8} {'名称':10} | {'年化(基/H)':>20} | {'回撤(基/H)':>20} | {'占用(基/H)':>18} | {'买(基/H)':>12}")
    for r in k1["results"]:
        b = base_by_code[r["code"]]
        print(f"{r['code']:8} {r['name']:10} | {b['principal_annual']*100:6.2f}%/{r['principal_annual']*100:6.2f}% | "
              f"{b['max_drawdown']*100:6.2f}%/{r['max_drawdown']*100:6.2f}% | "
              f"{b['occupancy']*100:5.1f}%/{r['occupancy']*100:5.1f}% | {b['buys']:>4}/{r['buys']:>4}")

    out = {
        "title": "赫斯特指数信心指数 买入调制扫描",
        "note": "H=R/S法(20周对数价格); d=趋势方向; f=clamp(1+k·d·max(H-0.5,0), min, max); 仅买入侧调制, 卖出侧不变",
        "hurst_window": HURST_WINDOW, "hurst_min": HURST_MIN, "hurst_max": HURST_MAX,
        "k_grid": K_GRID,
        "params": BALANCED_PARAMS, "buy_mults": list(BALANCED_MULTS), "base_amount": BASE,
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "metrics": {
            "max_drawdown": "账户净值最大回撤比例",
            "avg_occupied": "时间加权平均净占用本金(元), max(0, 净占用) 修正负占用",
            "occupancy": "资金占用率=avg_occupied/30万",
            "efficiency": "资金使用效率=期末净收益/(avg_occupied×年数)",
        },
        "baseline": {"train": base_train, "test": base_test, "results": base_results},
        "combos": combos,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "hurst_confidence.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'hurst_confidence.json'}")


if __name__ == "__main__":
    main()
