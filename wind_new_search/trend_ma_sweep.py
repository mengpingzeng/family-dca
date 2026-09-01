#!/usr/bin/env python3
"""
20周均线制动阀 β/γ 二维扫描 — 找"软化制动 + 企稳加力"的最优组合.

β (ma_below): price < SMA20 时买入倍数缩放 (0=全刹, 0.25/0.5/0.75=软刹)
γ (ma_above): price >= SMA20 时买入倍数缩放 (1=不加力, 1.5/2/3=企稳加力)

在均衡策略 (8/4/2/0 mid=0.25, 30万固定口径) 上叠加制动阀, 全 14 指数回测,
指标: 固定年化 / 最大回撤 / 资金占用率 / 资金使用效率.

数据保持独立: 不改动原始数据与既有输出, 结果落盘 output/trend_ma_sweep.json.
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

BETA_GRID = [0.0, 0.25, 0.5, 0.75]
GAMMA_GRID = [1.0, 1.5, 2.0, 3.0]


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


def run_one(code, beta, gamma):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    window = int(df["window"].iloc[0])
    df = prep_df(df)
    bt = build_curve(df, BALANCED_PARAMS, base_amount=BASE,
                     commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                     lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP,
                     principal_pool=POOL, buy_mults=BALANCED_MULTS,
                     ma_window=MA_WINDOW, ma_below=beta, ma_above=gamma)
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
    print("20周均线制动阀 β/γ 扫描 — 均衡策略 8/4/2/0 mid=0.25, 30万固定口径\n")

    combos = []
    for beta in BETA_GRID:
        for gamma in GAMMA_GRID:
            results = [run_one(code, beta, gamma) for code in CODES]
            train = [r for r in results if r["code"] in TRAIN_CODES]
            test = [r for r in results if r["code"] not in TRAIN_CODES]
            combos.append({
                "beta": beta, "gamma": gamma,
                "train": aggregate(train), "test": aggregate(test),
                "results": results,
            })

    # 基线 (无制动阀) 参考
    base_results = [run_one(code, 0.0, 1.0) for code in CODES]  # 占位, 下面单独用 ma_window=None
    base_results = []
    for code in CODES:
        df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        df = prep_df(df)
        bt = build_curve(df, BALANCED_PARAMS, base_amount=BASE,
                         commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                         lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP,
                         principal_pool=POOL, buy_mults=BALANCED_MULTS)
        meta = bt["meta"]; daily = bt["daily"]
        ft = meta.get("first_tradable"); dts = pd.to_datetime([d["date"] for d in daily])
        mask = np.asarray(dts >= pd.Timestamp(ft)) if ft else np.ones(len(daily), dtype=bool)
        principals = [d["principal"] for d, m in zip(daily, mask) if m and d.get("principal") is not None]
        occ = [d["cum_invested"] - d["cash"] for d, m in zip(daily, mask) if m]
        years = (dts[-1] - pd.Timestamp(ft)).days / 365.25 if ft else 0.0
        mdd = max_drawdown(principals)
        avg_occ = sum(occ) / len(occ) if occ else 0.0
        occ_rate = avg_occ / POOL
        net_profit = (meta["principal_final"] or POOL) - POOL
        eff = net_profit / (avg_occ * years) if avg_occ > 0 and years > 0 else None
        base_results.append({
            "code": code, "name": NAMES.get(code, code),
            "principal_annual": meta["principal_annual"],
            "max_drawdown": round(mdd, 4),
            "occupancy": round(occ_rate, 4),
            "efficiency": round(eff, 4) if eff is not None else None,
            "buys": meta["buys"],
        })
    base_test = aggregate([r for r in base_results if r["code"] not in TRAIN_CODES])
    base_train = aggregate([r for r in base_results if r["code"] in TRAIN_CODES])

    print(f"{'β':>5} {'γ':>4} | {'测试年化(均值/中位)':>24} | {'测试回撤(均值/中位)':>24} | {'占用率均值':>10} | {'效率均值':>10}")
    print("-" * 100)
    for c in combos:
        t = c["test"]
        print(f"{c['beta']:>5} {c['gamma']:>4} | {t['annual_mean']*100:6.2f}%/{t['annual_median']*100:6.2f}% | "
              f"{t['mdd_mean']*100:6.2f}%/{t['mdd_median']*100:6.2f}% | {t['occ_mean']*100:6.2f}% | "
              f"{t['eff_mean']*100:6.2f}%" if t['eff_mean'] is not None else
              f"{c['beta']:>5} {c['gamma']:>4} | {t['annual_mean']*100:6.2f}%/{t['annual_median']*100:6.2f}% | "
              f"{t['mdd_mean']*100:6.2f}%/{t['mdd_median']*100:6.2f}% | {t['occ_mean']*100:6.2f}% |    n/a  ")
    print("-" * 100)
    print(f"基线(无制动阀) 测试: 年化 均值{base_test['annual_mean']*100:.2f}%/中位{base_test['annual_median']*100:.2f}% | "
          f"回撤 均值{base_test['mdd_mean']*100:.2f}%/中位{base_test['mdd_median']*100:.2f}% | "
          f"占用率{base_test['occ_mean']*100:.2f}% | 效率{base_test['eff_mean']*100:.2f}%")
    print(f"基线(无制动阀) 训练: 年化 均值{base_train['annual_mean']*100:.2f}%/中位{base_train['annual_median']*100:.2f}% | "
          f"回撤 均值{base_train['mdd_mean']*100:.2f}% | 占用率{base_train['occ_mean']*100:.2f}%")

    # 帕累托前沿: (测试年化中位, -回撤均值) 非支配点
    pareto = []
    for c in combos:
        t = c["test"]
        dominated = any(o["test"]["annual_median"] >= t["annual_median"]
                        and o["test"]["mdd_mean"] <= t["mdd_mean"]
                        and (o["test"]["annual_median"] > t["annual_median"] or o["test"]["mdd_mean"] < t["mdd_mean"])
                        for o in combos)
        if not dominated:
            pareto.append(c)
    pareto.sort(key=lambda c: c["test"]["mdd_mean"])
    print("\n=== 帕累托前沿 (测试年化中位 ↑ vs 测试回撤均值 ↓) ===")
    for c in pareto:
        t = c["test"]
        print(f"  β={c['beta']} γ={c['gamma']}: 年化中位{t['annual_median']*100:.2f}% 回撤均值{t['mdd_mean']*100:.2f}% "
              f"占用{t['occ_mean']*100:.2f}% 效率{t['eff_mean']*100:.2f}%")

    out = {
        "title": "20周均线制动阀 β/γ 二维扫描",
        "note": "β=price<SMA20买入倍数缩放(0全刹), γ=price>=SMA20买入倍数缩放(1不加力); 卖出侧不变; warm-up前20周不制动",
        "ma_window": MA_WINDOW,
        "beta_grid": BETA_GRID, "gamma_grid": GAMMA_GRID,
        "params": BALANCED_PARAMS, "buy_mults": list(BALANCED_MULTS), "base_amount": BASE,
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "metrics": {
            "max_drawdown": "账户净值最大回撤比例", "avg_occupied": "时间加权平均净占用本金(元)",
            "occupancy": "资金占用率=avg_occupied/30万",
            "efficiency": "资金使用效率=期末净收益/(avg_occupied×年数)",
        },
        "baseline": {"train": base_train, "test": base_test, "results": base_results},
        "combos": combos,
        "pareto": [{"beta": c["beta"], "gamma": c["gamma"], "test": c["test"], "train": c["train"]} for c in pareto],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "trend_ma_sweep.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'trend_ma_sweep.json'}")


if __name__ == "__main__":
    main()
