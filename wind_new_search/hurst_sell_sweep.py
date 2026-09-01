#!/usr/bin/env python3
"""
赫斯特指数 — 卖出侧优化 (买入侧固定 打折0.5/加码1.0).

买入侧固定配置 (上一轮已固化):
  H>0.5 且方向跌 -> 买入倍数 × 0.5 (打折); H>0.5 且方向涨 -> × 1.0 (不加码)
  H = 单段 R/S, 完整价格历史滚动, L=250(长指数)/104(短指数), H 阈值固定 0.5

卖出侧优化 (本轮扫描):
  在 PE 卖出 (≥85% 卖20%, ≥95% 清仓) 上叠加赫斯特调制:
  H>0.5 且方向涨 -> 卖出比例 × hurst_sell_up (抑制卖飞, 让利润奔跑)
  H>0.5 且方向跌 -> 卖出比例 × 1.0 (不变)

扫描: hurst_sell_up ∈ {1.0, 0.5, 0.0} + 基线.

指标 (含夏普): 固定年化 / 最大回撤 / 夏普 / 资金占用率 / 资金使用效率 / 买/卖.

数据保持独立: 不改动原始数据与既有输出, 结果落盘 output/hurst_sell_sweep.json.
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

CODES = ["000300", "000905", "000015", "000016", "000852", "399006", "399330",
         "HSI", "NDX100", "SPX500", "930931", "930930", "000688", "HSTECH"]
TRAIN_CODES = {"000300", "000905"}
NAMES = {
    "000300": "沪深300", "000905": "中证500", "000015": "上证红利", "000016": "上证50",
    "000852": "中证1000", "399006": "创业板指", "399330": "深证100", "HSI": "恒生指数",
    "NDX100": "纳斯达克100", "SPX500": "标普500", "930931": "港股通50", "930930": "港股综合",
    "000688": "科创50", "HSTECH": "恒生科技",
}

SELL_UP_GRID = [1.0, 0.5, 0.0]
FIX_DISCOUNT = 0.5
FIX_BOOST = 1.0


def window_of(code):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet", columns=["window"])
    w = int(df["window"].iloc[0])
    return 250 if w >= 10 else 104


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


def sharpe_annual(daily, ft):
    dts = pd.to_datetime([d["date"] for d in daily])
    mask = np.asarray(dts >= pd.Timestamp(ft)) if ft else np.ones(len(daily), dtype=bool)
    pr = np.array([d["principal"] for d, m in zip(daily, mask) if m and d.get("principal") is not None], dtype=float)
    if len(pr) < 3:
        return None
    r = pr[1:] / pr[:-1] - 1.0
    s = r.std(ddof=0)
    if s <= 0:
        return None
    return float(r.mean() / s * np.sqrt(52))


def run_one(code, sell_up, use_hurst=True):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    window = int(df["window"].iloc[0])
    df = prep_df(df)
    kw = dict(base_amount=BASE, commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
              lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP,
              principal_pool=POOL, buy_mults=BALANCED_MULTS)
    if not use_hurst:
        bt = build_curve(df, BALANCED_PARAMS, **kw)
    else:
        bt = build_curve(df, BALANCED_PARAMS, hurst_window=window_of(code),
                         hurst_discount=FIX_DISCOUNT, hurst_boost=FIX_BOOST,
                         hurst_sell_up=sell_up, **kw)
    meta = bt["meta"]
    daily = bt["daily"]

    ft = meta.get("first_tradable")
    dts = pd.to_datetime([d["date"] for d in daily])
    mask = np.asarray(dts >= pd.Timestamp(ft)) if ft else np.ones(len(daily), dtype=bool)

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
    shp = sharpe_annual(daily, ft)

    return {
        "code": code, "name": NAMES.get(code, code), "window": window,
        "principal_annual": meta["principal_annual"], "xirr": meta["xirr"],
        "buys": meta["buys"], "sells": meta["sells"],
        "max_drawdown": round(mdd, 4),
        "avg_occupied": round(avg_occupied, 0),
        "occupancy": round(occupancy, 4),
        "efficiency": round(efficiency, 4) if efficiency is not None else None,
        "sharpe": round(shp, 4) if shp is not None else None,
    }


def aggregate(results):
    ann = [r["principal_annual"] for r in results]
    mdd = [r["max_drawdown"] for r in results]
    occ = [r["occupancy"] for r in results]
    eff = [r["efficiency"] for r in results if r["efficiency"] is not None]
    shp = [r["sharpe"] for r in results if r["sharpe"] is not None]
    return {
        "annual_mean": round(sum(ann) / len(ann), 4),
        "annual_median": round(sorted(ann)[len(ann) // 2], 4),
        "mdd_mean": round(sum(mdd) / len(mdd), 4),
        "occ_mean": round(sum(occ) / len(occ), 4),
        "eff_mean": round(sum(eff) / len(eff), 4) if eff else None,
        "sharpe_mean": round(sum(shp) / len(shp), 4) if shp else None,
        "sharpe_median": round(sorted(shp)[len(shp) // 2], 4) if shp else None,
    }


def main():
    print("赫斯特指数 卖出侧优化 (买入侧固定 打折0.5/加码1.0, 30万固定口径)\n")

    base_results = [run_one(code, None, use_hurst=False) for code in CODES]
    base_test = aggregate([r for r in base_results if r["code"] not in TRAIN_CODES])

    combos = []
    for sell_up in SELL_UP_GRID:
        results = [run_one(code, sell_up) for code in CODES]
        train = [r for r in results if r["code"] in TRAIN_CODES]
        test = [r for r in results if r["code"] not in TRAIN_CODES]
        combos.append({"sell_up": sell_up, "train": aggregate(train), "test": aggregate(test), "results": results})

    def fmt(v):
        return f"{v*100:6.2f}%" if v is not None else "  n/a  "

    print(f"{'sell_up':>8} | {'年化中位':>8} | {'回撤均值':>8} | {'夏普(均值/中位)':>16} | {'占用率':>8} | {'效率均值':>8}")
    print("-" * 92)
    for c in combos:
        t = c["test"]
        shp = f"{t['sharpe_mean']:.3f}/{t['sharpe_median']:.3f}" if t['sharpe_mean'] is not None else " n/a"
        print(f"{c['sell_up']:>8} | {t['annual_median']*100:6.2f}% | {t['mdd_mean']*100:6.2f}% | "
              f"{shp:>16} | {t['occ_mean']*100:6.2f}% | {fmt(t['eff_mean'])}")
    print("-" * 92)
    bshp = f"{base_test['sharpe_mean']:.3f}/{base_test['sharpe_median']:.3f}" if base_test['sharpe_mean'] is not None else " n/a"
    print(f"基线(无任何调制) 测试: 年化中位 {base_test['annual_median']*100:.2f}% | 回撤 {base_test['mdd_mean']*100:.2f}% | "
          f"夏普 {bshp} | 占用 {base_test['occ_mean']*100:.2f}% | 效率 {fmt(base_test['eff_mean'])}")

    # 逐指数明细: sell_up=0.0 (完全不抑制卖) vs 固定配置(sell_up=1.0)
    fixed = next(c for c in combos if c["sell_up"] == 1.0)
    pick = next(c for c in combos if c["sell_up"] == 0.0)
    fixed_by = {r["code"]: r for r in fixed["results"]}
    print("\n=== 明细: 固定买入侧(sell_up=1.0) vs 卖出侧完全抑制(sell_up=0.0) ===")
    print(f"{'代码':8} {'名称':10} | {'年化(固/0.0)':>18} | {'回撤(固/0.0)':>18} | {'夏普(固/0.0)':>18} | {'卖(固/0.0)':>12}")
    for r in pick["results"]:
        b = fixed_by[r["code"]]
        bs = f"{b['sharpe']:.2f}" if b['sharpe'] is not None else "n/a"
        rs = f"{r['sharpe']:.2f}" if r['sharpe'] is not None else "n/a"
        print(f"{r['code']:8} {r['name']:10} | {b['principal_annual']*100:6.2f}%/{r['principal_annual']*100:6.2f}% | "
              f"{b['max_drawdown']*100:6.2f}%/{r['max_drawdown']*100:6.2f}% | {bs:>8}/{rs:>8} | {b['sells']:>4}/{r['sells']:>4}")

    out = {
        "title": "赫斯特指数 卖出侧优化",
        "note": "买入侧固定: H>0.5跌->×0.5, 涨->×1.0; 卖出侧: H>0.5涨->卖出比例×sell_up; L=250/104; 完整价格历史",
        "fix": {"hurst_discount": FIX_DISCOUNT, "hurst_boost": FIX_BOOST},
        "sell_up_grid": SELL_UP_GRID,
        "window_rule": "window>=10 -> 250周, window<=5 -> 104周",
        "params": BALANCED_PARAMS, "buy_mults": list(BALANCED_MULTS), "base_amount": BASE,
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "metrics": {
            "max_drawdown": "账户净值最大回撤比例",
            "sharpe": "年化夏普(周收益, ×√52)",
            "occupancy": "资金占用率=max(0,净占用)/30万",
            "efficiency": "资金使用效率=期末净收益/(avg_occupied×年数)",
        },
        "baseline": {"test": base_test, "results": base_results},
        "combos": combos,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "hurst_sell_sweep.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'hurst_sell_sweep.json'}")


if __name__ == "__main__":
    main()
