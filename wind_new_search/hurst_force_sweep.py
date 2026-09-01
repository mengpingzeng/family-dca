#!/usr/bin/env python3
"""
赫斯特指数「信心指数」二值触发 + 力度扫描.

设计 (H 阈值固定 0.5, 力度直接调):
  方向 d = sign(当前价 - L周前价),  L 按指数自适应: window>=10 -> 250周, window<=5 -> 104周
  H = 单段 R/S 法, 用完整价格历史逐期滚动计算 (含可交易期之前的历史)
  if H > 0.5 (趋势持续):
      涨 -> 买入倍数 × boost   (加码力度)
      跌 -> 买入倍数 × discount (打折力度)
  else: 不变

扫描: discount ∈ {1.0, 0.5, 0.25} × boost ∈ {1.0, 1.5, 2.0} + 基线.

指标 (含夏普比率): 固定年化 / 最大回撤 / 夏普 / 资金占用率 / 资金使用效率 / 买/卖次数.

数据保持独立: 不改动原始数据与既有输出, 结果落盘 output/hurst_force_sweep.json.
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

DISCOUNT_GRID = [1.0, 0.5, 0.25]
BOOST_GRID = [1.0, 1.5, 2.0]


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


def run_one(code, discount, boost):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    window = int(df["window"].iloc[0])
    df = prep_df(df)
    kw = dict(base_amount=BASE, commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
              lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP,
              principal_pool=POOL, buy_mults=BALANCED_MULTS)
    if discount is None and boost is None:
        bt = build_curve(df, BALANCED_PARAMS, **kw)
    else:
        bt = build_curve(df, BALANCED_PARAMS, hurst_window=window_of(code),
                         hurst_discount=discount, hurst_boost=boost, **kw)
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

    return {
        "code": code, "name": NAMES.get(code, code), "window": window,
        "principal_annual": meta["principal_annual"], "xirr": meta["xirr"],
        "buys": meta["buys"], "sells": meta["sells"],
        "max_drawdown": round(mdd, 4),
        "avg_occupied": round(avg_occupied, 0),
        "occupancy": round(occupancy, 4),
        "efficiency": round(efficiency, 4) if efficiency is not None else None,
        "sharpe": round(sharpe_annual(daily, ft), 4) if sharpe_annual(daily, ft) is not None else None,
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
    print("赫斯特指数 二值触发 + 力度扫描 (H固定0.5, 长指数250周/短指数104周, 30万固定口径)\n")

    base_results = [run_one(code, None, None) for code in CODES]
    base_test = aggregate([r for r in base_results if r["code"] not in TRAIN_CODES])
    base_train = aggregate([r for r in base_results if r["code"] in TRAIN_CODES])

    combos = []
    for discount in DISCOUNT_GRID:
        for boost in BOOST_GRID:
            results = [run_one(code, discount, boost) for code in CODES]
            train = [r for r in results if r["code"] in TRAIN_CODES]
            test = [r for r in results if r["code"] not in TRAIN_CODES]
            combos.append({"discount": discount, "boost": boost,
                           "train": aggregate(train), "test": aggregate(test), "results": results})

    def fmt(v):
        return f"{v*100:6.2f}%" if v is not None else "  n/a  "

    print(f"{'disc':>5} {'boost':>5} | {'年化中位':>8} | {'回撤均值':>8} | {'夏普(均值/中位)':>16} | {'占用率':>8} | {'效率均值':>8} | 训练夏普中位")
    print("-" * 104)
    for c in combos:
        t = c["test"]; tr = c["train"]
        shp = f"{t['sharpe_mean']:.3f}/{t['sharpe_median']:.3f}" if t['sharpe_mean'] is not None else " n/a"
        trshp = f"{tr['sharpe_median']:.3f}" if tr['sharpe_median'] is not None else "n/a"
        print(f"{c['discount']:>5} {c['boost']:>5} | {t['annual_median']*100:6.2f}% | {t['mdd_mean']*100:6.2f}% | "
              f"{shp:>16} | {t['occ_mean']*100:6.2f}% | {fmt(t['eff_mean'])} | {trshp}")
    print("-" * 104)
    bshp = f"{base_test['sharpe_mean']:.3f}/{base_test['sharpe_median']:.3f}" if base_test['sharpe_mean'] is not None else " n/a"
    print(f"基线(无调制) 测试: 年化中位 {base_test['annual_median']*100:.2f}% | 回撤 {base_test['mdd_mean']*100:.2f}% | "
          f"夏普 {bshp} | 占用 {base_test['occ_mean']*100:.2f}% | 效率 {fmt(base_test['eff_mean'])}")

    # 逐指数明细: discount=0.5, boost=1.5
    pick = next(c for c in combos if c["discount"] == 0.5 and c["boost"] == 1.5)
    base_by_code = {r["code"]: r for r in base_results}
    print("\n=== 明细: 打折0.5/加码1.5 vs 基线 (年化 / 回撤 / 夏普 / 买) ===")
    print(f"{'代码':8} {'名称':10} | {'年化(基/新)':>18} | {'回撤(基/新)':>18} | {'夏普(基/新)':>18} | {'买(基/新)':>12}")
    for r in pick["results"]:
        b = base_by_code[r["code"]]
        bs = f"{b['sharpe']:.2f}" if b['sharpe'] is not None else "n/a"
        rs = f"{r['sharpe']:.2f}" if r['sharpe'] is not None else "n/a"
        print(f"{r['code']:8} {r['name']:10} | {b['principal_annual']*100:6.2f}%/{r['principal_annual']*100:6.2f}% | "
              f"{b['max_drawdown']*100:6.2f}%/{r['max_drawdown']*100:6.2f}% | {bs:>8}/{rs:>8} | {b['buys']:>4}/{r['buys']:>4}")

    out = {
        "title": "赫斯特指数信心指数 二值触发+力度扫描",
        "note": "H阈值固定0.5; 方向d=sign(价-L周前价); H>0.5且跌->×discount, 涨->×boost; H<=0.5不变; L=250(长指数)/104(短指数); 用完整价格历史",
        "discount_grid": DISCOUNT_GRID, "boost_grid": BOOST_GRID,
        "window_rule": "window>=10 -> 250周, window<=5 -> 104周",
        "params": BALANCED_PARAMS, "buy_mults": list(BALANCED_MULTS), "base_amount": BASE,
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "metrics": {
            "max_drawdown": "账户净值最大回撤比例",
            "sharpe": "年化夏普(周收益, ×√52)",
            "avg_occupied": "时间加权平均净占用本金(元), max(0, 净占用)",
            "occupancy": "资金占用率=avg_occupied/30万",
            "efficiency": "资金使用效率=期末净收益/(avg_occupied×年数)",
        },
        "baseline": {"train": base_train, "test": base_test, "results": base_results},
        "combos": combos,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "hurst_force_sweep.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'hurst_force_sweep.json'}")


if __name__ == "__main__":
    main()
