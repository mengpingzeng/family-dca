#!/usr/bin/env python3
"""
赫斯特「方向 × 强度百分位」二维调制 对比.

配置 (方向轴 × 强度百分位轴):
  baseline : 无调制
  dir_only : 方向折扣 (H>0.5且跌 -> ×0.5, 涨 -> ×1.0)   [上一轮已固化]
  pct_only : 只按 H 百分位调制 (方向无关)
  dir_pct  : 方向 × 百分位 二维 (本方案)

H 百分位分带 (阈值 40/60/80):
  pct>80% -> ×0.7 (趋势很强, 谨慎)
  60-80%  -> ×0.85 (趋势较强)
  40-60%  -> ×1.0 (中性)
  <40%    -> ×1.15 (趋势最弱, 积极)

窗口: H 窗口 L=250(长)/104(短); 百分位回看 200(长)/52(短).

指标: 固定年化 / 最大回撤 / 夏普 / 资金占用率 / 资金使用效率.

数据保持独立: 结果落盘 output/hurst_pct_sweep.json.
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

CONFIGS = ["baseline", "dir_only", "pct_only", "dir_pct"]


def window_of(code):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet", columns=["window"])
    w = int(df["window"].iloc[0])
    return 250 if w >= 10 else 104


def pct_window_of(code):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet", columns=["window"])
    w = int(df["window"].iloc[0])
    return 200 if w >= 10 else 52


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


def run_one(code, cfg):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    window = int(df["window"].iloc[0])
    df = prep_df(df)
    kw = dict(base_amount=BASE, commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
              lot_size=0, principal_threshold=THRESHOLD, principal_cap=CAP,
              principal_pool=POOL, buy_mults=BALANCED_MULTS)
    if cfg == "baseline":
        bt = build_curve(df, BALANCED_PARAMS, **kw)
    elif cfg == "dir_only":
        bt = build_curve(df, BALANCED_PARAMS, hurst_window=window_of(code),
                         hurst_discount=0.5, hurst_boost=1.0, **kw)
    elif cfg == "pct_only":
        bt = build_curve(df, BALANCED_PARAMS, hurst_window=window_of(code),
                         hurst_pct_window=pct_window_of(code), **kw)
    elif cfg == "dir_pct":
        bt = build_curve(df, BALANCED_PARAMS, hurst_window=window_of(code),
                         hurst_discount=0.5, hurst_boost=1.0,
                         hurst_pct_window=pct_window_of(code), **kw)
    else:
        raise ValueError(cfg)
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
    print("赫斯特 方向×强度百分位 二维调制对比 (30万固定口径)\n")
    results = {}
    for cfg in CONFIGS:
        rows = [run_one(code, cfg) for code in CODES]
        results[cfg] = rows

    def fmt(v):
        return f"{v*100:6.2f}%" if v is not None else "  n/a  "

    print(f"{'配置':12} | {'年化中位':>8} | {'回撤均值':>8} | {'夏普(均值/中位)':>16} | {'占用率':>8} | {'效率均值':>8}")
    print("-" * 92)
    for cfg in CONFIGS:
        test = [r for r in results[cfg] if r["code"] not in TRAIN_CODES]
        train = [r for r in results[cfg] if r["code"] in TRAIN_CODES]
        t = aggregate(test); tr = aggregate(train)
        shp = f"{t['sharpe_mean']:.3f}/{t['sharpe_median']:.3f}" if t['sharpe_mean'] is not None else " n/a"
        print(f"{cfg:12} | {t['annual_median']*100:6.2f}% | {t['mdd_mean']*100:6.2f}% | {shp:>16} | {t['occ_mean']*100:6.2f}% | {fmt(t['eff_mean'])}")
        results[cfg + "_test"] = t
        results[cfg + "_train"] = tr

    # 逐指数明细: baseline / dir_only / dir_pct
    base_by = {r["code"]: r for r in results["baseline"]}
    dir_by = {r["code"]: r for r in results["dir_only"]}
    pct_by = {r["code"]: r for r in results["dir_pct"]}
    print("\n=== 明细: 基线 / 方向 / 方向+百分位 (年化 / 回撤 / 夏普) ===")
    def fshp(v):
        return f"{v:.2f}" if v is not None else "n/a"
    print(f"{'代码':8} {'名称':10} | {'年化(基/向/向+分)':>24} | {'回撤(基/向/向+分)':>26} | {'夏普(基/向/向+分)':>26}")
    for code in CODES:
        b, d_, p_ = base_by[code], dir_by[code], pct_by[code]
        print(f"{code:8} {p_['name']:10} | "
              f"{b['principal_annual']*100:6.2f}/{d_['principal_annual']*100:6.2f}/{p_['principal_annual']*100:6.2f} | "
              f"{b['max_drawdown']*100:6.2f}/{d_['max_drawdown']*100:6.2f}/{p_['max_drawdown']*100:6.2f} | "
              f"{fshp(b['sharpe'])}/{fshp(d_['sharpe'])}/{fshp(p_['sharpe'])}")

    out = {
        "title": "赫斯特 方向×强度百分位 二维调制对比",
        "note": "方向轴: H>0.5且跌->×0.5/涨->×1.0; 百分位轴: pct>80->0.7, 60-80->0.85, 40-60->1.0, <40->1.15; L=250/104, 百分位回看200/52",
        "configs": CONFIGS,
        "params": BALANCED_PARAMS, "buy_mults": list(BALANCED_MULTS), "base_amount": BASE,
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "metrics": {
            "max_drawdown": "账户净值最大回撤比例", "sharpe": "年化夏普(周收益,×√52)",
            "occupancy": "资金占用率=max(0,净占用)/30万", "efficiency": "资金使用效率=净收益/(avg_occupied×年数)",
        },
        "results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "hurst_pct_sweep.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'hurst_pct_sweep.json'}")


if __name__ == "__main__":
    main()
