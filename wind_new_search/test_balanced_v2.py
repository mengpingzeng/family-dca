#!/usr/bin/env python3
"""均衡策略 v2 测试集验证 + 基线对比 — 生成 test_balanced_v2.json.

对比基线(均衡策略)与 v2 (更早止盈0.80 + 20周均线软制动β0.5 + 顺势加码1.5),
口径与均衡策略一致: 固定30万, 阈值20万收缩, 封顶30万, 费用万5低消5。
同时输出 ETF 口径 (后复权, 整手100, 佣金万5低消5)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import build_curve, prep_df, sharpe_annual
from wind_new_search.balanced_v2 import (
    PARAMS as V2_PARAMS, KW as V2_KW, MULTS, BASE, COMMISSION_RATE,
    MIN_COMMISSION, THRESHOLD, CAP, POOL, ETF_MAP,
)
from wind_new_search.test_balanced import BALANCED_PARAMS, BALANCED_MULTS, NAMES

RF_ANNUAL = 0.013  # 无风险利率 年化 1.3%

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
ETF_DIR = PROJECT_DIR / "data-store" / "parquet" / "etf"
OUTPUT_DIR = PROJECT_DIR / "wind_new_search" / "output"

ALL_CODES = ["000300", "000905", "000015", "000016", "000852", "399006", "399330",
             "HSI", "HSTECH", "SPX500", "NDX100", "930931", "930930", "000688"]
BROAD = ["000300", "000905", "000852", "000016", "000688", "399006", "399330"]


def max_drawdown(series):
    peak, mdd = float("-inf"), 0.0
    for v in series:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > mdd:
                mdd = dd
    return mdd


def _sharpe(daily, ft):
    return sharpe_annual(daily, ft, rf_annual=RF_ANNUAL)


def occupancy_eff(daily, ft, pool):
    dts = pd.to_datetime([d["date"] for d in daily])
    mask = np.asarray(dts >= pd.Timestamp(ft)) if ft else np.ones(len(daily), dtype=bool)
    occs = [max(0.0, d["cum_invested"] - d["cash"]) for d, m in zip(daily, mask) if m]
    if not occs:
        return 0.0, None
    avg_occ = sum(occs) / len(occs)
    first = dts[mask].min()
    years = (dts[mask].max() - first).days / 365.25 if mask.any() else 0.0
    net = daily[-1].get("principal", 0) - pool if daily else 0
    eff = net / (avg_occ * years) if avg_occ > 0 and years > 0 else None
    return avg_occ / pool, eff


def run_curve(code, params, kw):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = prep_df(df)
    bt = build_curve(df, params, base_amount=BASE, commission_rate=COMMISSION_RATE,
                     min_commission=MIN_COMMISSION, lot_size=0,
                     principal_threshold=THRESHOLD, principal_cap=CAP, principal_pool=POOL,
                     buy_mults=MULTS, **kw)
    daily = bt["daily"]
    ft = bt["meta"].get("first_tradable")
    principals = [d["principal"] for d in daily if d.get("principal") is not None]
    occ, eff = occupancy_eff(daily, ft, POOL)
    return {
        "principal_annual": bt["meta"]["principal_annual"],
        "xirr": bt["meta"]["xirr"],
        "sharpe": _sharpe(daily, ft),
        "max_drawdown": max_drawdown(principals),
        "occupancy": occ,
        "efficiency": eff,
        "buys": bt["meta"]["buys"], "sells": bt["meta"]["sells"],
        "principal_final": bt["meta"]["principal_final"],
    }


def run_etf_curve(idx_code, params, kw):
    etf_code, etf_name = ETF_MAP[idx_code]
    df = pd.read_parquet(MERGED_DIR / f"{idx_code}.parquet")
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    etf = pd.read_parquet(ETF_DIR / f"{etf_code}.parquet")
    etf["date"] = pd.to_datetime(etf["date"]).astype("datetime64[ns]")
    aligned = pd.merge_asof(df[["date"]], etf[["date", "hfq"]], on="date", direction="backward")
    mask = aligned["hfq"].notna() & df["pe_pct"].notna()
    df_etf = df[mask].reset_index(drop=True)
    exec_price = aligned.loc[mask, "hfq"].to_numpy(float)
    df_etf = prep_df(df_etf)
    bt = build_curve(df_etf, params, base_amount=BASE, exec_price=exec_price,
                     commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                     lot_size=100, principal_threshold=THRESHOLD, principal_cap=CAP,
                     principal_pool=POOL, buy_mults=MULTS, **kw)
    daily = bt["daily"]
    ft = bt["meta"].get("first_tradable")
    principals = [d["principal"] for d in daily if d.get("principal") is not None]
    occ, eff = occupancy_eff(daily, ft, POOL)
    return {
        "etf_code": etf_code, "etf_name": etf_name,
        "principal_annual": bt["meta"]["principal_annual"],
        "xirr": bt["meta"]["xirr"],
        "sharpe": _sharpe(daily, ft),
        "max_drawdown": max_drawdown(principals),
        "occupancy": occ,
        "efficiency": eff,
        "buys": bt["meta"]["buys"], "sells": bt["meta"]["sells"],
        "principal_final": bt["meta"]["principal_final"],
    }


def agg(codes, key, d):
    vals = [d[c][key] for c in codes if d[c].get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    results = {}
    for code in ALL_CODES:
        base = run_curve(code, BALANCED_PARAMS, {})
        v2 = run_curve(code, V2_PARAMS, V2_KW)
        results[code] = {
            "code": code, "name": NAMES.get(code, code),
            "base": base, "v2": v2,
            "etf_base": None, "etf_v2": None,
        }
        if code in ETF_MAP:
            try:
                results[code]["etf_base"] = run_etf_curve(code, BALANCED_PARAMS, {})
                results[code]["etf_v2"] = run_etf_curve(code, V2_PARAMS, V2_KW)
            except Exception as e:
                print("ETF ERR", code, e)

    print(f"{'指数':<10} | {'基线 年化/夏普/回撤':<22} | {'v2 年化/夏普/回撤':<22} | 夏普Δ | 年化Δ")
    print("-" * 88)
    for code in ALL_CODES:
        b, v = results[code]["base"], results[code]["v2"]
        bshp = b["sharpe"] or 0
        vshp = v["sharpe"] or 0
        ad = (v['principal_annual'] / b['principal_annual'] - 1) * 100 if b['principal_annual'] and b['principal_annual'] != 0 else 0.0
        print(f"{NAMES.get(code,code):<10} | {b['principal_annual']*100:5.1f}%/{bshp:5.2f}/{b['max_drawdown']*100:5.1f}% | "
              f"{v['principal_annual']*100:5.1f}%/{vshp:5.2f}/{v['max_drawdown']*100:5.1f}% | {vshp-bshp:+5.2f} | {ad:+6.1f}%")

    # 宽基汇总
    summary = {}
    for mode in ("base", "v2"):
        summary[mode] = {
            "annual_mean": agg(BROAD, "principal_annual", {c: results[c][mode] for c in ALL_CODES}),
            "sharpe_mean": agg(BROAD, "sharpe", {c: results[c][mode] for c in ALL_CODES}),
            "mdd_mean": agg(BROAD, "max_drawdown", {c: results[c][mode] for c in ALL_CODES}),
            "occupancy_mean": agg(BROAD, "occupancy", {c: results[c][mode] for c in ALL_CODES}),
            "efficiency_mean": agg(BROAD, "efficiency", {c: results[c][mode] for c in ALL_CODES}),
        }
    win = sum(1 for c in BROAD if (results[c]["v2"]["sharpe"] or 0) > (results[c]["base"]["sharpe"] or 0))
    print("\n=== 宽基7个 平均 ===")
    for mode in ("base", "v2"):
        s = summary[mode]
        print(f"{mode:5}: 年化 {s['annual_mean']*100:.2f}%  夏普 {s['sharpe_mean']:.3f}  回撤 {s['mdd_mean']*100:.1f}%  占用 {s['occupancy_mean']*100:.1f}%  效率 {s['efficiency_mean']:.3f}")
    print(f"夏普跑赢基线: {win}/{len(BROAD)}")

    out = {
        "title": "均衡策略 v2 vs 基线 (固定30万口径)",
        "note": "v2 = 更早止盈(卖PE 0.80) + 20周均线软制动(β0.5) + 顺势加码(γ1.5)",
        "base_params": BALANCED_PARAMS,
        "v2_params": V2_PARAMS,
        "v2_kw": {k: v for k, v in V2_KW.items()},
        "cost_model": {"commission_rate": COMMISSION_RATE, "min_commission": MIN_COMMISSION},
        "principal": {"threshold": THRESHOLD, "cap": CAP, "pool": POOL, "base": BASE, "mults": list(MULTS)},
        "risk_free_rate": RF_ANNUAL, "sharpe_note": "超额夏普: (日收益均值-日无风险利率)/日收益标准差×√52, 日无风险利率=年化1.3%/252",
        "broad": BROAD, "summary": summary, "win_sharpe": win, "n_broad": len(BROAD),
        "results": [results[c] for c in ALL_CODES],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "test_balanced_v2.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'test_balanced_v2.json'}")


if __name__ == "__main__":
    main()
