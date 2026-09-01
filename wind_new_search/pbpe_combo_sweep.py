#!/usr/bin/env python3
"""PB/PE 双信号闸门扫描 — 买入更谨慎(PE也低) / 卖出更严格(PB也高) 对夏普的影响。

口径: 固定30万, 超额夏普(Rf=1.3%), 与 v2 页面一致。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import build_curve, prep_df, sharpe_annual

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"

BALANCED_PARAMS = {
    "buy_signal": "PB", "buy_gate": "FED", "buy_gate_cap": 0.55,
    "sell_signal": "PE", "sell_gate": None, "sell_gate_floor": None,
    "buy_floor": 0.10, "buy_low": 0.15, "buy_mid": 0.25, "buy_high": 0.70,
    "sell_heavy": 0.85, "sell_extreme": 0.95,
}
MULTS = (8, 4, 2, 0)
BASE = 1000
COMMISSION_RATE = 0.0005
MIN_COMMISSION = 5.0
THRESHOLD = 200_000
CAP = 300_000
POOL = 300_000
RF = 0.013

BROAD = ["000300", "000905", "000852", "000016", "000688", "399006", "399330"]
NAMES = {
    "000300": "沪深300", "000905": "中证500", "000015": "上证红利", "000016": "上证50",
    "000852": "中证1000", "399006": "创业板指", "399330": "深证100", "HSI": "恒生指数",
    "NDX100": "纳斯达克100", "SPX500": "标普500", "930931": "港股通50", "930930": "港股综合",
    "000688": "科创50", "HSTECH": "恒生科技",
}

_df_cache = {}


def load(code):
    if code not in _df_cache:
        df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        _df_cache[code] = prep_df(df)
    return _df_cache[code]


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


def run(code, params, kw):
    df = load(code)
    bt = build_curve(df, params, base_amount=BASE, commission_rate=COMMISSION_RATE,
                     min_commission=MIN_COMMISSION, lot_size=0,
                     principal_threshold=THRESHOLD, principal_cap=CAP, principal_pool=POOL,
                     buy_mults=MULTS, **kw)
    daily = bt["daily"]
    ft = bt["meta"].get("first_tradable")
    principals = [d["principal"] for d in daily if d.get("principal") is not None]
    return {
        "annual": bt["meta"]["principal_annual"],
        "sharpe": sharpe_annual(daily, ft, rf_annual=RF),
        "mdd": max_drawdown(principals),
        "buys": bt["meta"]["buys"],
    }


CONFIGS = []


def add(name, label, pmod, kw):
    CONFIGS.append((name, label, pmod, kw))


add("base", "基线", {}, {})
# 买入更谨慎: 要求 PE 也便宜 (PE%<=cap), 与 FED 闸门并列
for pe in (0.40, 0.45, 0.50):
    add(f"buy_pe{int(pe*100)}", f"买谨慎:PE%≤{pe:.0%}", {"buy_gate": ["FED", "PE"], "buy_gate_cap": [0.55, pe]}, {})
# 卖出更严格: 要求 PB 也贵 (PB%>=floor)
for pb in (0.70, 0.75, 0.80):
    add(f"sell_pb{int(pb*100)}", f"卖严格:PB%≥{pb:.0%}", {"sell_gate": ["PB"], "sell_gate_floor": [pb]}, {})
# 两者同时
for pe, pb in ((0.45, 0.70), (0.45, 0.75), (0.50, 0.75)):
    add(f"both_{int(pe*100)}_{int(pb*100)}", f"双闸门:买PE≤{pe:.0%}+卖PB≥{pb:.0%}",
        {"buy_gate": ["FED", "PE"], "buy_gate_cap": [0.55, pe],
         "sell_gate": ["PB"], "sell_gate_floor": [pb]}, {})
# 双闸门 + v2 趋势调制 (更早止盈0.80 + 20周均线β0.5 + γ1.5)
add("both_v2", "双闸门45/70 + v2趋势",
    {"sell_heavy": 0.80, "buy_gate": ["FED", "PE"], "buy_gate_cap": [0.55, 0.45],
     "sell_gate": ["PB"], "sell_gate_floor": [0.70]},
    {"ma_window": 20, "ma_below": 0.5, "ma_above": 1.5})


def main():
    results = {}
    for cname, label, pmod, kw in CONFIGS:
        params = {**BALANCED_PARAMS, **pmod}
        results[cname] = {code: run(code, params, kw) for code in BROAD}

    base = results["base"]
    print(f"{'配置':<24} {'年化均值':>8} {'夏普均值':>8} {'回撤均值':>8} {'年化Δ':>8} {'夏普Δ':>8} {'夏普胜':>6}")
    print("-" * 80)
    for cname, label, pmod, kw in CONFIGS:
        anns = [results[cname][c]["annual"] for c in BROAD]
        shps = [results[cname][c]["sharpe"] for c in BROAD]
        mdds = [results[cname][c]["mdd"] for c in BROAD]
        bann = np.mean([base[c]["annual"] for c in BROAD])
        bshp = np.mean([base[c]["sharpe"] for c in BROAD])
        win = sum(1 for c in BROAD if results[cname][c]["sharpe"] > base[c]["sharpe"])
        print(f"{label:<24} {np.mean(anns)*100:7.2f}% {np.mean(shps):8.3f} {np.mean(mdds)*100:7.1f}% "
              f"{(np.mean(anns)/bann-1)*100:7.1f}% {(np.mean(shps)/bshp-1)*100:7.1f}% {win:>3}/{len(BROAD)}")

    print("\n=== 逐宽基 夏普 ===")
    for code in BROAD:
        parts = [f"{NAMES[code]}:基={base[code]['sharpe']:.2f}"]
        for cname, label, pmod, kw in CONFIGS:
            if cname == "base":
                continue
            parts.append(f"{label[:5]}={results[cname][code]['sharpe']:.2f}")
        print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
