#!/usr/bin/env python3
"""PB/PE 双闸门 变体分析 — 单独/组合效果汇总 (宽基7个平均, 固定30万, 超额夏普 Rf=1.3%).

变体:
  buy_cautious : 仅买入更谨慎 (PB买 + 需 PE%≤45%)
  sell_strict  : 仅卖出更严格 (卖 PE + 需 PB%≥70%)
  both         : 双闸门同时满足
  输出到 output/pbpe_variant_summary.json, 供 v3 页面展示分析。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import build_curve, prep_df, sharpe_annual
from wind_new_search.test_balanced import BALANCED_PARAMS
from wind_new_search.balanced_v2 import MULTS, BASE, COMMISSION_RATE, MIN_COMMISSION, THRESHOLD, CAP, POOL

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
OUTPUT_DIR = PROJECT_DIR / "wind_new_search" / "output"
RF = 0.013
BROAD = ["000300", "000905", "000852", "000016", "000688", "399006", "399330"]
NAMES = {
    "000300": "沪深300", "000905": "中证500", "000852": "中证1000", "000016": "上证50",
    "000688": "科创50", "399006": "创业板指", "399330": "深证100",
}

_cache = {}


def load(code):
    if code not in _cache:
        df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        _cache[code] = prep_df(df)
    return _cache[code]


def metrics(code, params, kw):
    bt = build_curve(load(code), params, base_amount=BASE, commission_rate=COMMISSION_RATE,
                     min_commission=MIN_COMMISSION, lot_size=0,
                     principal_threshold=THRESHOLD, principal_cap=CAP, principal_pool=POOL,
                     buy_mults=MULTS, **kw)
    daily = bt["daily"]
    ft = bt["meta"].get("first_tradable")
    pr = [d["principal"] for d in daily if d.get("principal") is not None]
    peak, mdd = float("-inf"), 0.0
    for v in pr:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return {"annual": bt["meta"]["principal_annual"],
            "sharpe": sharpe_annual(daily, ft, rf_annual=RF), "mdd": mdd}


def variants():
    base = BALANCED_PARAMS
    buy = {**base, "buy_gate": ["FED", "PE"], "buy_gate_cap": [0.55, 0.45]}
    sell = {**base, "sell_gate": ["PB"], "sell_gate_floor": [0.70]}
    both = {**buy, "sell_gate": ["PB"], "sell_gate_floor": [0.70]}
    v3 = {**both, "sell_heavy": 0.80}
    kw_v3 = {"ma_window": 20, "ma_below": 0.5, "ma_above": 1.5}

    def agg(params, kw):
        rows = [metrics(c, params, kw) for c in BROAD]
        return {
            "annual_mean": round(float(np.mean([r["annual"] for r in rows])), 4),
            "sharpe_mean": round(float(np.mean([r["sharpe"] for r in rows])), 4),
            "mdd_mean": round(float(np.mean([r["mdd"] for r in rows])), 4),
        }

    out = {
        "note": "宽基7个平均 · 固定30万 · 超额夏普(Rf=1.3%)",
        "base": agg(base, {}),
        "buy_cautious": agg(buy, {}),
        "sell_strict": agg(sell, {}),
        "both": agg(both, {}),
        "v3": agg(v3, kw_v3),
        "labels": {
            "base": "基线(均衡策略)",
            "buy_cautious": "仅买入更谨慎 (需PE%≤45%)",
            "sell_strict": "仅卖出更严格 (需PB%≥70%)",
            "both": "双闸门同时 (买PE≤45% + 卖PB≥70%)",
            "v3": "双闸门 + v2趋势 (推荐 v3)",
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "pbpe_variant_summary.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for k in ("base", "buy_cautious", "sell_strict", "both", "v3"):
        r = out[k]
        print(f"{out['labels'][k]:<28} 年化 {r['annual_mean']*100:6.2f}%  夏普 {r['sharpe_mean']:6.3f}  回撤 {r['mdd_mean']*100:5.1f}%")
    print("保存:", OUTPUT_DIR / "pbpe_variant_summary.json")


if __name__ == "__main__":
    variants()
