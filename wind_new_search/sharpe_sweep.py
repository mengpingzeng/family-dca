#!/usr/bin/env python3
"""夏普提升 - 系统化网格扫描 + 排行榜。

目标: 固定30万口径下, 提升宽基平均夏普, 同时年化收益不低于基线的 ~85%。
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

BROAD = ["000300", "000905", "000852", "000016", "000688", "399006", "399330"]
ALL = BROAD
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


RF_ANNUAL = 0.013  # 无风险利率 年化 1.3%


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
        "sharpe": sharpe_annual(daily, ft, rf_annual=RF_ANNUAL),
        "mdd": max_drawdown(principals),
    }


CONFIGS = [
    ("base", "基线", {}, {}),
    ("ma20_b50", "20周均线β0.5", {}, {"ma_window": 20, "ma_below": 0.5, "ma_above": 1.0}),
    ("ma20_b50_g15", "β0.5+加码1.5", {}, {"ma_window": 20, "ma_below": 0.5, "ma_above": 1.5}),
    ("ma40_b50", "40周均线β0.5", {}, {"ma_window": 40, "ma_below": 0.5, "ma_above": 1.0}),
    ("sell80", "卖heavy0.80", {"sell_heavy": 0.80}, {}),
    ("sell80_ma20", "卖0.80+β0.5", {"sell_heavy": 0.80}, {"ma_window": 20, "ma_below": 0.5, "ma_above": 1.0}),
    ("sell80_ma20_g15", "卖0.80+β0.5+加码1.5", {"sell_heavy": 0.80}, {"ma_window": 20, "ma_below": 0.5, "ma_above": 1.5}),
    ("trail30h", "回撤30%减半仓", {}, {"trail_stop": 0.30, "trail_stop_ratio": 0.5}),
    ("trail30h_ma20", "回撤30%减半+β0.5", {}, {"trail_stop": 0.30, "trail_stop_ratio": 0.5, "ma_window": 20, "ma_below": 0.5, "ma_above": 1.0}),
    ("hurst_ma20", "赫斯特+β0.5", {}, {"hurst_window": 20, "hurst_discount": 0.5, "hurst_boost": 1.0, "ma_window": 20, "ma_below": 0.5, "ma_above": 1.0}),
    ("trail40h_ma20", "回撤40%减半+β0.5", {}, {"trail_stop": 0.40, "trail_stop_ratio": 0.5, "ma_window": 20, "ma_below": 0.5, "ma_above": 1.0}),
    ("voltarget", "波动率目标15%", {}, {"vol_window": 40, "vol_target": 0.15}),
    ("voltarget_ma20", "波动目标15+β0.5", {}, {"vol_window": 40, "vol_target": 0.15, "ma_window": 20, "ma_below": 0.5, "ma_above": 1.0}),
    ("sell80_vol", "卖0.80+波动目标", {"sell_heavy": 0.80}, {"vol_window": 40, "vol_target": 0.15}),
    ("eqstop", "净值回撤25%减半", {}, {"equity_stop": 0.25, "equity_stop_ratio": 0.5}),
    ("sell80_ma20_vol", "卖0.80+β0.5+波动目标", {"sell_heavy": 0.80}, {"vol_window": 40, "vol_target": 0.15, "ma_window": 20, "ma_below": 0.5, "ma_above": 1.0}),
]


def main():
    results = {}
    for cname, label, pmod, kw in CONFIGS:
        params = {**BALANCED_PARAMS, **pmod}
        results[cname] = {}
        for code in ALL:
            results[cname][code] = run(code, params, kw)

    base = results["base"]
    leaderboard = []
    for cname, label, pmod, kw in CONFIGS:
        anns = [results[cname][c]["annual"] for c in BROAD]
        shps = [results[cname][c]["sharpe"] for c in BROAD]
        mdds = [results[cname][c]["mdd"] for c in BROAD]
        bann = [base[c]["annual"] for c in BROAD]
        bshp = [base[c]["sharpe"] for c in BROAD]
        ann_mean = np.mean(anns); shp_mean = np.mean(shps); mdd_mean = np.mean(mdds)
        ann_delta = ann_mean / np.mean(bann) - 1
        shp_delta = shp_mean / np.mean(bshp) - 1
        win = sum(1 for c in BROAD if results[cname][c]["sharpe"] > base[c]["sharpe"])
        leaderboard.append((cname, label, ann_mean, shp_mean, mdd_mean, ann_delta, shp_delta, win))

    print(f"{'配置':<24} {'年化均值':>8} {'夏普均值':>8} {'回撤均值':>8} {'年化变化':>8} {'夏普变化':>8} {'夏普胜':>6}")
    print("-" * 78)
    for cname, label, ann, shp, mdd, ad, sd, win in leaderboard:
        print(f"{label:<24} {ann*100:7.2f}% {shp:8.3f} {mdd*100:7.1f}% {ad*100:7.1f}% {sd*100:7.1f}% {win:>3}/{len(BROAD)}")

    print("\n=== 各配置 逐宽基夏普 (格式: 年化%/夏普) ===")
    for code in BROAD:
        line = f"{NAMES.get(code,code):<8}"
        for cname, label, pmod, kw in CONFIGS:
            r = results[cname][code]
            line += f" | {label[:6]}:{r['annual']*100:.1f}%/{r['sharpe']:.2f}"
        print(line)


if __name__ == "__main__":
    main()
