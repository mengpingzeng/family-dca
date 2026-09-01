#!/usr/bin/env python3
"""10年期夏普1.2 — 多指数分散化 + 股债混合 组合测算 (周度口径, 公募常用).

组合 = 各指数 v3 策略独立跑(各自30万池), 等权/逆波动率加权合并成组合净值。
周度夏普 ×√52, 超额减 1.3%/252。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import build_curve, prep_df
from wind_new_search.pbpe_v3 import PARAMS as V3_PARAMS, KW as V3_KW
from wind_new_search.balanced_v2 import MULTS, BASE, COMMISSION_RATE, MIN_COMMISSION, THRESHOLD, CAP, POOL

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
RF = 0.013
RF_DAILY = RF / 252
NAMES = {"000300": "沪深300", "000905": "中证500", "000852": "中证1000", "000016": "上证50",
         "000688": "科创50", "399006": "创业板指", "399330": "深证100", "000015": "上证红利",
         "HSI": "恒生指数", "HSTECH": "恒生科技", "SPX500": "标普500", "NDX100": "纳斯达克100",
         "930930": "港股综合", "930931": "港股通50"}

_cache = {}


def series(code):
    if code not in _cache:
        df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        bt = build_curve(prep_df(df), V3_PARAMS, base_amount=BASE, commission_rate=COMMISSION_RATE,
                         min_commission=MIN_COMMISSION, lot_size=0,
                         principal_threshold=THRESHOLD, principal_cap=CAP, principal_pool=POOL,
                         buy_mults=MULTS, **V3_KW)
        d = bt["daily"]
        ft = pd.Timestamp(bt["meta"]["first_tradable"])
        s = pd.DataFrame([{"date": pd.Timestamp(x["date"]), "principal": x["principal"]} for x in d])
        s = s[s["date"] >= ft].set_index("date")["principal"]
        _cache[code] = s
    return _cache[code]


def weekly_sharpe(daily_ret_series):
    r = np.asarray(daily_ret_series, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return None
    s = r.std(ddof=0)
    if s <= 0:
        return None
    return float((r.mean() - RF_DAILY) / s * np.sqrt(52))


def annualize(nav, dates):
    years = (dates[-1] - dates[0]).days / 365.25
    return (nav[-1] / nav[0]) ** (1 / years) - 1, years


def portfolio(codes, weights):
    sers = {c: series(c) for c in codes}
    dates = sorted(set().union(*[set(s.index) for s in sers.values()]))
    df = pd.DataFrame({"date": [pd.Timestamp(x) for x in dates]})
    for c in codes:
        df[c] = df["date"].map(sers[c])
    df = df.set_index("date")
    df = df.interpolate(method="linear", limit_area="inside")
    valid = df.notna().all(axis=1)
    runs = (valid != valid.shift()).cumsum()
    best = runs[valid].value_counts().idxmax()
    seg = df[valid][runs[valid] == best]
    if weights is None:  # 逆波动率加权
        vols = seg.diff().div(seg.shift()).std(ddof=0)
        weights = (1.0 / vols).fillna(0.0).values
    w = np.asarray(weights) / np.sum(weights)
    nav = (seg.values @ w)
    return seg.index, nav


def report(name, codes, weights, add_rf_sleeve=0.0):
    idx, nav = portfolio(codes, weights)
    ann, years = annualize(nav, idx)
    if add_rf_sleeve > 0:
        # 股债混合: 将组合按比例拆出债券仓 (债券年化=RF), 再按原权重回填
        eq = 1 - add_rf_sleeve
        # 日收益混合
        r = np.diff(nav) / nav[:-1]
        r2 = eq * r + add_rf_sleeve * RF_DAILY
        nav2 = nav[0] * np.cumprod(1 + np.concatenate([[0], r2]))
        ann, years = annualize(nav2, idx)
        shp = weekly_sharpe(np.diff(nav2) / nav2[:-1])
    else:
        shp = weekly_sharpe(np.diff(nav) / nav[:-1])
    print(f"{name:<26} 区间{idx[0].date()}~{idx[-1].date()} ({years:5.1f}yr)  年化 {ann*100:6.2f}%  周度夏普 {shp:5.2f}")


if __name__ == "__main__":
    # 10年组: PE百分位起点最早的指数 (上证50/深证100/恒生 2012-2014, 沪深300/红利 2015)
    T10 = ["000300", "000016", "399330", "000015", "HSI"]                      # 公共 ~11年 (2015起)
    T10_500 = T10 + ["000905"]                                                  # +中证500 → ~9.5年
    A7 = ["000300", "000905", "000852", "000016", "399006", "399330", "000015"]  # A股7 (窗口受1000/创业板限制)
    print("周度超额夏普 (×√52, Rf=1.3%). 单指数 v3 周度夏普参考: 沪深300 0.39 / 中证1000 0.68 / 创业板 0.59")
    print("-" * 100)
    report("10年组(300/50/深100/红利/恒生)等权", T10, [1] * 5)
    report("10年组 + 中证500 (~9.5yr)等权", T10_500, [1] * 6)
    report("A股7等权 (窗口受限)", A7, [1] * 7)
    report("10年组逆波动率加权", T10, None)
    report("10年组等权 + 30%债券仓", T10, [1] * 5, add_rf_sleeve=0.30)
    report("10年组等权 + 50%债券仓", T10, [1] * 5, add_rf_sleeve=0.50)
    report("10年组等权 + 100%杠杆放大0.4倍", T10, [1] * 5, add_rf_sleeve=-0.0)
