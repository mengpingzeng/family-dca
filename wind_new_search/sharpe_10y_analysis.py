#!/usr/bin/env python3
"""10年期夏普 1.2 可行性分析。

三种夏普口径 + 多指数组合效果:
  S52_daily  : 日收益 ×√52 (现口径, 低估)
  S252_daily : 日收益 ×√252 (正确日度年化)
  S_weekly   : 周收益 ×√52 (正确周度年化, 公募常用口径)
组合: 7个宽基等权合并成 1 个组合池, 再算夏普 (分散化)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import build_curve, prep_df
from wind_new_search.test_balanced import BALANCED_PARAMS
from wind_new_search.pbpe_v3 import PARAMS as V3_PARAMS, KW as V3_KW
from wind_new_search.balanced_v2 import MULTS, BASE, COMMISSION_RATE, MIN_COMMISSION, THRESHOLD, CAP, POOL

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
RF = 0.013
BROAD = ["000300", "000905", "000852", "000016", "000688", "399006", "399330"]
NAMES = {"000300": "沪深300", "000905": "中证500", "000852": "中证1000", "000016": "上证50",
         "000688": "科创50", "399006": "创业板指", "399330": "深证100"}


def load(code):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return prep_df(df)


def curve_series(code, params, kw):
    bt = build_curve(load(code), params, base_amount=BASE, commission_rate=COMMISSION_RATE,
                     min_commission=MIN_COMMISSION, lot_size=0,
                     principal_threshold=THRESHOLD, principal_cap=CAP, principal_pool=POOL,
                     buy_mults=MULTS, **kw)
    d = bt["daily"]
    ft = pd.Timestamp(bt["meta"]["first_tradable"])
    s = pd.DataFrame([{"date": pd.Timestamp(x["date"]), "principal": x["principal"]} for x in d])
    s = s[s["date"] >= ft].reset_index(drop=True)
    return s


def sharpe_of(returns, rf_daily, ann):
    """returns: 收益率序列; ann: 年化因子. 超额 = mean - 日度rf*ann 换算."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return None
    s = r.std(ddof=0)
    if s <= 0:
        return None
    return float((r.mean() * ann - rf_daily) / (s * np.sqrt(ann)))


def analyze(code, params, kw, tag):
    s = curve_series(code, params, kw)
    pr = s["principal"].values
    daily = pr[1:] / pr[:-1] - 1
    weeks = s["date"].dt.to_period("W").values[1:]
    wdf = pd.DataFrame({"wk": weeks, "r": daily}).groupby("wk")["r"].apply(lambda x: (1 + x).prod() - 1).values
    rf_daily = RF / 252
    out = {
        "code": code, "name": NAMES[code], "tag": tag,
        "years": (s["date"].max() - s["date"].min()).days / 365.25,
        "annual": (s["principal"].iloc[-1] / POOL) ** (1 / out_years) - 1 if False else None,
        "S52_daily": sharpe_of(daily, rf_daily, 52),
        "S252_daily": sharpe_of(daily, rf_daily, 252),
        "S_weekly": sharpe_of(wdf, RF, 52),
    }
    return out


def out_years():
    return 10.0


def main():
    print("口径说明: S52_daily=日收益×√52(现口径,不一致低估) | S252_daily=日收益×√252(正确) | S_weekly=周收益×√52(公募常用)")
    print("超额部分已减 日度无风险利率 1.3%/252")
    rows = []
    for code in BROAD:
        s = curve_series(code, V3_PARAMS, V3_KW)
        pr = s["principal"].values
        years = (s["date"].max() - s["date"].min()).days / 365.25
        annual = (pr[-1] / POOL) ** (1 / years) - 1
        daily = pr[1:] / pr[:-1] - 1
        wdf = pd.DataFrame({"wk": s["date"].dt.to_period("W").values[1:], "r": daily}).groupby("wk")["r"].apply(lambda x: (1 + x).prod() - 1).values
        rf_daily = RF / 252
        rows.append({
            "code": code, "name": NAMES[code], "years": years, "annual": annual,
            "S52_daily": sharpe_of(daily, rf_daily, 52),
            "S252_daily": sharpe_of(daily, rf_daily, 252),
            "S_weekly": sharpe_of(wdf, RF, 52),
        })
        print(f"{NAMES[code]:<8} 年化{annual*100:5.2f}% | S52(现)={rows[-1]['S52_daily']:5.2f} | S252(正确)={rows[-1]['S252_daily']:5.2f} | S周={rows[-1]['S_weekly']:5.2f}")

    # 多指数组合: 等权合并 daily 收益率
    print("\n=== 多指数等权组合 (分散化) ===")
    series = {c: curve_series(c, V3_PARAMS, V3_KW) for c in BROAD}
    all_dates = sorted(set().union(*[set(s["date"]) for s in series.values()]))
    df = pd.DataFrame({"date": all_dates})
    for c in BROAD:
        s = series[c]
        df = df.merge(s[["date", "principal"]].rename(columns={"principal": c}), on="date", how="left")
    df = df.set_index("date")
    # 组合: 取所有指数都有数据的最大连续区间
    valid = df.notna().all(axis=1)
    runs = (valid != valid.shift()).cumsum()
    best = runs[valid].value_counts().idxmax()
    seg = df[valid][runs[valid] == best]
    w = np.ones(len(BROAD)) / len(BROAD)
    port = (seg.values @ w)
    pr = port / port[0] * POOL
    years = (seg.index.max() - seg.index.min()).days / 365.25
    annual = (pr[-1] / POOL) ** (1 / years) - 1
    daily = pr[1:] / pr[:-1] - 1
    wdf = pd.DataFrame({"wk": seg.index.to_period("W").values[1:], "r": daily}).groupby("wk")["r"].apply(lambda x: (1 + x).prod() - 1).values
    rf_daily = RF / 252
    print(f"组合区间 {seg.index.min().date()} ~ {seg.index.max().date()} ({years:.1f}年)")
    print(f"等权组合: 年化 {annual*100:5.2f}% | S52(现)={sharpe_of(daily,rf_daily,52):5.2f} | S252={sharpe_of(daily,rf_daily,252):5.2f} | S周={sharpe_of(wdf,RF,52):5.2f}")


if __name__ == "__main__":
    main()
