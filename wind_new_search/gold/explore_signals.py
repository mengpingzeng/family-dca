#!/usr/bin/env python3
"""
黄金信号扩展探索（只读回测）

探索更多信号，找出在 2016-2026 真实有效的：
  动量类: 过去N月收益率 (3/6/12/24)
  趋势类: MA N日 (100/200/250/500)
  绝对实际利率阈值: TIPS < 0 满仓, > 2 空仓
  波动率过滤: 金价高位波动减仓
  组合: 动量 + 趋势
策略按月调仓，单边成本0.1%，输出与买入持有对比
"""
import os, sys
sys.path.insert(0, '/usr/local/python3.11/lib/python3.11/site-packages')
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
REBAL = "ME"
COST = 0.001


def load():
    au = pd.read_parquet(os.path.join(DATA_DIR, "au99.99.parquet"))
    au["date"] = pd.to_datetime(au["date"])
    tips = pd.read_parquet(os.path.join(DATA_DIR, "tips_10y.parquet"))
    tips["date"] = pd.to_datetime(tips["date"])
    df = pd.merge_asof(au.sort_values("date"), tips.sort_values("date"),
                       on="date", direction="backward")
    df = df.dropna(subset=["tips10y"]).sort_values("date").reset_index(drop=True)
    return df


def backtest(df, pos):
    pos = pos.reindex(df.index).ffill().fillna(0)
    monthly = df.set_index("date")[["price"]].resample(REBAL).last().index
    rebal_dates = set(monthly)
    prev = 0.0
    trades = 0
    ret = np.zeros(len(df))
    for i in range(len(df)):
        w = pos[i]
        if df["date"].iloc[i] in rebal_dates and i > 0:
            if abs(w - prev) > 1e-9:
                trades += 1
            prev = w
        r = 0.0
        if i > 0:
            r = df["price"].iloc[i] / df["price"].iloc[i - 1] - 1
        if abs(prev) > 1e-9:
            ret[i] = prev * r
    cost_arr = np.zeros(len(df))
    for i in range(1, len(df)):
        if df["date"].iloc[i] in rebal_dates:
            dw = abs(pos[i] - pos[i - 1])
            cost_arr[i] = -dw * COST
    net = ret + cost_arr
    nav = np.cumprod(1 + net)
    nav = pd.Series(nav)
    years = len(nav) / 252
    cagr = nav.iloc[-1] ** (1 / years) - 1
    dd = (nav / nav.cummax() - 1).min()
    sharpe = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0
    return {"CAGR": cagr, "MaxDD": dd, "Sharpe": sharpe, "Trades": trades,
            "Final": nav.iloc[-1]}


def main():
    df = load()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # 动量 (过去N日收益率)
    for m in [63, 126, 252, 504]:
        df[f"mom_{m}"] = df["price"] / df["price"].shift(m) - 1
    # 趋势 MA
    for m in [100, 200, 250, 500]:
        df[f"ma_{m}"] = df["price"].rolling(m).mean()
        df[f"above_{m}"] = (df["price"] > df[f"ma_{m}"]).astype(float)
    # 波动率 (过去60日年化)
    df["vol60"] = df["price"].pct_change().rolling(60).std() * np.sqrt(252)

    results = {}

    # 0 买入持有
    nav = df["price"] / df["price"].iloc[0]
    ret = nav.pct_change().fillna(0).to_numpy()
    results["0.买入持有"] = backtest(df, pd.Series(1.0, index=df.index))
    results["0.买入持有"]["Trades"] = 0

    # 1 动量
    for m in [63, 126, 252, 504]:
        results[f"1.动量{m//21}m"] = backtest(df, (df[f"mom_{m}"] > 0).astype(float))

    # 2 MA趋势
    for m in [100, 200, 250, 500]:
        results[f"2.MA{m}"] = backtest(df, df[f"above_{m}"])

    # 3 绝对TIPS阈值 (线性: TIPS<-0.5 满仓, >2 空仓)
    pos = ((-0.5 - df["tips10y"]) / 2.5).clip(0, 1)
    results["3.TIPS阈值"] = backtest(df, pos)

    # 4 动量+趋势组合
    for m, ma in [(126, 100), (252, 250)]:
        results[f"4.动量{m//21}m+MA{ma}"] = backtest(df,
            (df[f"mom_{m}"] > 0).astype(float) * df[f"above_{ma}"])

    # 5 趋势×波动率过滤 (高波动减半)
    vol_adj = pd.Series(np.where(df["vol60"] < 0.20, 1.0, 0.5), index=df.index)
    results["5.MA250×波动"] = backtest(df, df["above_250"] * vol_adj)

    print(f"{'策略':<18}{'年化':>8}{'最大回撤':>10}{'夏普':>8}{'交易次数':>8}{'终值倍数':>9}")
    print("-" * 66)
    for name, r in sorted(results.items()):
        print(f"{name:<18}{r['CAGR']:>8.2%}{r['MaxDD']:>10.2%}{r['Sharpe']:>8.2f}{r['Trades']:>8d}{r['Final']:>9.2f}")


if __name__ == "__main__":
    main()
