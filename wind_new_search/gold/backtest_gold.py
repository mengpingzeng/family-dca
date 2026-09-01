#!/usr/bin/env python3
"""
黄金信号回测验证（只读，不污染宽基）

对比三种策略在 Au99.99 上的表现（2016-12 ~ 2026-08）:
  0. 买入持有 Buy&Hold
  1. 实际利率分位 TIPS 分位策略:
       TIPS 实际收益率处于历史高分位(高利率=压金价) → 卖出/低配
       TIPS 处于历史低分位(低利率=利好金价) → 买入/超配
  2. MA 趋势策略:
       price > MA250 → 持仓; price < MA250 → 空仓
  3. 组合: TIPS 分位方向 + MA 过滤

仓位口径: 用信号强度分档 (0/0.5/1.0)，每月调仓，扣成本(买卖各0.1%)
输出: 年化收益, 最大回撤, 夏普, 交易次数, 与买入持有对比
"""
import os, sys
sys.path.insert(0, '/usr/local/python3.11/lib/python3.11/site-packages')
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

REBAL = "ME"          # 每月调仓
COST = 0.001          # 单边交易成本 0.1%


def load():
    au = pd.read_parquet(os.path.join(DATA_DIR, "au99.99.parquet"))
    au["date"] = pd.to_datetime(au["date"])
    tips = pd.read_parquet(os.path.join(DATA_DIR, "tips_10y.parquet"))
    tips["date"] = pd.to_datetime(tips["date"])
    df = pd.merge_asof(au.sort_values("date"), tips.sort_values("date"),
                       on="date", direction="backward")
    df = df.dropna(subset=["tips10y"]).sort_values("date").reset_index(drop=True)
    return df


def add_signals(df, tips_window=5 * 252, ma_window=250):
    """TIPS 分位(滚动) + MA 趋势"""
    df = df.copy()
    # TIPS 实际利率滚动分位: 历史越高分位 → 越接近卖出区
    df["tips_pct"] = df["tips10y"].rolling(tips_window).rank(pct=True)
    # MA 趋势
    df["ma"] = df["price"].rolling(ma_window).mean()
    df["above_ma"] = (df["price"] > df["ma"]).astype(float)
    return df


def backtest(df, position):
    """position: Series of target weight, resampled monthly. Returns metrics."""
    pos = position.reindex(df.index).ffill().fillna(0)
    # 计算调仓日（月频）
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
    # 调仓日扣交易成本
    cost_arr = np.zeros(len(df))
    for i in range(1, len(df)):
        if df["date"].iloc[i] in rebal_dates:
            dw = abs(pos[i] - pos[i - 1])
            cost_arr[i] = -dw * COST
    net = ret + cost_arr
    nav = np.cumprod(1 + net)
    return analyze(nav, net, trades)


def analyze(nav, ret, trades):
    nav = pd.Series(nav)
    ret = pd.Series(ret)
    years = len(nav) / 252
    cagr = nav.iloc[-1] ** (1 / years) - 1
    dd = (nav / nav.cummax() - 1).min()
    if ret.std() > 0:
        sharpe = ret.mean() / ret.std() * np.sqrt(252)
    else:
        sharpe = 0
    return {"CAGR": cagr, "MaxDD": dd, "Sharpe": sharpe, "Trades": trades,
            "Final": nav.iloc[-1]}


def main():
    df = load()
    print(f"数据: {df['date'].min().date()} ~ {df['date'].max().date()}, {len(df)}行")
    print(f"金价: {df['price'].iloc[0]:.1f} → {df['price'].iloc[-1]:.1f} 元/克\n")
    df = add_signals(df)

    results = {}

    # 0. 买入持有
    nav = df["price"] / df["price"].iloc[0]
    ret = nav.pct_change().fillna(0).to_numpy()
    results["0.买入持有"] = analyze(pd.Series(nav), ret, 0)

    # 1. TIPS 分位策略（方向分档: 低分位满仓, 高分位空仓, 中间线性）
    p = df["tips_pct"]
    # 低分位(<30%)满仓, 高分位(>70%)空仓, 中间线性过渡
    pos1 = ((0.7 - p) / 0.4).clip(0, 1)
    results["1.TIPS分位"] = backtest(df, pos1)

    # 2. MA250 趋势
    results["2.MA趋势"] = backtest(df, df["above_ma"])

    # 3. TIPS × MA 组合
    results["3.TIPS×MA"] = backtest(df, pos1 * df["above_ma"])

    print(f"{'策略':<12}{'年化':>8}{'最大回撤':>10}{'夏普':>8}{'交易次数':>8}{'终值倍数':>9}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<12}{r['CAGR']:>8.2%}{r['MaxDD']:>10.2%}{r['Sharpe']:>8.2f}{r['Trades']:>8d}{r['Final']:>9.2f}")

    # 保存信号表供后续模块使用
    df.to_parquet(os.path.join(DATA_DIR, "gold_signals.parquet"), index=False)
    print(f"\n信号表已保存: {os.path.join(DATA_DIR, 'gold_signals.parquet')}")


if __name__ == "__main__":
    main()
