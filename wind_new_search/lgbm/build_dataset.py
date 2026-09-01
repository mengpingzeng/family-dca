#!/usr/bin/env python3
"""
LGBM 分位数模型 — 构造数据集.

目标: 未来 12 周收益率 (price[t+12]/price[t] - 1)
特征:
  pe_pct / pb_pct / fed_pct : 3 年(156周)滚动百分位 (低=便宜, 与原 pe_pct 方向一致)
  vol                       : 12 周年化波动率 (周对数收益标准差 × √52)
  is_hs300                  : 指数标签 (沪深300=1, 中证500=0)

样本范围: 沪深300(000300) + 中证500(000905), 原始价格/估值自 2005/2007 起.

注: 原数据自带的 pe_pct/pb_pct/fed_pct 是 10 年百分位, 最早 2015/2017 才有效,
    无法覆盖 2005-2015 训练窗口, 故此处用 3 年滚动百分位重算 (独立计算, 不改原数据).

输出(独立): wind_new_search/lgbm/data/lgbm_dataset.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
OUT_DIR = Path(__file__).resolve().parent / "data"

PCT_W = 156   # 3 年滚动百分位窗口
VOL_W = 12    # 波动率窗口 (周)
FWD = int(sys.argv[1]) if len(sys.argv) > 1 else 12  # 未来 N 周 (默认 12)

CODES = [("000300", 1), ("000905", 0)]


def rolling_pct(s, w):
    """滚动百分位: 当前值在过去 w 个值(含当前)中的分位, 低=便宜."""
    arr = np.asarray(s, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        win = arr[i - w + 1: i + 1]
        out[i] = float((win <= arr[i]).sum() / len(win))
    return out


def build(code, is_hs300):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    price = df["price"].astype(float).values
    pe = df["pe"].astype(float).values
    pb = df["pb"].astype(float).values
    fed = df["fed"].astype(float).values

    # 周对数收益 + 年化波动率
    logr = np.log(price[1:] / price[:-1])
    logr = np.concatenate([[np.nan], logr])
    vol = pd.Series(logr).rolling(VOL_W).std().values * np.sqrt(52)

    # 未来 12 周简单收益
    y = np.full(len(df), np.nan)
    for i in range(len(df) - FWD):
        if price[i] > 0 and price[i + FWD] > 0:
            y[i] = price[i + FWD] / price[i] - 1.0

    out = pd.DataFrame({
        "date": df["date"].values,
        "is_hs300": is_hs300,
        "pe_pct": rolling_pct(pe, PCT_W),
        "pb_pct": rolling_pct(pb, PCT_W),
        "fed_pct": rolling_pct(fed, PCT_W),
        "vol": vol,
        "y": y,
    })
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [build(code, is_hs) for code, is_hs in CODES]
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])

    valid = data.dropna(subset=["pe_pct", "pb_pct", "fed_pct", "vol", "y"]).reset_index(drop=True)
    print(f"总样本(含NaN): {len(data)}, 有效样本: {len(valid)}")
    print(f"日期范围: {data['date'].min().date()} ~ {data['date'].max().date()}")
    print(f"有效样本日期范围: {valid['date'].min().date()} ~ {valid['date'].max().date()}")
    print(f"\n各指数有效样本数:")
    print(valid.groupby("is_hs300").size())
    print(f"\n目标 y 分布: 均值 {valid['y'].mean()*100:.2f}%  中位 {valid['y'].median()*100:.2f}%  "
          f"std {valid['y'].std()*100:.2f}%  P10 {valid['y'].quantile(0.1)*100:.2f}%  P90 {valid['y'].quantile(0.9)*100:.2f}%")
    print(f"\n特征统计:")
    print(valid[["pe_pct", "pb_pct", "fed_pct", "vol"]].describe().round(3).to_string())

    out_path = OUT_DIR / f"lgbm_dataset_{FWD}w.parquet"
    valid.to_parquet(out_path, index=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
