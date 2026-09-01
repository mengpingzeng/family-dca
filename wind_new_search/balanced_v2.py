#!/usr/bin/env python3
"""
均衡策略 v2 (Sharpe 提升版) — 共享配置。

在均衡策略基础上 (PB买/FED闸门/PE卖, 固定30万口径) 加入三项调制, 目标: 提升夏普、压缩回撤、保持收益:
  1. 更早止盈: sell_heavy 0.85 -> 0.80 (PE%>=80% 即开始分批卖出 20%)
  2. 下跌少买: 20周均线软制动 β=0.5 (价格 < SMA20 时买入倍数打 5 折)
  3. 顺势加码: 价格 >= SMA20 时买入倍数 ×1.5 (γ=1.5)
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import build_curve, prep_df
from wind_new_search.test_balanced import BALANCED_PARAMS, BALANCED_MULTS, NAMES

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"

PARAMS = {**BALANCED_PARAMS, "sell_heavy": 0.80}
KW = {"ma_window": 20, "ma_below": 0.5, "ma_above": 1.5}
MULTS = BALANCED_MULTS
BASE = 1000
COMMISSION_RATE = 0.0005
MIN_COMMISSION = 5.0
THRESHOLD = 200_000
CAP = 300_000
POOL = 300_000

ETF_MAP = {
    "000300": ("510300", "华泰柏瑞沪深300ETF"),
    "000905": ("510500", "南方中证500ETF"),
    "000015": ("510880", "华泰柏瑞红利ETF"),
    "000016": ("510050", "华夏上证50ETF"),
    "399330": ("159901", "易方达深证100ETF"),
    "399006": ("159915", "易方达创业板ETF"),
    "000688": ("588000", "华夏科创50ETF"),
    "000852": ("512100", "南方中证1000ETF"),
    "HSI":    ("159920", "华夏恒生ETF"),
    "HSTECH": ("513180", "华夏恒生科技ETF"),
    "SPX500": ("513500", "博时标普500ETF"),
    "NDX100": ("513100", "国泰纳指100ETF"),
}
