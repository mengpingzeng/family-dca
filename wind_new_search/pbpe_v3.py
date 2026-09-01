#!/usr/bin/env python3
"""
均衡策略 v3 (PB×PE 信号质量版) — 共享配置。

在 v2 (更早止盈0.80 + 20周均线软制动β0.5 + 顺势加码1.5) 基础上, 加入 PB/PE 双信号质量闸门:
  买入更谨慎: 需 PE 也便宜 (PE% ≤ 0.45), 避免"PB破净但盈利下滑"的假便宜
  卖出更严格: 需 PB 也偏贵 (PB% ≥ 0.70), 避免"PE偏高但估值仍便宜"的误卖
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.balanced_v2 import (
    PARAMS as V2_PARAMS, KW as V2_KW, MULTS, BASE, COMMISSION_RATE,
    MIN_COMMISSION, THRESHOLD, CAP, POOL, ETF_MAP,
)

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"

PARAMS = {**V2_PARAMS,
          "buy_gate": ["FED", "PE"], "buy_gate_cap": [0.55, 0.45],
          "sell_gate": ["PB"], "sell_gate_floor": [0.70]}
KW = dict(V2_KW)
