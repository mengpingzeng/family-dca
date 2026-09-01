#!/usr/bin/env python3
"""
选项3: 把分位数模型定位为「不确定性带」。

思路: 模型不预测方向(已证明相关≈0), 只输出区间宽度 = 不确定性。
  宽度窄 -> 相对确定 -> 可操作(方向交给估值)
  宽度宽 -> 高度不确定 -> 观望

本脚本验证两点:
  1. 宽度是否是真实的不确定性信号: corr(宽度, |实际-P50|) 是否显著为正
  2. 估值信号是否在"低不确定"时期更锐利: 便宜-贵的收益差 是否在窄区间时更大

数据: 12 周样本外预测 + 特征, 独立只读.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

FWD = 12


def main():
    preds = pd.DataFrame(json.load(open(BASE / "output" / f"quantile_result_{FWD}w.json"))["preds"])
    preds["date"] = pd.to_datetime(preds["date"])

    m = preds.copy()
    m["width"] = m["p90"] - m["p10"]
    m["abs_err"] = (m["actual"] - m["p50"]).abs()

    print(f"样本外样本: {len(m)}")

    # ---- 1. 宽度是否为真实不确定性信号 ----
    c_err = float(np.corrcoef(m["width"], m["abs_err"])[0, 1])
    c_vol = float(np.corrcoef(m["width"], m["vol"])[0, 1])
    print(f"\n=== 1. 宽度作为不确定性信号 ===")
    print(f"corr(宽度, |实际-P50|) = {c_err:.3f}  (越正, 越说明宽度=真实不确定性)")
    print(f"corr(宽度, 波动率vol) = {c_vol:.3f}  (宽度是否跟着历史波动率走)")

    # 分位对比: 宽度最低 1/4 vs 最高 1/4 的实际误差
    q_lo, q_hi = m["width"].quantile(0.25), m["width"].quantile(0.75)
    lo = m[m["width"] <= q_lo]; hi = m[m["width"] >= q_hi]
    print(f"宽度最低 1/4 (中位宽 {lo['width'].median()*100:.1f}%): 实际|误差|中位 {lo['abs_err'].median()*100:.2f}%")
    print(f"宽度最高 1/4 (中位宽 {hi['width'].median()*100:.1f}%): 实际|误差|中位 {hi['abs_err'].median()*100:.2f}%")

    # ---- 2. 估值信号在低不确定时是否更锐利 ----
    m["cheap"] = m["pb_pct"] < m["pb_pct"].quantile(0.33)
    m["expensive"] = m["pb_pct"] > m["pb_pct"].quantile(0.67)
    m["low_unc"] = m["width"] < m["width"].median()

    print(f"\n=== 2. 估值信号 × 不确定性 ===")
    print(f"{'不确定度':8} {'便宜实际':>10} {'贵实际':>10} {'价差(便宜-贵)':>14} {'便宜上涨占比':>12}")
    for label, sub in [("低(窄)", m[m["low_unc"]]), ("高(宽)", m[~m["low_unc"]]), ("全部", m)]:
        cheap_r = sub[sub["cheap"]]["actual"].mean() if sub["cheap"].sum() else np.nan
        exp_r = sub[sub["expensive"]]["actual"].mean() if sub["expensive"].sum() else np.nan
        cheap_win = (sub[sub["cheap"]]["actual"] > 0).mean() if sub["cheap"].sum() else np.nan
        print(f"{label:8} {cheap_r*100:9.2f}% {exp_r*100:9.2f}% {(cheap_r-exp_r)*100:13.2f}% {cheap_win*100:11.1f}%")

    # ---- 3. 简单决策规则回测 ----
    # 规则: 窄区间(可操作) & 便宜 -> 持有(做多); 其余 -> 空仓(0收益)
    m["signal"] = (m["low_unc"] & m["cheap"]).astype(int)
    m["strat_return"] = m["actual"] * m["signal"]
    n_buy = int(m["signal"].sum())
    print(f"\n=== 3. 简单规则(窄区间&便宜 -> 持有, 其余空仓) ===")
    print(f"持有期占比: {n_buy}/{len(m)} = {n_buy/len(m)*100:.1f}%")
    print(f"持有期平均实际收益: {m[m['signal']==1]['actual'].mean()*100:.2f}%  (上涨占比 {(m[m['signal']==1]['actual']>0).mean()*100:.1f}%)")
    print(f"空仓期平均实际收益: {m[m['signal']==0]['actual'].mean()*100:.2f}%")
    # 仅估值(不看宽度)对照: 便宜 -> 持有
    m["signal_val"] = m["cheap"].astype(int)
    print(f"\n对照-仅估值(便宜就持有): 持有 {(m['cheap']).sum()}/{len(m)} 期, 持有期平均 {m[m['cheap']]['actual'].mean()*100:.2f}%")
    print(f"对照-全仓持有: 平均 {m['actual'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
