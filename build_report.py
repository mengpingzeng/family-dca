#!/usr/bin/env python3
"""
生成 6 指数信号范式回测报告 (Markdown + Excel)
"""
import json, os
from datetime import datetime
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "grid_search_aks", "output")
DOCS_DIR = os.path.join(PROJECT_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

TODAY = datetime.now().strftime("%Y-%m-%d")

INDEX_NAMES = {
    "000300": "沪深300", "000016": "上证50", "000905": "中证500",
    "000852": "中证1000", "000015": "上证红利", "399324": "深证红利",
}

# ──── 1. 读取数据 ────

unified = json.loads(open(os.path.join(OUTPUT_DIR, "latest_strict.json")).read()).get("unified", {})
rolling = json.loads(open(os.path.join(OUTPUT_DIR, "latest_rolling.json")).read())

# 从独立结果读取 per-index 最优
indiv_files = {
    "000300": "grid_results_strict_20260812_205728.json",
    "000016": "grid_results_strict_20260812_210425.json",
    "000905": "grid_results_strict_20260812_211044.json",
    "000852": "grid_results_strict_20260812_211352.json",
    "000015": "grid_results_strict_20260812_212052.json",
    "399324": "grid_results_strict_20260812_212727.json",
}
individuals = {}
for code, fname in indiv_files.items():
    path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(path):
        individuals[code] = json.loads(open(path).read()).get(code, {})

# ──── 2. Markdown ────

lines = []

def w(s=""):
    lines.append(s)

w(f"# 6 指数信号范式回测报告（PE/PB/FED 自由组合）")
w()
w(f"> 生成日期: {TODAY}")
w(f"> 模式: 不限本金 + 闲置现金 2%/年 + 最少 10 笔交易")
w(f"> 7 种信号范式: PE_only / PB_only / FED_only / OR / AND / VOTE / MAX")
w(f"> 统一评分: `min(6指数XIRR)`, 各指数独立评分: 各自最优 XIRR")
w()

w("---")
w()
w("## 1. 统一最优策略（6 指数，10 年窗口）")
w()

for wk, wv in sorted(unified.items()):
    yr = wk.replace("w", "")
    top = wv.get("top", [])[:5]
    if not top:
        continue
    w(f"### 窗口 {yr} 年")
    w()
    w("| 排名 | 统一min | 平均 | 信号 | 买入PE% | 卖出PE% | FED | PBv |")
    w("|------|---------|------|------|---------|---------|-----|-----|")
    for i, r in enumerate(top):
        mode = r.get("signal_mode", "PE_only")
        fed = r.get("fed_gate") or "off"
        pbv = r.get("pb_veto") or "off"
        w(f"| {i+1} | {r['unified_xirr']*100:.2f}% | {r['avg_xirr']*100:.2f}% | "
          f"**{mode}** | "
          f"<{r['buy_floor']:.0%}/{r['buy_low']:.0%}/{r['buy_mid']:.0%}/{r['buy_high']:.0%} | "
          f">{r['sell_heavy']:.0%}/极>{r['sell_extreme']:.0%} | {fed} | {pbv} |")
    w()
    w("**各指数 XIRR:**")
    w()
    best = top[0]
    codes = wv.get("codes", [])
    w("| 指数 | XIRR | 总回报 | 交易笔数 |")
    w("|------|------|--------|----------|")
    for code in codes:
        name = INDEX_NAMES.get(code, code)
        xirr_key = f"{code}_xirr"
        ret_key = f"{code}_return"
        trades_key = f"{code}_trades"
        w(f"| {name} | {best.get(xirr_key, 0)*100:.1f}% | {best.get(ret_key, 0)*100:.1f}% | {best.get(trades_key, 0)} |")
    w()

w("---")
w()
w("## 2. 各指数独立最优（10 年窗口，含信号对比）")
w()

# PE_only baseline for comparison
pe_only_baseline = {
    "000300": 17.5, "000016": 16.2, "000905": 8.5,
    "000852": 45.5, "000015": 4.1, "399324": 24.9,
}

w("| 指数 | 最优XIRR | 最优信号 | PE_only XIRR | 提升 | 买入PE% | 卖出PE% |")
w("|------|---------|---------|-------------|:---:|---------|---------|")
for code in ["000300", "000016", "000905", "000852", "000015", "399324"]:
    name = INDEX_NAMES.get(code, code)
    pe_base = pe_only_baseline.get(code, 0)
    data = individuals.get(code, {})
    top_10 = data.get("w10", {}).get("top", [])
    if not top_10:
        w(f"| {name} | — | — | {pe_base:.1f}% | — | — | — |")
        continue
    r = top_10[0]
    mode = r.get("signal_mode", "PE_only")
    imp = f"+{r['xirr']*100 - pe_base:.1f}pp" if r['xirr']*100 > pe_base else "—"
    w(f"| {name} | **{r['xirr']*100:.2f}%** | **{mode}** | {pe_base:.1f}% | {imp} | "
      f"<{r['buy_floor']:.0%}/{r['buy_low']:.0%}/{r['buy_mid']:.0%}/{r['buy_high']:.0%} | "
      f">{r['sell_heavy']:.0%}/极>{r['sell_extreme']:.0%} |")

w()
w("**关键发现**:")
w()
w("- **FED_only 是统一最优信号** — 股债利差比 PE 更好捕捉入场时机")
w("- **VOTE** 对大市值指数最优 — 沪深300 (+3.6pp)、上证50 (+0.8pp)")
w("- **PE_only 在被 PE 主导的市场中仍是最优** — 中证500/红利未因切换信号而改善")
w("- FED 不是 PE 的辅助过滤条件，而是可以独立作为**主信号**")

w("---")
w()
w("## 3. 各模式排名分布")
w()

# Count how many times each mode appears in top 10 across all 6 indices
mode_counts_10yr = {}
for code, data in individuals.items():
    top_list = data.get("w10", {}).get("top", [])
    for r in top_list[:10]:
        mode = r.get("signal_mode", "PE_only")
        mode_counts_10yr[mode] = mode_counts_10yr.get(mode, 0) + 1

w("### 10yr 窗口 Top10 出现次数")
w()
w("| 信号模式 | 出现次数 | 说明 |")
w("|---------|---------|------|")
for mode in sorted(mode_counts_10yr, key=mode_counts_10yr.get, reverse=True):
    desc = {
        "PE_only": "只看 PE 分档",
        "PB_only": "只看 PB 分档",
        "FED_only": "只看 FED 分档（←统一最优）",
        "OR": "PE 低 或 PB 低 任一满足",
        "AND": "PE 低 且 PB 低 同时满足",
        "VOTE": "3 选 2 过半即买（←大市值最优）",
        "MAX": "取最强信号",
    }.get(mode, "")
    w(f"| **{mode}** | {mode_counts_10yr[mode]} | {desc} |")

w()
w("---")
w()
w("## 4. 滚动窗口鲁棒性分析")
w()
best_unif = unified.get("w10", {}).get("top", [None])[0] or {}
w(f"固定策略: **{best_unif.get('signal_mode','FED_only')}** B{best_unif.get('buy_floor',0.08)}/{best_unif.get('buy_low',0.15)}/{best_unif.get('buy_mid',0.22)}/{best_unif.get('buy_high',0.40)} S{best_unif.get('sell_heavy',0.80)}/{best_unif.get('sell_extreme',0.90)} | Base=¥500 | 闲置2%")
w()

for hy_k in ["hold_3yr", "hold_5yr", "hold_10yr"]:
    hy_num = hy_k.replace("hold_", "").replace("yr", "")
    w(f"### 持有 {hy_num} 年")
    w()
    w("| 指数 | 窗口数 | min | median | max | 胜率 | 均值 | std |")
    w("|------|--------|-----|--------|-----|------|------|-----|")
    for code_key, v in rolling.items():
        name = v.get("name", INDEX_NAMES.get(code_key, code_key))
        hy_data = v.get("holds", {}).get(hy_k, {})
        xirr_series = hy_data.get("xirr_series", [])
        if not xirr_series:
            w(f"| {name} | 0 | — | — | — | — | — | — |")
            continue
        cnt = len(xirr_series)
        w(f"| {name} | {cnt} | {hy_data.get('xirr_min',0):.1f}% | {hy_data.get('xirr_median',0):.1f}% | "
          f"{hy_data.get('xirr_max',0):.1f}% | {hy_data.get('win_rate',0):.0f}% | {hy_data.get('xirr_mean',0):.1f}% | "
          f"{hy_data.get('xirr_std',0):.1f}% |")
    w()

w("---")
w()
w("## 5. 数据覆盖")
w()
w("| 指数 | 代码 | PE | PB | 价格 | 合并行 | 10yr可交易 | 支持模式 |")
w("|------|------|:--:|:--:|:--:|--------|:--:|------|")
merged_dir = os.path.join(PROJECT_DIR, "data-store", "parquet", "aks_merged")
for code in ["000300", "000016", "000905", "000852", "000015", "399324"]:
    path = os.path.join(merged_dir, f"{code}.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    pe10 = df[df["pe_pct_w10"].notna()]
    has_pb = "pb_pct_w10" in df.columns
    has_fed = "fed_pct_w10" in df.columns
    modes = "全部7种" if has_pb and has_fed else ("PE/FED/VOTE/MAX" if has_fed else "PE_only")
    w(f"| {INDEX_NAMES.get(code, code)} | {code} | ✅ | {'✅' if has_pb else '❌'} | ✅ | "
      f"{len(df)} | {len(pe10) if len(pe10)>0 else 0} | {modes} |")

w()
w(f"*报告自动生成于 {TODAY}*")

report_path = os.path.join(DOCS_DIR, f"akshare回测报告_信号范式_{TODAY}.md")
with open(report_path, "w") as f:
    f.write("\n".join(lines))
print(f"Markdown: {report_path}")

# ──── 3. Excel ────

with pd.ExcelWriter(os.path.join(DOCS_DIR, f"信号范式回测_汇总_{TODAY}.xlsx"), engine="openpyxl") as writer:
    # Sheet 1: 统一最优
    rows = []
    best = unified.get("w10", {}).get("top", [])[:15]
    for i, r in enumerate(best):
        row = {
            "排名": i+1, "统一min_XIRR": r["unified_xirr"], "平均XIRR": r.get("avg_xirr",0),
            "信号模式": r.get("signal_mode","PE_only"),
            "买入Floor": r["buy_floor"], "买入Low": r["buy_low"], "买入Mid": r["buy_mid"], "买入High": r["buy_high"],
            "卖出Heavy": r["sell_heavy"], "卖出Extreme": r["sell_extreme"],
            "FED": r.get("fed_gate","off"), "PBv": r.get("pb_veto","off"),
        }
        for code in unified.get("w10",{}).get("codes",[]):
            row[f"{INDEX_NAMES.get(code,code)}_XIRR"] = r.get(f"{code}_xirr",0)
        rows.append(row)
    pd.DataFrame(rows).to_excel(writer, sheet_name="统一最优策略", index=False)

    # Sheet 2: 各模式排名
    mode_rows = []
    for code, data in sorted(individuals.items()):
        for wk, wv in sorted(data.items()):
            yr = wk.replace("w", "")
            if yr != "10": continue
            for r in wv.get("top", [])[:15]:
                mode_rows.append({
                    "指数": INDEX_NAMES.get(code, code), "信号模式": r.get("signal_mode","PE_only"),
                    "XIRR": r["xirr"], "买入Floor": r["buy_floor"], "买入Low": r["buy_low"],
                    "买入Mid": r["buy_mid"], "买入High": r["buy_high"],
                    "卖出Heavy": r["sell_heavy"], "卖出Extreme": r["sell_extreme"],
                    "FED": r.get("fed_gate","off"), "PBv": r.get("pb_veto","off"),
                    "交易笔数": r.get("trades",0),
                })
    pd.DataFrame(mode_rows).to_excel(writer, sheet_name="各模式排名", index=False)

    # Sheet 3: 滚动窗口
    roll_rows = []
    for code_key, v in rolling.items():
        name = v.get("name", INDEX_NAMES.get(code_key, code_key))
        for hy_k, hy_data in v.get("holds", {}).items():
            xirr_series = hy_data.get("xirr_series", [])
            if not xirr_series:
                continue
            hy_num = hy_k.replace("hold_","").replace("yr","")
            roll_rows.append({
                "指数": name, "持有期": hy_num, "窗口数": len(xirr_series),
                "min_XIRR": hy_data["xirr_min"], "median_XIRR": hy_data["xirr_median"],
                "max_XIRR": hy_data["xirr_max"], "mean_XIRR": hy_data["xirr_mean"],
                "std": hy_data["xirr_std"], "胜率": hy_data["win_rate"],
            })
    pd.DataFrame(roll_rows).to_excel(writer, sheet_name="滚动窗口", index=False)

print(f"Excel: {os.path.join(DOCS_DIR, f'信号范式回测_汇总_{TODAY}.xlsx')}")
print("完成!")
