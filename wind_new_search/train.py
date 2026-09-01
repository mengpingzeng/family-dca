#!/usr/bin/env python3
"""
训练集网格搜索 — 不对称「主信号 + 闸门」组合实验。

在沪深300 + 中证500 上搜索最优策略:
  买入主信号 buy_signal ∈ {PE, PB, FED}
  买入闸门   buy_gate    ∈ {None, PE, PB, FED} (上限 buy_gate_cap ∈ 0.55~0.80)
  卖出信号   sell_signal ∈ {PE, PB, FED}
  买入分档/卖出阈值沿用既有范围

统一评分 = min(两指数 XIRR) (防过拟合)
与旧最优 (PB_only B10/15/30/70 S85/95) 做基线对比。

用法:
  python wind_new_search/train.py              # 原版 (无本金阈值)
  python wind_new_search/train.py --capped     # 训练集 + 本金阈值 30万
输出:
  原版   -> wind_new_search/output/train_results.json / full_results.parquet
  --capped -> wind_new_search/output/train_results_capped.json / full_results_capped.parquet
"""

import argparse
import concurrent.futures
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import run_backtest, prep_df

# 模块级全局, 由命令行 --capped 在 main() 中设置; 通过 fork 继承给 worker 进程。
PRINCIPAL_THRESHOLD = None

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
OUTPUT_DIR = PROJECT_DIR / "wind_new_search" / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CODES = ["000300", "000905"]
TRAIN_NAMES = {"000300": "沪深300", "000905": "中证500"}

# 参数空间 (本版新增卖出侧确认闸门, 并收窄买入/卖出档以控制组合数)
BUY_SIGNALS = ["PE", "PB", "FED"]
BUY_GATES = [None, "PE", "PB", "FED"]
BUY_GATE_CAPS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
SELL_SIGNALS = ["PE", "PB", "FED"]
SELL_GATES = [None, "PE", "PB", "FED"]
SELL_GATE_FLOORS = [0.50, 0.60, 0.70, 0.80]
BUY_FLOORS = [0.08, 0.10]
BUY_LOWS = [0.15, 0.20]
BUY_MIDS = [0.30]
BUY_HIGHS = [0.50, 0.55, 0.60, 0.70]
SELL_HEAVYS = [0.80, 0.85]
SELL_EXTREMES = [0.90, 0.95]

MIN_TRADES = 5
WORKERS = int(os.environ.get("TRAIN_WORKERS", min(4, os.cpu_count() or 1)))
FLOOR = 0.08  # 方案 C 的 min XIRR 下限

# 旧最优 (PB_only B10/15/30/70 S85/95) — 用于基线对比
BASELINE = {
    "buy_signal": "PB", "buy_gate": None, "buy_gate_cap": None, "sell_signal": "PE",
    "sell_gate": None, "sell_gate_floor": None,
    "buy_floor": 0.10, "buy_low": 0.15, "buy_mid": 0.30, "buy_high": 0.70,
    "sell_heavy": 0.85, "sell_extreme": 0.95,
}


def gen_combos():
    combos = []
    for buy_signal in BUY_SIGNALS:
        for buy_gate in BUY_GATES:
            caps = BUY_GATE_CAPS if buy_gate else [None]
            for buy_gate_cap in caps:
                for sell_signal in SELL_SIGNALS:
                    for sell_gate in SELL_GATES:
                        floors = SELL_GATE_FLOORS if sell_gate else [None]
                        for sell_gate_floor in floors:
                            for bf, bl, bm, bh in itertools.product(BUY_FLOORS, BUY_LOWS, BUY_MIDS, BUY_HIGHS):
                                if not (bf < bl < bm < bh):
                                    continue
                                for sh, se in itertools.product(SELL_HEAVYS, SELL_EXTREMES):
                                    if not (sh < se):
                                        continue
                                    combos.append({
                                        "buy_signal": buy_signal,
                                        "buy_gate": buy_gate,
                                        "buy_gate_cap": buy_gate_cap,
                                        "sell_signal": sell_signal,
                                        "sell_gate": sell_gate,
                                        "sell_gate_floor": sell_gate_floor,
                                        "buy_floor": bf, "buy_low": bl, "buy_mid": bm, "buy_high": bh,
                                        "sell_heavy": sh, "sell_extreme": se,
                                    })
    return combos


_WORKER_DFS = {}


def _init_worker():
    global _WORKER_DFS
    for code in TRAIN_CODES:
        df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        _WORKER_DFS[code] = prep_df(df)


def _eval(params):
    row = dict(params)
    xirrs = []
    for code in TRAIN_CODES:
        r = run_backtest(_WORKER_DFS[code], params, principal_threshold=PRINCIPAL_THRESHOLD)
        if r["trades"] < MIN_TRADES:
            return None
        row[f"{code}_xirr"] = r["xirr"]
        row[f"{code}_return"] = r["final_return"]
        row[f"{code}_trades"] = r["trades"]
        xirrs.append(r["xirr"])
    row["unified_xirr"] = round(min(xirrs), 4)
    row["avg_xirr"] = round(sum(xirrs) / len(xirrs), 4)
    return row


def _pct_cmp_train(signal, op, value):
    """训练控制台输出用的百分位口径文案 (FED 取反)."""
    if signal == "FED":
        flip = {"<=": "≥", ">=": "≤"}
        return f"{signal}%{flip[op]}{1 - value:.0%}"
    disp = {"<=": "≤", ">=": "≥"}
    return f"{signal}%{disp[op]}{value:.0%}"


def _fmt_params(p):
    gate = _pct_cmp_train(p["buy_gate"], "<=", p["buy_gate_cap"]) if p["buy_gate"] else "无"
    sgate = _pct_cmp_train(p["sell_gate"], ">=", p["sell_gate_floor"]) if p["sell_gate"] else "无"
    return (f"{p['buy_signal']}主/{gate}闸/卖{p['sell_signal']}/{sgate}卖闸 "
            f"B{p['buy_floor']:.0%}/{p['buy_low']:.0%}/{p['buy_mid']:.0%}/{p['buy_high']:.0%} "
            f"S{p['sell_heavy']:.0%}/{p['sell_extreme']:.0%}")


def main():
    global PRINCIPAL_THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--capped", action="store_true",
                    help="训练集 + 本金阈值 30万 (输出 *_capped.json / *_capped.parquet, 不覆盖原版结果)")
    ap.add_argument("--threshold", type=float, default=300_000,
                    help="本金阈值(元), 默认 300000")
    args = ap.parse_args()
    PRINCIPAL_THRESHOLD = args.threshold if args.capped else None
    suffix = "_capped" if args.capped else ""
    tag = f" + 本金阈值{args.threshold/10000:.0f}万" if args.capped else ""

    combos = gen_combos()
    print(f"训练集不对称网格搜索{tag} — {len(combos)} 参数组合 × {len(TRAIN_CODES)} 指数 ({WORKERS} workers)")

    for code in TRAIN_CODES:
        df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        t = df[df["pe_pct"].notna()]
        print(f"  {code} {TRAIN_NAMES[code]}: {len(t)} 可交易行 ({t.date.min().date()} ~ {t.date.max().date()})")

    results = []
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as ex:
        for ci, row in enumerate(ex.map(_eval, combos, chunksize=200), 1):
            if row is not None:
                results.append(row)
            if ci % 10000 == 0:
                print(f"  进度 {ci}/{len(combos)} ({time.time()-t0:.0f}s)", flush=True)

    results.sort(key=lambda x: x["unified_xirr"], reverse=True)
    print(f"\n完成 {len(results)} 有效策略 ({time.time()-t0:.0f}s)\n")

    # 计算各评分指标的排序
    for r in results:
        r["score_balanced"] = round(0.5 * r["unified_xirr"] + 0.5 * r["avg_xirr"], 4)

    top_min = results[:200]
    top_avg = sorted(results, key=lambda x: x["avg_xirr"], reverse=True)[:200]
    top_balanced = sorted(results, key=lambda x: x["score_balanced"], reverse=True)[:200]
    qualified = [r for r in results if r["unified_xirr"] >= FLOOR]
    top_floored = sorted(qualified, key=lambda x: x["avg_xirr"], reverse=True)[:200]

    rankings = {
        "min": {"label": "min XIRR", "top": top_min},
        "avg": {"label": "avg XIRR", "top": top_avg},
        "balanced": {"label": "0.5·min + 0.5·avg", "top": top_balanced},
        "floored_avg": {"label": f"min≥{FLOOR:.0%} 后 max avg", "floor": FLOOR, "top": top_floored},
    }

    def _match_baseline(r):
        return (r["buy_signal"] == BASELINE["buy_signal"] and r["buy_gate"] == BASELINE["buy_gate"]
                and r["sell_signal"] == BASELINE["sell_signal"] and r["sell_gate"] == BASELINE["sell_gate"]
                and r["buy_floor"] == BASELINE["buy_floor"] and r["buy_low"] == BASELINE["buy_low"]
                and r["buy_mid"] == BASELINE["buy_mid"] and r["buy_high"] == BASELINE["buy_high"]
                and r["sell_heavy"] == BASELINE["sell_heavy"] and r["sell_extreme"] == BASELINE["sell_extreme"])

    baseline_row = next((r for r in results if _match_baseline(r)), None)
    if baseline_row:
        baseline_row = dict(baseline_row)
        baseline_row["rank"] = results.index(baseline_row) + 1
        print(f"\n旧最优基线 (PB_only B10/15/30/70 S85/95): min={baseline_row['unified_xirr']*100:.2f}% 排名 #{baseline_row['rank']}/{len(results)}")

    # 打印各指标 Top1 对比
    print("\n各指标 Top1 对比:")
    for key, rk in rankings.items():
        r = rk["top"][0]
        print(f"  [{rk['label']:20s}] min={r['unified_xirr']*100:.2f}% avg={r['avg_xirr']*100:.2f}% "
              f"{_fmt_params(r)} | 300={r['000300_xirr']*100:.1f}% 500={r['000905_xirr']*100:.1f}%")

    # 卖出闸门消融对比: 同一指标下, 无卖闸 vs 有卖闸 的最优
    _sort_key = {"min": "unified_xirr", "avg": "avg_xirr", "balanced": "score_balanced",
                 "floored_avg": "avg_xirr"}
    sell_gate_ablation = {}
    print("\n卖出闸门消融 (无卖闸 vs 有卖闸):")
    for key, rk in rankings.items():
        sk = _sort_key[key]
        pool = qualified if key == "floored_avg" else results
        no_g = max([r for r in pool if r["sell_gate"] is None], key=lambda x: x[sk])
        with_g = max([r for r in pool if r["sell_gate"] is not None], key=lambda x: x[sk])
        sell_gate_ablation[key] = {"label": rk["label"], "no_gate": no_g, "with_gate": with_g}
        print(f"  [{rk['label']:20s}] 无卖闸 min={no_g['unified_xirr']*100:.2f}% avg={no_g['avg_xirr']*100:.2f}% | "
              f"有卖闸 min={with_g['unified_xirr']*100:.2f}% avg={with_g['avg_xirr']*100:.2f}%  ({with_g['sell_gate']}≥{with_g['sell_gate_floor']:.0%})")

    out = {
        "codes": TRAIN_CODES,
        "names": TRAIN_NAMES,
        "min_trades": MIN_TRADES,
        "total_combos": len(combos),
        "valid_combos": len(results),
        "floor": FLOOR,
        "principal_threshold": PRINCIPAL_THRESHOLD,
        "baseline": baseline_row,
        "rankings": rankings,
        "sell_gate_ablation": sell_gate_ablation,
        "top": top_min,
    }
    with open(OUTPUT_DIR / f"train_results{suffix}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / f'train_results{suffix}.json'}")

    # 保存完整结果 (便于后续按新指标重排, 无需重跑)
    try:
        pd.DataFrame(results).to_parquet(OUTPUT_DIR / f"full_results{suffix}.parquet", index=False)
        print(f"保存全量: {OUTPUT_DIR / f'full_results{suffix}.parquet'} ({len(results)} 行)")
    except Exception as e:
        print(f"[warn] 全量结果保存失败: {e}")
    return top_min[0] if top_min else None


if __name__ == "__main__":
    main()
