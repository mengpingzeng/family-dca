#!/usr/bin/env python3
"""
固定30万本金池 训练集网格搜索 — 买卖策略 × 定投基础金额 x。

相比 train.py 的差异:
  1. 定投基础金额 base = x ∈ {350, 500, 750, 1000} (保留 0.5x/1x/2x/3x 分档)
  2. 交易费用: 佣金万5 + 单笔低消5元 (双向)
  3. 本金硬封顶 30万: 累计买入本金(含费)达 30万 后停止买入
  4. 新增固定30万口径的年化收益 (把闲置本金的机会成本算进去)

两种排序并列对比:
  A. 固定30万年化 = min(两指数 principal_annual)
  B. 加权 XIRR    = 0.5·min XIRR + 0.5·avg XIRR

内存策略: 结果以 top-N(堆) + 分块 DataFrame 落盘, 避免全量结果驻留内存导致 OOM。

用法:
  python wind_new_search/train_principal.py
输出 (不覆盖旧文件):
  wind_new_search/output/train_principal.json
  wind_new_search/output/full_results_principal.parquet
"""

import concurrent.futures
import heapq
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import run_backtest, prep_df
from wind_new_search.train import (gen_combos, TRAIN_CODES, TRAIN_NAMES,
                                   MERGED_DIR, OUTPUT_DIR, MIN_TRADES, _fmt_params)

COMMISSION_RATE = 0.0005   # 万5
MIN_COMMISSION = 5.0       # 低消 5 元
PRINCIPAL_CAP = 300_000    # 本金硬封顶
PRINCIPAL_POOL = 300_000   # 固定本金池 (年化/本金曲线口径)
BASE_CANDIDATES = [350, 500, 750, 1000]

TOP_N = 200
CHUNK_ROWS = 20000

WORKERS = int(os.environ.get("TRAIN_WORKERS", min(4, os.cpu_count() or 1)))

RANK_KEYS = ["unified_principal_annual", "avg_principal_annual",
             "score_principal_balanced", "score_xirr_balanced"]
RANK_LABELS = {
    "unified_principal_annual": "min 固定30万年化",
    "avg_principal_annual": "avg 固定30万年化",
    "score_principal_balanced": "0.5·min+0.5·avg 固定30万年化",
    "score_xirr_balanced": "0.5·min XIRR + 0.5·avg XIRR",
}

_WORKER_DFS = {}


def _init_worker():
    global _WORKER_DFS
    for code in TRAIN_CODES:
        df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        _WORKER_DFS[code] = prep_df(df)


def _eval(args):
    params, base = args
    row = dict(params)
    row["base_amount"] = base
    xirrs = []
    anns = []
    for code in TRAIN_CODES:
        r = run_backtest(_WORKER_DFS[code], params, base_amount=base,
                         commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                         lot_size=0, principal_cap=PRINCIPAL_CAP, principal_pool=PRINCIPAL_POOL)
        if r["trades"] < MIN_TRADES:
            return None
        row[f"{code}_xirr"] = r["xirr"]
        row[f"{code}_return"] = r["final_return"]
        row[f"{code}_trades"] = r["trades"]
        row[f"{code}_buys"] = r["buys"]
        row[f"{code}_sells"] = r["sells"]
        row[f"{code}_invested"] = r["total_invested"]
        row[f"{code}_principal_annual"] = r["principal_annual"]
        row[f"{code}_principal_return"] = r["principal_return"]
        row[f"{code}_principal_final"] = r["principal_final"]
        xirrs.append(r["xirr"])
        anns.append(r["principal_annual"])
    row["unified_xirr"] = round(min(xirrs), 4)
    row["avg_xirr"] = round(sum(xirrs) / len(xirrs), 4)
    row["unified_principal_annual"] = round(min(anns), 4)
    row["avg_principal_annual"] = round(sum(anns) / len(anns), 4)
    row["score_xirr_balanced"] = round(0.5 * row["unified_xirr"] + 0.5 * row["avg_xirr"], 4)
    row["score_principal_balanced"] = round(
        0.5 * row["unified_principal_annual"] + 0.5 * row["avg_principal_annual"], 4)
    return row


def _push(heap, key, seq, item):
    if len(heap) < TOP_N:
        heapq.heappush(heap, (key, seq, item))
    elif key > heap[0][0]:
        heapq.heapreplace(heap, (key, seq, item))


def main():
    combos = gen_combos()
    total = len(combos) * len(BASE_CANDIDATES)
    print(f"固定30万本金池网格搜索 — {len(combos)} 策略 × {len(BASE_CANDIDATES)} base = {total} 组合 ({WORKERS} workers)")
    print(f"费用: 万5 + 低消5元 | 本金硬封顶 {PRINCIPAL_CAP/10000:.0f}万 | 固定本金池 {PRINCIPAL_POOL/10000:.0f}万 | base={BASE_CANDIDATES}")

    for code in TRAIN_CODES:
        df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        t = df[df["pe_pct"].notna()]
        print(f"  {code} {TRAIN_NAMES[code]}: {len(t)} 可交易行 ({t.date.min().date()} ~ {t.date.max().date()})")

    jobs = [(p, b) for p in combos for b in BASE_CANDIDATES]

    heaps = {k: [] for k in RANK_KEYS}
    seq = 0
    valid = 0
    chunk = []
    t0 = time.time()

    parquet_path = OUTPUT_DIR / "full_results_principal.parquet"
    writer = None

    def _flush():
        nonlocal writer
        if not chunk:
            return
        df = pd.DataFrame(chunk)
        chunk.clear()
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(str(parquet_path), table.schema)
        writer.write_table(table)

    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as ex:
        for ci, row in enumerate(ex.map(_eval, jobs, chunksize=2000), 1):
            if row is None:
                continue
            valid += 1
            seq += 1
            for k in RANK_KEYS:
                _push(heaps[k], row[k], seq, row)
            chunk.append(row)
            if len(chunk) >= CHUNK_ROWS:
                _flush()
            if ci % 50000 == 0:
                print(f"  进度 {ci}/{total} ({time.time()-t0:.0f}s)", flush=True)
    _flush()
    if writer is not None:
        writer.close()

    print(f"\n完成 {valid} 有效策略 ({time.time()-t0:.0f}s)\n")

    rankings = {}
    for k in RANK_KEYS:
        top = [it for (_, _, it) in heapq.nlargest(TOP_N, heaps[k], key=lambda t: t[0])]
        rankings[k] = {"label": RANK_LABELS[k], "key": k, "top": top}

    def _show(label, r):
        print(f"  [{label}]")
        print(f"    base={r['base_amount']}  {_fmt_params(r)}")
        print(f"    固定30万年化: min={r['unified_principal_annual']*100:.2f}% avg={r['avg_principal_annual']*100:.2f}% "
              f"| 300={r['000300_principal_annual']*100:.2f}% 500={r['000905_principal_annual']*100:.2f}%")
        print(f"    加权XIRR: {r['score_xirr_balanced']*100:.2f}% (min={r['unified_xirr']*100:.2f}% avg={r['avg_xirr']*100:.2f}%) "
              f"| 300投入={r['000300_invested']:.0f} 500投入={r['000905_invested']:.0f}")

    print("两种目标 Top1 对比:")
    _show("A 固定30万年化(min)", rankings["unified_principal_annual"]["top"][0])
    _show("B 加权XIRR", rankings["score_xirr_balanced"]["top"][0])

    out = {
        "codes": TRAIN_CODES,
        "names": TRAIN_NAMES,
        "min_trades": MIN_TRADES,
        "total_combos": total,
        "valid_combos": valid,
        "cost_model": {
            "commission_rate": COMMISSION_RATE,
            "min_commission": MIN_COMMISSION,
            "lot_size": 0,
            "note": "指数口径, 无整手限制; 佣金万5 低消5元, 双向",
        },
        "principal_cap": PRINCIPAL_CAP,
        "principal_pool": PRINCIPAL_POOL,
        "base_candidates": BASE_CANDIDATES,
        "note": "本金曲线=30万+累计盈亏; 年化=(本金/30万)^(365.25/天数)-1; 日收益=本金环比涨跌幅%",
        "rankings": rankings,
        "top": rankings["unified_principal_annual"]["top"],
    }
    with open(OUTPUT_DIR / "train_principal.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'train_principal.json'}")
    if parquet_path.exists():
        print(f"保存全量: {parquet_path}")


if __name__ == "__main__":
    main()
