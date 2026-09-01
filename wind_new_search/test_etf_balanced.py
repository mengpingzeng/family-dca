#!/usr/bin/env python3
"""
均衡策略 ETF 验证 — 用均衡策略在 12 只真实 ETF 上回测。

信号口径与训练集完全一致 (wind_new_merged 预计算 pe/pb/fed 百分位),
仅把执行价换成 ETF 后复权价, 计入佣金万5低消5 + 整手100份。

均衡策略: PB主/FED<=55%闸/卖PE(S85/95) + 买档8/4/2/0 + buy_mid=0.25 + base=1000
口径: 阈值20万收缩 + 封顶30万 + 固定30万

输出 (独立): wind_new_search/output/test_etf_balanced.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import run_backtest, prep_df

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
ETF_DIR = PROJECT_DIR / "data-store" / "parquet" / "etf"
OUTPUT_DIR = PROJECT_DIR / "wind_new_search" / "output"

BALANCED_PARAMS = {
    "buy_signal": "PB", "buy_gate": "FED", "buy_gate_cap": 0.55,
    "sell_signal": "PE", "sell_gate": None, "sell_gate_floor": None,
    "buy_floor": 0.10, "buy_low": 0.15, "buy_mid": 0.25, "buy_high": 0.70,
    "sell_heavy": 0.85, "sell_extreme": 0.95,
}
BALANCED_MULTS = (8, 4, 2, 0)
BASE = 1000
COMMISSION_RATE = 0.0005
MIN_COMMISSION = 5.0
LOT_SIZE = 100
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
NAMES = {
    "000300": "沪深300", "000905": "中证500", "000015": "上证红利", "000016": "上证50",
    "000852": "中证1000", "399006": "创业板指", "399330": "深证100", "HSI": "恒生指数",
    "NDX100": "纳斯达克100", "SPX500": "标普500", "930931": "港股通50", "930930": "港股综合",
    "000688": "科创50", "HSTECH": "恒生科技",
}


def backtest_etf(idx_code):
    df = pd.read_parquet(MERGED_DIR / f"{idx_code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    etf_code, etf_name = ETF_MAP[idx_code]
    etf = pd.read_parquet(ETF_DIR / f"{etf_code}.parquet")
    etf["date"] = pd.to_datetime(etf["date"]).astype("datetime64[ns]")
    df["date"] = df["date"].astype("datetime64[ns]")

    aligned = pd.merge_asof(df[["date"]], etf[["date", "hfq"]], on="date", direction="backward")
    mask = aligned["hfq"].notna() & df["pe_pct"].notna()
    df_etf = df[mask].reset_index(drop=True)
    exec_price = aligned.loc[mask, "hfq"].to_numpy(float)

    if len(df_etf) < 2:
        return None

    bt_df = prep_df(df_etf)
    r = run_backtest(bt_df, BALANCED_PARAMS, base_amount=BASE, exec_price=exec_price,
                     commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                     lot_size=LOT_SIZE, principal_threshold=THRESHOLD, principal_cap=CAP,
                     principal_pool=POOL, buy_mults=BALANCED_MULTS)

    start_date = df_etf["date"].min()
    end_date = df_etf["date"].max()
    years = (end_date - start_date).days / 365.25
    first_hfq = exec_price[0]
    last_hfq = exec_price[-1]
    bh = (last_hfq / first_hfq) ** (1 / years) - 1 if years > 0 and first_hfq > 0 else 0.0

    return {
        "code": idx_code, "name": NAMES.get(idx_code, idx_code),
        "etf_code": etf_code, "etf_name": etf_name,
        "start_date": str(start_date.date()), "end_date": str(end_date.date()),
        "rows": len(df_etf),
        "xirr": r["xirr"], "final_return": r["final_return"],
        "total_invested": r["total_invested"], "final_value": r["final_value"],
        "principal_final": r["principal_final"], "principal_annual": r["principal_annual"],
        "principal_return": r["principal_return"],
        "buys": r["buys"], "sells": r["sells"],
        "buy_hold_annual": round(bh, 4),
    }


def main():
    print("均衡策略 ETF 验证 (8/4/2/0 mid=0.25, 阈值20万+封顶30万, 整手100, 万5低消5)\n")
    print(f"{'ETF代码':8} {'名称':18} | {'固定年化':>8} {'XIRR':>8} {'满仓年化':>8} {'终值':>10} {'买/卖':>8}")
    print("-" * 95)
    results = []
    for idx_code, (etf_code, etf_name) in ETF_MAP.items():
        r = backtest_etf(idx_code)
        if r is None:
            print(f"[跳过] {idx_code} 数据不足")
            continue
        results.append(r)
        tag = "训练" if idx_code in ("000300", "000905") else ""
        print(f"{etf_code:8} {etf_name:18} | {r['principal_annual']*100:7.2f}% {r['xirr']*100:7.2f}% {r['buy_hold_annual']*100:7.2f}% {r['principal_final']:>10,.0f} {r['buys']:>3}/{r['sells']:<3}  {tag}")

    test = [r for r in results if r["code"] not in ("000300", "000905")]
    if test:
        anns = [r["principal_annual"] for r in test]
        xirrs = [r["xirr"] for r in test]
        beats = sum(1 for r in test if r["principal_annual"] > r["buy_hold_annual"])
        print("-" * 95)
        print(f"ETF 测试 10 只: 固定年化 均值={sum(anns)/len(anns)*100:.2f}% 中位={sorted(anns)[len(anns)//2]*100:.2f}% "
              f"min={min(anns)*100:.2f}% max={max(anns)*100:.2f}%")
        print(f"               XIRR      均值={sum(xirrs)/len(xirrs)*100:.2f}% 中位={sorted(xirrs)[len(xirrs)//2]*100:.2f}%")
        print(f"跑赢满仓持有: {beats}/{len(test)} 只")

    out = {
        "title": "均衡策略 ETF 验证",
        "params": BALANCED_PARAMS, "buy_mults": list(BALANCED_MULTS), "base_amount": BASE,
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "cost_model": {"commission_rate": COMMISSION_RATE, "min_commission": MIN_COMMISSION,
                       "lot_size": LOT_SIZE, "note": "管理费/托管费已含在ETF后复权价中; 佣金双向; ETF免印花税"},
        "results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "test_etf_balanced.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'test_etf_balanced.json'}")


if __name__ == "__main__":
    main()
