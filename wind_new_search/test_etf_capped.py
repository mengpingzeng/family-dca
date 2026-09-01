#!/usr/bin/env python3
"""
ETF 本金阈值回测 — 策略 B + 本金阈值 vs 策略 B 原版 (12 只真实 ETF)。

本金阈值规则 (单只 ETF 的累计买入本金, 含佣金):
  < 30万          : 0.5x / 1x / 2x / 3x (原样)
  30万 ~ 36万     : 1x / 2x / 3x        (去掉 0.5x)
  36万 ~ 42万     : 2x / 3x             (去掉 1x)
  > 42万          : 仅 3x               (去掉 2x)

目的: 修正"某些 ETF 投入很大但收益不成正比"的资金分布不均问题 —
      本金累计越多, 只在更强的便宜信号(更高倍数)时才继续加仓。

费用: 佣金万5 最低5元/笔(双向), 整手 100 份; 管理费/托管费已含在后复权价中。

用法: python wind_new_search/test_etf_capped.py
输出: wind_new_search/output/test_etf_capped.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import run_backtest
from wind_new_search.test_windowed import build_windowed_df, OPTIMAL_PARAMS, NAMES

ETF_DIR = PROJECT_DIR / "data-store" / "parquet" / "etf"
OUTPUT_DIR = PROJECT_DIR / "wind_new_search" / "output"

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

COMMISSION_RATE = 0.0005  # 万5
MIN_COMMISSION = 5.0
LOT_SIZE = 100

PRINCIPAL_THRESHOLD = 300_000  # 30万


def main():
    results = []
    for idx_code, (etf_code, etf_name) in ETF_MAP.items():
        df, cfg = build_windowed_df(idx_code)
        if df is None:
            print(f"[跳过] {idx_code} 无指数数据")
            continue
        etf_path = ETF_DIR / f"{etf_code}.parquet"
        if not etf_path.exists():
            print(f"[跳过] {idx_code} 无 ETF 数据 {etf_code}")
            continue
        etf = pd.read_parquet(etf_path)
        etf["date"] = pd.to_datetime(etf["date"]).astype("datetime64[ns]")
        df["date"] = df["date"].astype("datetime64[ns]")

        aligned = pd.merge_asof(df[["date"]], etf[["date", "hfq"]],
                                on="date", direction="backward")
        mask = aligned["hfq"].notna()
        df_etf = df[mask].reset_index(drop=True)
        exec_price = aligned.loc[mask, "hfq"].to_numpy(float)

        # 原版 B (无本金阈值)
        r_base = run_backtest(df_etf, OPTIMAL_PARAMS, exec_price=exec_price,
                              commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                              lot_size=LOT_SIZE)
        # B + 本金阈值
        r_capped = run_backtest(df_etf, OPTIMAL_PARAMS, exec_price=exec_price,
                                commission_rate=COMMISSION_RATE, min_commission=MIN_COMMISSION,
                                lot_size=LOT_SIZE, principal_threshold=PRINCIPAL_THRESHOLD)

        results.append({
            "code": idx_code, "name": NAMES.get(idx_code, idx_code),
            "etf_code": etf_code, "etf_name": etf_name,
            "window": cfg["window"],
            "start_date": str(df_etf["date"].min().date()),
            "end_date": str(df_etf["date"].max().date()),
            "rows": len(df_etf),
            "base_xirr": r_base["xirr"], "base_return": r_base["final_return"],
            "base_invested": r_base["total_invested"], "base_buys": r_base["buys"],
            "base_sells": r_base["sells"],
            "capped_xirr": r_capped["xirr"], "capped_return": r_capped["final_return"],
            "capped_invested": r_capped["total_invested"], "capped_buys": r_capped["buys"],
            "capped_sells": r_capped["sells"],
            "diff_xirr": round(r_capped["xirr"] - r_base["xirr"], 4),
            "diff_invested": round(r_capped["total_invested"] - r_base["total_invested"], 0),
        })
        print(f"  {idx_code:8s} {NAMES.get(idx_code, idx_code):8s} -> {etf_code} {etf_name}")
        print(f"    原版 B : XIRR={r_base['xirr']*100:6.2f}%  投入={r_base['total_invested']:>9.0f}  买{r_base['buys']}/卖{r_base['sells']}")
        print(f"    B+阈值 : XIRR={r_capped['xirr']*100:6.2f}%  投入={r_capped['total_invested']:>9.0f}  买{r_capped['buys']}/卖{r_capped['sells']}"
              f"  ΔXIRR={results[-1]['diff_xirr']*100:+5.2f}pp  Δ投入={results[-1]['diff_invested']:+.0f}")

    out = {
        "params": OPTIMAL_PARAMS,
        "principal_threshold": {
            "threshold": PRINCIPAL_THRESHOLD,
            "tiers": [
                {"up_to": 300000, "min_mult": 0.5, "desc": "0.5x/1x/2x/3x"},
                {"up_to": 360000, "min_mult": 1.0, "desc": "1x/2x/3x"},
                {"up_to": 420000, "min_mult": 2.0, "desc": "2x/3x"},
                {"up_to": None, "min_mult": 3.0, "desc": "仅3x"},
            ],
            "note": "累计买入本金(含佣金) 达30万只做1x/2x/3x; 36万后只2x/3x; 42万后只3x",
        },
        "cost_model": {
            "commission_rate": COMMISSION_RATE,
            "min_commission": MIN_COMMISSION,
            "lot_size": LOT_SIZE,
            "note": "管理费/托管费已含在 ETF 后复权价中, 不另扣; 佣金双向; ETF 免印花税/过户费",
        },
        "results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "test_etf_capped.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'test_etf_capped.json'}")


if __name__ == "__main__":
    main()
