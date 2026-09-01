#!/usr/bin/env python3
"""
ETF 实盘可行性回测 — 最优策略 B 在 12 只真实 ETF 上回测(自 ETF 上市/交易起点至今)。

与窗口分层回测一致: 信号仍来自指数 PE/PB/FED 百分位, 仅把执行价从指数点位
换成 ETF 后复权价, 并计入佣金 + 整手(100份)凑整。

费用:
  - 管理费/托管费: 已含在 ETF 后复权价中, 不再另扣
  - 佣金: 万2.5 最低5元 为基准, 另跑万1/万3 敏感度
  - 整手: 100 份, 不足整手跳过

对比: 同期指数口径(同时间段, 指数点位 + 同佣金, 不凑整) 作为参照。

用法: python wind_new_search/test_etf.py
输出: wind_new_search/output/test_etf.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import run_backtest
from wind_new_search.test_windowed import build_windowed_df, OPTIMAL_PARAMS, NAMES

ETF_DIR = PROJECT_DIR / "data-store" / "parquet" / "etf"
OUTPUT_DIR = PROJECT_DIR / "wind_new_search" / "output"

# 指数 code -> (ETF代码, ETF名称)
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

COMMISSION_PRESETS = {"w3": 0.0003, "w5": 0.0005, "w8": 0.0008}
BASE_COMMISSION = "w5"  # 佣金万5(0.05%), 最低5元/笔
MIN_COMMISSION = 5.0
LOT_SIZE = 100


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

        # 对齐 ETF 后复权价到指数周频日期
        aligned = pd.merge_asof(df[["date"]], etf[["date", "hfq"]],
                                on="date", direction="backward")
        mask = aligned["hfq"].notna()
        df_etf = df[mask].reset_index(drop=True)
        exec_price = aligned.loc[mask, "hfq"].to_numpy(float)

        # ETF 口径: 各佣金档
        sensitivity = {}
        for label, rate in COMMISSION_PRESETS.items():
            r = run_backtest(df_etf, OPTIMAL_PARAMS, exec_price=exec_price,
                             commission_rate=rate, min_commission=MIN_COMMISSION,
                             lot_size=LOT_SIZE)
            sensitivity[label] = round(r["xirr"], 4)

        # 基准 ETF 口径(万2.5)
        r_etf = run_backtest(df_etf, OPTIMAL_PARAMS, exec_price=exec_price,
                             commission_rate=COMMISSION_PRESETS[BASE_COMMISSION],
                             min_commission=MIN_COMMISSION, lot_size=LOT_SIZE)

        # 同期指数口径(万2.5, 不凑整) 参照
        r_idx = run_backtest(df_etf, OPTIMAL_PARAMS,
                             commission_rate=COMMISSION_PRESETS[BASE_COMMISSION],
                             min_commission=MIN_COMMISSION, lot_size=0)

        results.append({
            "code": idx_code, "name": NAMES.get(idx_code, idx_code),
            "etf_code": etf_code, "etf_name": etf_name,
            "window": cfg["window"],
            "start_date": str(df_etf["date"].min().date()),
            "end_date": str(df_etf["date"].max().date()),
            "rows": len(df_etf),
            "etf_xirr": r_etf["xirr"], "etf_return": r_etf["final_return"],
            "etf_buys": r_etf["buys"], "etf_sells": r_etf["sells"],
            "index_xirr": r_idx["xirr"], "index_return": r_idx["final_return"],
            "diff_xirr": round(r_etf["xirr"] - r_idx["xirr"], 4),
            "sensitivity": sensitivity,
        })
        print(f"  {idx_code:8s} {NAMES.get(idx_code, idx_code):8s} -> {etf_code} {etf_name}"
              f"  [{results[-1]['start_date']} ~ {results[-1]['end_date']}]"
              f"  ETF XIRR={r_etf['xirr']*100:6.2f}%  指数(同期)={r_idx['xirr']*100:6.2f}%"
              f"  差值={results[-1]['diff_xirr']*100:+5.2f}pp  买{r_etf['buys']}/卖{r_etf['sells']}")

    out = {
        "params": OPTIMAL_PARAMS,
        "cost_model": {
            "commission_rate": COMMISSION_PRESETS[BASE_COMMISSION],
            "commission_presets": COMMISSION_PRESETS,
            "min_commission": MIN_COMMISSION,
            "lot_size": LOT_SIZE,
            "note": "管理费/托管费已含在 ETF 后复权价中, 不另扣; 佣金双向; ETF 免印花税/过户费",
        },
        "results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "test_etf.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'test_etf.json'}")


if __name__ == "__main__":
    main()
