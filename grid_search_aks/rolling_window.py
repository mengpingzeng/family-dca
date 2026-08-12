#!/usr/bin/env python3
"""
滚动窗口回测 —— 固定策略，滑动起点，固定持有期，评估鲁棒性

用法:
  python3 rolling_window.py
  python3 rolling_window.py --sample 100

输出: rolling_analysis.json (供前端) + rolling_analysis.csv (Excel 导入)
"""

import os, sys, json, itertools, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_search import run_backtest, INDICES_ALL

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MERGED_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data-store", "parquet", "aks_merged")
MERGED_DIR = os.path.abspath(MERGED_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_AMOUNT = 500
HOLD_YEARS = [3, 5, 10]
SLIDE_MONTHS = 3  # quarterly sliding for performance, monthly for final

# 固定策略参数（当前严格买入最优）
FIXED_PARAMS = {
    "buy_floor": 0.08, "buy_low": 0.12, "buy_mid": 0.22, "buy_high": 0.40,
    "sell_heavy": 0.75, "sell_extreme": 0.85,
    "fed_gate": None, "pb_veto": None, "pb_sell": None,
}


def generate_windows(df: pd.DataFrame, hold_years: int, slide_months: int = 1):
    """生成所有可用的 (start_date, end_date) 滑动窗口。"""
    date_col = df["date"]
    pe_col = [c for c in df.columns if c.startswith("pe_pct_w")][0]
    valid = df[df[pe_col].notna()]
    if len(valid) < 500:
        return []

    min_date = valid["date"].iloc[0]
    max_date = valid["date"].iloc[-1]

    windows = []
    cur = pd.Timestamp(min_date)
    end_limit = max_date - pd.DateOffset(years=hold_years)
    while cur <= end_limit:
        end = cur + pd.DateOffset(years=hold_years)
        if end <= max_date:
            windows.append((str(cur)[:10], str(end)[:10]))
        cur += pd.DateOffset(months=slide_months)
    return windows


def run_rolling_analysis(codes=None, hold_years=None, n_sample=0):
    """对每个指数 × 每个持有期 × 每个窗口, 运行回测。"""
    if codes is None:
        codes = ["000300", "000016", "000905"]
    if hold_years is None:
        hold_years = HOLD_YEARS

    names = {c: INDICES_ALL[c] for c in codes if c in INDICES_ALL}

    results = {}
    for code in codes:
        name = names.get(code, code)
        path = os.path.join(MERGED_DIR, f"{code}.parquet")
        if not os.path.exists(path):
            print(f"  跳过 {name}：数据不存在")
            continue

        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])

        index_results = {}
        for hy in hold_years:
            w_list = generate_windows(df, hy, slide_months=SLIDE_MONTHS)
            if n_sample and n_sample < len(w_list):
                import random
                w_list = random.sample(w_list, n_sample)

            rows = []
            t0 = time.time()
            for i, (start, end) in enumerate(w_list):
                mask = (df["date"] >= start) & (df["date"] <= end)
                sub = df[mask].reset_index(drop=True)
                if len(sub) < 250 * hy * 0.5:
                    continue
                try:
                    r = run_backtest(sub, FIXED_PARAMS, hy,
                                     base_amount=BASE_AMOUNT,
                                     idle_cash_rate=0.02,
                                     min_trades=5)
                    rows.append({
                        "start": start, "end": end,
                        "hold_years": hy,
                        "xirr": r["xirr"], "final_return": r["final_return"],
                        "total_invested": int(r["total_invested"]),
                        "total_cash_in": int(r["total_cash_in"]),
                        "final_value": int(r["final_value"]),
                        "trades": r["trades"], "buys": r["buys"], "sells": r["sells"],
                    })
                except Exception:
                    continue

            rows.sort(key=lambda x: x["xirr"], reverse=True)
            elapsed = time.time() - t0

            if rows:
                xirrs = [r["xirr"] for r in rows]
                xirrs_arr = np.array(xirrs)
                index_results[f"hold_{hy}yr"] = {
                    "windows": len(rows),
                    "xirr_min": round(float(np.min(xirrs_arr)) * 100, 2),
                    "xirr_max": round(float(np.max(xirrs_arr)) * 100, 2),
                    "xirr_mean": round(float(np.mean(xirrs_arr)) * 100, 2),
                    "xirr_median": round(float(np.median(xirrs_arr)) * 100, 2),
                    "xirr_std": round(float(np.std(xirrs_arr)) * 100, 2),
                    "win_rate": round(np.sum(xirrs_arr > 0) / len(xirrs) * 100, 1),
                    "rows": rows,
                }
                top = rows[0]
                worst = rows[-1]
                print(f"  {name} {hy}yr: {len(rows)}窗口 "
                      f"min={xirrs_arr.min()*100:.1f}% median={np.median(xirrs_arr)*100:.1f}% "
                      f"max={xirrs_arr.max()*100:.1f}% 胜率={np.sum(xirrs_arr>0)/len(xirrs)*100:.0f}% "
                      f"({elapsed:.1f}s)")
            else:
                index_results[f"hold_{hy}yr"] = {"windows": 0, "rows": []}
                print(f"  {name} {hy}yr: 0 窗口")

        results[code] = {"name": name, "params": FIXED_PARAMS, "holds": index_results}

    return results


def dump_for_frontend(results, out_json):
    """输出前端友好的 JSON（去掉 row 详情，改用滚动 XIRR 时序）。"""
    front = {}
    for code, v in results.items():
        front[code] = {"name": v["name"], "params": v["params"], "holds": {}}
        for hk, hv in v["holds"].items():
            summary = {k: hv[k] for k in hv if k != "rows"}
            # 抽取 xirr 时序用于画图
            xirr_series = [{"start": r["start"], "end": r["end"], "xirr": r["xirr"],
                            "total_invested": r["total_invested"],
                            "final_value": r["final_value"],
                            "trades": r["trades"]} for r in hv.get("rows", [])]
            summary["xirr_series"] = xirr_series
            front[code]["holds"][hk] = summary
    with open(out_json, "w") as f:
        json.dump(front, f, indent=2, ensure_ascii=False)
    print(f"前端 JSON: {out_json}")


def dump_csv(results, out_csv):
    """输出 CSV (方便 Excel 透视)。"""
    rows = []
    for code, v in results.items():
        for hk, hv in v["holds"].items():
            for r in hv.get("rows", []):
                rows.append({
                    "指数": v["name"], "代码": code, "持有期": f'{hv["rows"] and r["hold_years"]}yr',
                    "起点": r["start"], "终点": r["end"],
                    "XIRR": round(r["xirr"] * 100, 2),
                    "总投入": r["total_invested"],
                    "终值": r["final_value"],
                    "交易": r["trades"],
                })
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"CSV: {out_csv}")
    else:
        print("无数据可输出 CSV")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", type=str, nargs="*", default=None)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("滚动窗口鲁棒性分析")
    print(f"固定策略: B{FIXED_PARAMS['buy_floor']}/{FIXED_PARAMS['buy_low']}/"
          f"{FIXED_PARAMS['buy_mid']}/{FIXED_PARAMS['buy_high']} "
          f"S{FIXED_PARAMS['sell_heavy']}/{FIXED_PARAMS['sell_extreme']} "
          f"FED=off PBv=off")
    print(f"Base={BASE_AMOUNT} 闲置2% 持有期={HOLD_YEARS}yr 滑动={SLIDE_MONTHS}月")
    print("=" * 60)

    results = run_rolling_analysis(args.codes, n_sample=args.sample)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"rolling_analysis_{ts}.json")
    dump_for_frontend(results, json_path)

    latest_json = os.path.join(OUTPUT_DIR, "latest_rolling.json")
    dump_for_frontend(results, latest_json)

    csv_path = os.path.join(OUTPUT_DIR, f"rolling_analysis_{ts}.csv")
    dump_csv(results, csv_path)


if __name__ == "__main__":
    main()
