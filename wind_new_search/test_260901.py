#!/usr/bin/env python3
"""
均衡策略 + 收割/底仓优化 测试集验证 — 生成 /wind-new/260901_update 网页数据。

新增规则 (参数来自 harvest_scan.py 扫描最优: 超额60%收割/收割20%/底仓20%):
  规则1 盈利收割: 持仓市值 > 净占用本金×1.6 时卖出超额利润20%, 抽出分配其他理财
  规则2 底仓保护: 任何卖出后持仓市值 ≥ 净占用本金×20%

输出 (独立, 不覆盖现有): wind_new_search/output/test_260901.json
结构:
  {title, params, harvest_params, cost_model, train_reference, harvest_results,
   balanced_results, compare: [{code,name,harvest: {...}, balanced: {...}, delta: {...}}],
   summary}
每指数含 sharpe (基于固定口径 daily_return 年化).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine_harvest import run_harvest, build_curve_harvest
from wind_new_search.engine import run_backtest, build_curve, prep_df

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
OUTPUT_DIR = PROJECT_DIR / "wind_new_search" / "output"

PARAMS = {
    "buy_signal": "PB", "buy_gate": "FED", "buy_gate_cap": 0.55,
    "sell_signal": "PE", "sell_gate": None, "sell_gate_floor": None,
    "buy_floor": 0.10, "buy_low": 0.15, "buy_mid": 0.25, "buy_high": 0.70,
    "sell_heavy": 0.85, "sell_extreme": 0.95,
}
MULTS = (8, 4, 2, 0)
BASE = 1000
THRESHOLD, CAP, POOL = 200_000, 300_000, 300_000
# 参数由训练集(沪深300/中证500)上细网格扫描选出: 按两指数平均年化最优+min稳健
HARVEST_PARAMS = {"profit_ratio": 0.30, "profit_frac": 0.30, "floor_ratio": 0.30}

TRAIN_CODES = ["000300", "000905"]
TEST_CODES = ["000015", "000016", "000852", "399006", "399330",
              "HSI", "NDX100", "SPX500", "930931", "930930", "000688", "HSTECH"]
NAMES = {
    "000300": "沪深300", "000905": "中证500", "000015": "上证红利", "000016": "上证50",
    "000852": "中证1000", "399006": "创业板指", "399330": "深证100", "HSI": "恒生指数",
    "NDX100": "纳斯达克100", "SPX500": "标普500", "930931": "港股通50", "930930": "港股综合",
    "000688": "科创50", "HSTECH": "恒生科技",
}


def _sharpe(curve):
    """从 build_curve 的 daily principal 曲线算年化夏普 (周频, 风险调整收益).

    用 daily_return (固定口径日环比%) → 月度/周收益 → 年化夏普 = mean/std×√52.
    """
    daily = curve["daily"]
    dr = [d.get("daily_return") for d in daily]
    dr = [x for x in dr if x is not None and not np.isnan(x)]
    if len(dr) < 8:
        return None
    # 聚合成周收益 (每5个交易日一笔) 降低自相关
    arr = np.array(dr)
    weeks = [arr[i:i + 5].sum() for i in range(0, len(arr), 5)]
    weeks = np.array(weeks) / 100.0
    if weeks.std() == 0:
        return None
    return round(float(weeks.mean() / weeks.std() * np.sqrt(52)), 3)


def backtest_harvest(code):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    window = int(df["window"].iloc[0])
    r = run_harvest(df, PARAMS, base_amount=BASE, commission_rate=0.0005,
                    min_commission=5.0, lot_size=0, principal_threshold=THRESHOLD,
                    principal_cap=CAP, principal_pool=POOL, buy_mults=MULTS,
                    **HARVEST_PARAMS)
    curve = build_curve_harvest(df, PARAMS, base_amount=BASE, commission_rate=0.0005,
                                min_commission=5.0, lot_size=0, principal_threshold=THRESHOLD,
                                principal_cap=CAP, principal_pool=POOL, buy_mults=MULTS,
                                **HARVEST_PARAMS)
    t = df[df["pe_pct"].notna()]
    years = (t["date"].max() - t["date"].min()).days / 365.25
    bh = (t["price"].iloc[-1] / t["price"].iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    return {
        "code": code, "name": NAMES.get(code, code), "window": window,
        "xirr": r["xirr"], "final_return": r["final_return"],
        "total_invested": r["total_invested"], "withdrawn": r["withdrawn"],
        "final_value": r["final_value"], "position_value": r["position_value"],
        "principal_final": r["principal_final"], "principal_annual": r["principal_annual"],
        "buys": r["buys"], "sells": r["sells"], "harvests": r["harvests"],
        "buy_hold_annual": round(bh, 4), "sharpe": _sharpe(curve),
    }


def backtest_balanced(code):
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    window = int(df["window"].iloc[0])
    r = run_backtest(prep_df(df), PARAMS, base_amount=BASE, commission_rate=0.0005,
                     min_commission=5.0, lot_size=0, principal_threshold=THRESHOLD,
                     principal_cap=CAP, principal_pool=POOL, buy_mults=MULTS)
    curve = build_curve(prep_df(df), PARAMS, base_amount=BASE, commission_rate=0.0005,
                        min_commission=5.0, lot_size=0, principal_threshold=THRESHOLD,
                        principal_cap=CAP, principal_pool=POOL, buy_mults=MULTS)
    t = df[df["pe_pct"].notna()]
    years = (t["date"].max() - t["date"].min()).days / 365.25
    bh = (t["price"].iloc[-1] / t["price"].iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    return {
        "code": code, "name": NAMES.get(code, code), "window": window,
        "xirr": r["xirr"], "final_return": r["final_return"],
        "total_invested": r["total_invested"],
        "final_value": r["final_value"], "position_value": r["position_value"],
        "principal_final": r["principal_final"], "principal_annual": r["principal_annual"],
        "buys": r["buys"], "sells": r["sells"],
        "buy_hold_annual": round(bh, 4), "sharpe": _sharpe(curve),
    }


def main():
    print("均衡策略 + 收割/底仓优化 测试集验证\n")
    train_h, train_b = [], []
    for code in TRAIN_CODES:
        h = backtest_harvest(code)
        b = backtest_balanced(code)
        train_h.append(h); train_b.append(b)
        print(f"{code} {h['name']:8} harvest年化{h['principal_annual']*100:6.2f}% "
              f"(夏普{h['sharpe']}) vs balanced {b['principal_annual']*100:.2f}% (训练)")

    print()
    ALL_CODES = TRAIN_CODES + TEST_CODES
    harvest_results, balanced_results, compare = [], [], []
    for code in ALL_CODES:
        h = backtest_harvest(code)
        b = backtest_balanced(code)
        harvest_results.append(h); balanced_results.append(b)
        dh = h["principal_annual"] - b["principal_annual"]
        dsharpe = (h["sharpe"] or 0) - (b["sharpe"] or 0)
        compare.append({"code": code, "name": h["name"], "window": h["window"],
                        "train": code in TRAIN_CODES,
                        "harvest": h, "balanced": b,
                        "delta_annual": round(dh, 4), "delta_sharpe": round(dsharpe, 4)})
        tag = "(训练)" if code in TRAIN_CODES else ""
        print(f"{code} {h['name']:8} harvest{h['principal_annual']*100:6.2f}% "
              f"vs balanced {b['principal_annual']*100:6.2f}%  Δ{dh*100:+5.2f}pp  "
              f"买{h['buys']}/卖{h['sells']}/收割{h['harvests']} 抽¥{h['withdrawn']:,.0f} {tag}")

    def _med(rows):
        vals = [r["principal_annual"] for r in rows if r["principal_annual"] > 0.005]
        return sorted(vals)[len(vals) // 2] if vals else 0
    def _med_sharpe(rows):
        vals = [r["sharpe"] for r in rows if r["sharpe"] is not None and r["principal_annual"] > 0.005]
        return sorted(vals)[len(vals) // 2] if vals else 0
    def _avg_withdrawn(rows):
        vals = [r["withdrawn"] for r in rows if r["principal_annual"] > 0.005]
        return sum(vals) / len(vals) if vals else 0

    summary = {
        "n_test": len(TEST_CODES),
        "harvest_annual_med": round(_med(harvest_results), 4),
        "balanced_annual_med": round(_med(balanced_results), 4),
        "harvest_sharpe_med": round(_med_sharpe(harvest_results), 4),
        "balanced_sharpe_med": round(_med_sharpe(balanced_results), 4),
        "harvest_avg_withdrawn": round(_avg_withdrawn(harvest_results), 0),
        "beats_annual": sum(1 for c in compare if c["delta_annual"] > 0),
        "beats_sharpe": sum(1 for c in compare if c["delta_sharpe"] > 0),
    }
    print("\n=== 汇总 ===")
    print(f"测试集固定年化中位: harvest {summary['harvest_annual_med']*100:.2f}% vs balanced {summary['balanced_annual_med']*100:.2f}%")
    print(f"夏普中位: harvest {summary['harvest_sharpe_med']:.2f} vs balanced {summary['balanced_sharpe_med']:.2f}")
    print(f"平均抽出利润: ¥{summary['harvest_avg_withdrawn']:,.0f}  跑赢年化 {summary['beats_annual']}/{len(TEST_CODES)}  跑赢夏普 {summary['beats_sharpe']}/{len(TEST_CODES)}")

    out = {
        "title": "均衡策略 + 收割/底仓优化 (260901)",
        "params": PARAMS, "buy_mults": list(MULTS), "base_amount": BASE,
        "principal_threshold": THRESHOLD, "principal_cap": CAP, "principal_pool": POOL,
        "harvest_params": HARVEST_PARAMS,
        "cost_model": {"commission_rate": 0.0005, "min_commission": 5.0, "lot_size": 0},
        "train_reference": {"harvest": train_h, "balanced": train_b},
        "harvest_results": harvest_results, "balanced_results": balanced_results,
        "compare": compare, "summary": summary,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "test_260901.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT_DIR / 'test_260901.json'}")


if __name__ == "__main__":
    main()
