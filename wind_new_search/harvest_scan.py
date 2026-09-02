#!/usr/bin/env python3
"""
收割策略 参数扫描 — profit_ratio × profit_frac × floor_ratio 网格, 对比 balanced 基线。

用于在 12 测试集指数 + 2 训练集指数上评估各参数组合的固定年化, 选出稳健参数。
只读数据, 输出独立 JSON (output/harvest_scan.json), 不覆盖现有文件。
"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine_harvest import run_harvest
from wind_new_search.engine import run_backtest, prep_df

MERGED = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
OUTPUT = PROJECT_DIR / "wind_new_search" / "output"

TRAIN_CODES = ["000300", "000905"]
TEST_CODES = ["000015", "000016", "000852", "399006", "399330",
              "HSI", "NDX100", "SPX500", "930931", "930930", "000688", "HSTECH"]
NAMES = {
    "000300": "沪深300", "000905": "中证500", "000015": "上证红利", "000016": "上证50",
    "000852": "中证1000", "399006": "创业板指", "399330": "深证100", "HSI": "恒生指数",
    "NDX100": "纳斯达克100", "SPX500": "标普500", "930931": "港股通50", "930930": "港股综合",
    "000688": "科创50", "HSTECH": "恒生科技",
}

PARAMS = {
    "buy_signal": "PB", "buy_gate": "FED", "buy_gate_cap": 0.55,
    "sell_signal": "PE", "sell_gate": None, "sell_gate_floor": None,
    "buy_floor": 0.10, "buy_low": 0.15, "buy_mid": 0.25, "buy_high": 0.70,
    "sell_heavy": 0.85, "sell_extreme": 0.95,
}
MULTS = (8, 4, 2, 0)
BASE = 1000
THRESHOLD, CAP, POOL = 200_000, 300_000, 300_000


def load(code):
    df = pd.read_parquet(MERGED / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def baseline(code):
    df = load(code)
    df = prep_df(df)
    r = run_backtest(df, PARAMS, base_amount=BASE, commission_rate=0.0005,
                     min_commission=5.0, lot_size=0, principal_threshold=THRESHOLD,
                     principal_cap=CAP, principal_pool=POOL, buy_mults=MULTS)
    return r["principal_annual"]


def harvest(code, pr, pf, fr):
    df = load(code)
    r = run_harvest(df, PARAMS, base_amount=BASE, commission_rate=0.0005,
                    min_commission=5.0, lot_size=0, principal_threshold=THRESHOLD,
                    principal_cap=CAP, principal_pool=POOL, buy_mults=MULTS,
                    profit_ratio=pr, profit_frac=pf, floor_ratio=fr)
    return r


def main():
    ratios = [0.30, 0.40, 0.50, 0.60]
    fracs = [0.15, 0.20, 0.30]
    floors = [0.15, 0.20, 0.30]
    codes = TRAIN_CODES + TEST_CODES

    # 基线
    print("基线 balanced 固定年化:")
    bl = {}
    for c in codes:
        bl[c] = baseline(c)
        print(f"  {c} {NAMES[c]:8} {bl[c]*100:6.2f}%")
    print()

    # 扫描
    rows = []
    total = len(ratios) * len(fracs) * len(floors)
    idx = 0
    for pr in ratios:
        for pf in fracs:
            for fr in floors:
                idx += 1
                train_vals = []
                test_vals = []
                for c in TRAIN_CODES:
                    r = harvest(c, pr, pf, fr)
                    train_vals.append(r["principal_annual"])
                for c in TEST_CODES:
                    r = harvest(c, pr, pf, fr)
                    test_vals.append(r["principal_annual"])
                # 有效指数 (剔除 0 买入的美股成长)
                teff = [v for v in test_vals if v > 0.005]
                tmed = sorted(test_vals)[len(test_vals)//2]
                train_med = sorted(train_vals)[len(train_vals)//2]
                rows.append({
                    "profit_ratio": pr, "profit_frac": pf, "floor_ratio": fr,
                    "train_med": round(train_med, 4),
                    "test_med": round(tmed, 4),
                    "test_eff_mean": round(sum(teff)/len(teff), 4) if teff else 0,
                    "beats_base": sum(1 for v in test_vals if v > bl.get(c, 0) and False),  # placeholder
                })
                print(f"[{idx}/{total}] pr={pr:.2f} pf={pf:.2f} fr={fr:.2f} "
                      f"train_med={train_med*100:5.2f}% test_med={tmed*100:5.2f}%")

    # 汇总: 按 test_med 排序
    rows.sort(key=lambda x: -x["test_med"])
    print("\n=== 参数扫描汇总 (按测试集固定年化中位排序) ===")
    print(f"{'pr':>5}{'pf':>6}{'fr':>6}{'train_med':>10}{'test_med':>10}{'test_eff':>10}")
    for r in rows[:15]:
        print(f"{r['profit_ratio']:>5.2f}{r['profit_frac']:>6.2f}{r['floor_ratio']:>6.2f}"
              f"{r['train_med']*100:>9.2f}%{r['test_med']*100:>9.2f}%{r['test_eff_mean']*100:>9.2f}%")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT / "harvest_scan.json", "w") as f:
        json.dump({"params": {"ratios": ratios, "fracs": fracs, "floors": floors},
                   "baseline": bl, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUTPUT / 'harvest_scan.json'}")


if __name__ == "__main__":
    main()
