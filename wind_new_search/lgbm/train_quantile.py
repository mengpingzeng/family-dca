#!/usr/bin/env python3
"""
LGBM 分位数模型 — walk-forward 训练 + 三场景评估.

在合并数据集(沪深300+中证500)上, 用滚动窗口 walk-forward 训练 P10/P50/P90 分位数模型,
输出预测区间 [P10, P90], 并按三场景统计:
  偏正: P10 > 0   (下跌空间有限, 上涨可观 -> 积极)
  偏负: P90 < 0   (可能下跌, 上涨有限 -> 减仓/暂停)
  观望: 其余      (区间跨 0 或过宽 -> 观望)

评估: 区间覆盖率 + 三场景实际收益分布 + 与常数基线对比.

输出(独立): wind_new_search/lgbm/output/quantile_result.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "output"

FEATURES = ["pe_pct", "pb_pct", "fed_pct", "vol", "is_hs300"]
TARGET = "y"
ALPHAS = [0.1, 0.5, 0.9]
TRAIN_WINDOW_YEARS = 10
FWD = int(sys.argv[1]) if len(sys.argv) > 1 else 12
DATA_PATH = BASE / "data" / f"lgbm_dataset_{FWD}w.parquet"


def params_for(alpha):
    return {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": 0.05,
        "num_leaves": 8,
        "max_depth": 3,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 42,
    }


def main():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    test_years = list(range(2016, 2026))
    preds_all = []
    models_trained = 0

    for year in test_years:
        train = df[(df["date"] >= pd.Timestamp(f"{year - TRAIN_WINDOW_YEARS}-01-01")) &
                   (df["date"] < pd.Timestamp(f"{year}-01-01"))].reset_index(drop=True)
        test = df[(df["date"] >= pd.Timestamp(f"{year}-01-01")) &
                  (df["date"] < pd.Timestamp(f"{year + 1}-01-01"))].reset_index(drop=True)
        if len(train) < 60 or len(test) == 0:
            continue

        Xtr, ytr = train[FEATURES].values, train[TARGET].values
        Xte = test[FEATURES].values
        ds = lgb.Dataset(Xtr, label=ytr)

        preds = {}
        for a in ALPHAS:
            m = lgb.train(params_for(a), ds, num_boost_round=150)
            preds[a] = m.predict(Xte)
            models_trained += 1

        for j in range(len(test)):
            preds_all.append({
                "date": str(test["date"].iloc[j].date()),
                "year": year,
                "is_hs300": int(test["is_hs300"].iloc[j]),
                "pe_pct": float(test["pe_pct"].iloc[j]),
                "pb_pct": float(test["pb_pct"].iloc[j]),
                "fed_pct": float(test["fed_pct"].iloc[j]),
                "vol": float(test["vol"].iloc[j]),
                "actual": float(test[TARGET].iloc[j]),
                "p10": float(preds[0.1][j]),
                "p50": float(preds[0.5][j]),
                "p90": float(preds[0.9][j]),
            })

    P = pd.DataFrame(preds_all)
    print(f"walk-forward 完成: 共 {models_trained} 个模型, 样本外预测 {len(P)} 条")

    # ---- 评估 ----
    # 1. 区间覆盖率
    cover = ((P["actual"] >= P["p10"]) & (P["actual"] <= P["p90"])).mean()
    # 2. 区间平均宽度
    P["width"] = P["p90"] - P["p10"]
    # 3. 三场景
    P["scenario"] = np.where(P["p10"] > 0, "偏正",
                    np.where(P["p90"] < 0, "偏负", "观望"))

    print("\n=== 1. 区间质量 ===")
    print(f"样本外样本数: {len(P)}")
    print(f"区间覆盖率 [P10,P90]: {cover*100:.1f}% (理想 ~80%)")
    print(f"区间平均宽度: {P['width'].mean()*100:.2f}%  中位 {P['width'].median()*100:.2f}%")
    print(f"实际收益: 均值 {P['actual'].mean()*100:.2f}%  中位 {P['actual'].median()*100:.2f}%")
    corr = float(np.corrcoef(P["p50"], P["actual"])[0, 1])
    print(f"P50 与实际相关系数: {corr:.3f}")

    print("\n=== 2. 三场景分布与实际收益 ===")
    scen_stats = {}
    for s in ["偏正", "偏负", "观望"]:
        sub = P[P["scenario"] == s]
        if len(sub) == 0:
            print(f"{s}: 0 条")
            continue
        pos = (sub["actual"] > 0).mean()
        scen_stats[s] = {
            "n": int(len(sub)),
            "actual_mean": round(float(sub["actual"].mean()), 4),
            "actual_median": round(float(sub["actual"].median()), 4),
            "hit_rate_pos": round(float(pos), 4),
            "mean_width": round(float(sub["width"].mean()), 4),
        }
        print(f"{s}: {len(sub):>4} 条 ({len(sub)/len(P)*100:4.1f}%)  "
              f"实际均值 {sub['actual'].mean()*100:6.2f}%  中位 {sub['actual'].median()*100:6.2f}%  "
              f"上涨占比 {pos*100:4.1f}%  平均宽度 {sub['width'].mean()*100:5.1f}%")

    # 4. 常数基线: 用全样本 P10/P50/P90 常数预测
    print("\n=== 3. 与常数基线对比 ===")
    q10, q50, q90 = df[TARGET].quantile(0.1), df[TARGET].quantile(0.5), df[TARGET].quantile(0.9)
    print(f"常数基线: P10={q10*100:.2f}%  P50={q50*100:.2f}%  P90={q90*100:.2f}% (全样本分位)")
    base_cover = ((P["actual"] >= q10) & (P["actual"] <= q90)).mean()
    print(f"常数区间覆盖率: {base_cover*100:.1f}% (模型 {cover*100:.1f}%)")
    # 常数基线宽度
    base_width = q90 - q10
    print(f"常数区间宽度: {base_width*100:.2f}% (模型平均 {P['width'].mean()*100:.2f}%)")

    # 5. 分年覆盖率
    print("\n=== 4. 分年区间覆盖率 ===")
    for y in test_years:
        sub = P[P["year"] == y]
        if len(sub) == 0:
            continue
        c = ((sub["actual"] >= sub["p10"]) & (sub["actual"] <= sub["p90"])).mean()
        print(f"  {y}: {len(sub):>3} 条  覆盖率 {c*100:5.1f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "title": "LGBM 分位数模型 walk-forward 结果",
        "features": FEATURES, "target": TARGET, "alphas": ALPHAS,
        "fwd_weeks": FWD,
        "train_window_years": TRAIN_WINDOW_YEARS, "test_years": test_years,
        "n_models": models_trained, "n_preds": len(P),
        "coverage": round(float(cover), 4),
        "mean_width": round(float(P["width"].mean()), 4),
        "median_width": round(float(P["width"].median()), 4),
        "corr_p50_actual": round(corr, 4),
        "scenario_stats": scen_stats,
        "scenario_dist": P["scenario"].value_counts().to_dict(),
        "baseline": {
            "q10": round(float(q10), 4), "q50": round(float(q50), 4), "q90": round(float(q90), 4),
            "coverage": round(float(base_cover), 4), "width": round(float(base_width), 4),
        },
        "params": params_for(0.5),
        "preds": preds_all,
    }
    with open(OUT_DIR / f"quantile_result_{FWD}w.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUT_DIR / f'quantile_result_{FWD}w.json'}")


if __name__ == "__main__":
    main()
