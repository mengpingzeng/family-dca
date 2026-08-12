#!/usr/bin/env python3
"""
宽基统一策略网格搜索 v1 — 2026-08-09
======================================
目标: 针对7个宽基指数，所有宽基使用同一套策略参数，遍历:
  - 窗口年限: 6yr, 8yr, 10yr
  - 特征组合: PE_only, PE_PB, PE_FED, PE_PB_FED, PB_only, PB_FED
  - 权重: (1.0,0.0), (0.8,0.2), (0.6,0.4)
  - 策略参数: 每 (特征组合, 窗口) 随机采样 1000 组

评分: mean_i(avg_xirr_i * w1 - sigma_xirr_i * w2)
  avg_xirr_i  = 运行XIRR年化序列的均值
  sigma_xirr_i = 运行XIRR年化序列的标准差

XIRR计算: <1年用简单年化收益率, >=1年用二分法XIRR

输出: CSV结果 + HTML可视化报告
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =========================== 路径配置 ===========================
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "backtest"))

from backtest import vec_rolling_pct, vec_rolling_mean_std, calc_xirr

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "merged"
PRICE_DIR = PROJECT_DIR / "data-store" / "parquet" / "index_price"
OUTPUT_DIR = PROJECT_DIR / "grid_search" / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================== 常量 ===========================
CODES = ["000300", "000905", "000852", "000016", "000688", "399006", "399330"]
CODE_NAMES = {
    "000300": "沪深300", "000905": "中证500", "000852": "中证1000",
    "000016": "上证50", "000688": "科创50", "399006": "创业板指",
    "399330": "深证100",
}
WINDOWS = [6, 8, 10]
WEIGHTS = [(1.0, 0.0), (0.8, 0.2), (0.6, 0.4)]
N_SAMPLE = 1000
BASE_AMOUNT = 1500
RANDOM_SEED = 42

FEATURE_COMBOS = {
    "PE_only":    {"primary": "PE", "fed_gate": False, "pb_veto": False},
    "PE_PB":      {"primary": "PE", "fed_gate": False, "pb_veto": True},
    "PE_FED":     {"primary": "PE", "fed_gate": True,  "pb_veto": False},
    "PE_PB_FED":  {"primary": "PE", "fed_gate": True,  "pb_veto": True},
    "PB_only":    {"primary": "PB", "fed_gate": False, "pb_veto": False},
    "PB_FED":     {"primary": "PB", "fed_gate": True,  "pb_veto": False},
}

# =========================== 参数采样 ===========================
BUY_FLOORS   = [0.05, 0.10, 0.15, 0.20]
BUY_LOWS     = [0.15, 0.20, 0.25, 0.30]
BUY_MIDS     = [0.30, 0.35, 0.40, 0.45]
BUY_HIGHS    = [0.50, 0.55, 0.60, 0.65, 0.70]
SELL_WARNS   = [0.65, 0.70, 0.75]
SELL_HEAVYS  = [0.75, 0.80, 0.85]
SELL_EXTS    = [0.85, 0.90, 0.95]
FEDS         = [-0.5, 0.0, 0.5, 1.0]
VETOS        = [0.50, 0.55, 0.60, 0.65]
CONFIRMS     = [0.70, 0.75, 0.80]
DD_STDS      = [0.06, 0.08, 0.10, 0.12]
DD_TIGHTS    = [0.03, 0.04, 0.05]


def sample_params(feature_name, n=N_SAMPLE):
    """随机拒绝采样有效参数."""
    cfg = FEATURE_COMBOS[feature_name]
    need_fed = cfg["fed_gate"]
    need_veto = cfg["pb_veto"]
    need_confirm = (cfg["fed_gate"] and cfg["pb_veto"])

    items = set()
    attempts = 0
    max_attempts = n * 50

    while len(items) < n and attempts < max_attempts:
        fl = random.choice(BUY_FLOORS)
        lo = random.choice(BUY_LOWS)
        mi = random.choice(BUY_MIDS)
        hi = random.choice(BUY_HIGHS)
        wa = random.choice(SELL_WARNS)
        he = random.choice(SELL_HEAVYS)
        ex = random.choice(SELL_EXTS)
        fd = random.choice(FEDS) if need_fed else 0.0
        vt = random.choice(VETOS) if need_veto else 0.0
        cf = random.choice(CONFIRMS) if need_confirm else 0.0
        ds = random.choice(DD_STDS)
        dt = random.choice(DD_TIGHTS)
        attempts += 1

        if not (fl < lo < mi < hi and wa < he < ex):
            continue
        key = (fl, lo, mi, hi, wa, he, ex, fd, vt, cf, ds, dt)
        items.add(key)

    result = [tuple(k) for k in items]
    return result


# =========================== 数据加载 & 预处理 ===========================
def load_index_data(code, window_years):
    """加载单个指数的回测数据, 并计算给定窗口的滚动百分位."""
    merged = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    price = pd.read_parquet(PRICE_DIR / f"{code}.parquet")

    merged["date"] = pd.to_datetime(merged["date"])
    price["date"] = pd.to_datetime(price["date"])

    dj_mask = merged["pe_ttm_dj"].notna()
    dj = merged[dj_mask][["date", "pe_ttm_dj", "fed_dj", "pb_dj"]].copy()

    price_col = "index_open" if "index_open" in price.columns else "index_price"
    price_sorted = price[["date", price_col]].dropna().sort_values("date")
    dj_sorted = dj.sort_values("date")

    bt_df = pd.merge_asof(dj_sorted, price_sorted, on="date", direction="backward")
    bt_df = bt_df.dropna(subset=[price_col]).reset_index(drop=True)

    if len(bt_df) < 50:
        return None

    bt_df["price"] = bt_df[price_col].values
    bt_df["fed_val"] = bt_df["fed_dj"].values
    bt_df["pb_val"] = bt_df["pb_dj"].values if "pb_dj" in bt_df.columns else np.nan

    total_days = (bt_df["date"].max() - bt_df["date"].min()).days
    total_years = total_days / 365.25
    if total_years < window_years:
        return None

    rpy = len(bt_df) / max(total_years, 1)
    wr = int(window_years * rpy)

    pe_arr = bt_df["pe_ttm_dj"].values.astype(float)
    pb_arr = bt_df["pb_val"].values.astype(float)
    fed_arr = bt_df["fed_val"].values.astype(float)

    bt_df["pe_pct"] = vec_rolling_pct(pe_arr, wr)
    bt_df["pb_pct"] = vec_rolling_pct(pb_arr, wr) if not np.isnan(pb_arr).all() else np.full(len(bt_df), np.nan)
    m, s = vec_rolling_mean_std(fed_arr, wr)
    bt_df["fed_mean"] = m
    bt_df["fed_std"] = s

    return bt_df


def preload_all_data():
    """预加载所有 (指数, 窗口) 组合."""
    cache = {}
    for code in CODES:
        for w in WINDOWS:
            key = (code, w)
            try:
                df = load_index_data(code, w)
                if df is not None and len(df) >= 50:
                    cache[key] = df
                    print(f"  loaded: {code} w={w}yr rows={len(df)}")
                else:
                    print(f"  skip:   {code} w={w}yr (数据不足)")
            except Exception as e:
                print(f"  error:  {code} w={w}yr — {e}")
    return cache


# =========================== 回测引擎 (支持特征组合) ===========================
def run_one_combo(df, params, feature_cfg):
    """
    单次回测.
    params = (fl, lo, mi, hi, wa, he, ex, fd, vt, cf, ds, dt)
    feature_cfg = {"primary": "PE"/"PB", "fed_gate": bool, "pb_veto": bool}
    """
    fl, lo, mi, hi, wa, he, ex, fd, vt, cf, ds, dt = params

    dates = df["date"].values
    prices = df["price"].values
    fed_vals = df["fed_val"].values
    fed_means = df["fed_mean"].values
    fed_stds = df["fed_std"].values

    primary = feature_cfg["primary"]
    if primary == "PE":
        pcts = df["pe_pct"].values
    else:
        pcts = df["pb_pct"].values

    pb_pcts = df["pb_pct"].values
    fed_gate = feature_cfg["fed_gate"]
    pb_veto = feature_cfg["pb_veto"] and primary != "PB"

    n = len(df)
    shares = 0.0
    total_invested = 0.0
    peak_price = 0.0
    sell_mode = 0
    trades = []

    last_buy_week = -1
    last_sell_month = -1
    sell_month_done = False
    after_sell_cooldown = False

    for i in range(n):
        pct = pcts[i]
        if np.isnan(pct) or pct < 0 or pct > 1:
            continue
        price = prices[i]
        if np.isnan(price) or price <= 0:
            continue

        pb = pb_pcts[i]
        fm = fed_means[i]
        fs = fed_stds[i]
        fv = fed_vals[i]

        dt_str = str(dates[i])[:10]
        parts = dt_str.split("-")
        year_month = int(parts[0]) * 12 + int(parts[1])
        cal_year = int(parts[0])
        iso_week = int(pd.Timestamp(dt_str).isocalendar()[1])
        week_key = cal_year * 53 + iso_week

        if year_month != last_sell_month:
            sell_month_done = False
            after_sell_cooldown = False

        # ---- 止盈信号 ----
        if pct >= ex:
            sell_mode = 3
        elif pct >= he:
            sell_mode = 2
        elif pct >= wa:
            sell_mode = max(sell_mode, 1)
        else:
            sell_mode = 0

        can_sell = not sell_month_done

        # ---- 卖出 heavy (每月最多1次) ----
        if sell_mode == 2 and shares > 0 and can_sell:
            over = pct - he
            sell_ratio = min(over * 0.10, 0.25)
            s = shares * sell_ratio
            if s > 0:
                shares -= s
                trades.append((
                    dt_str, "sell", -s * price, shares, price,
                    pct, pb if not np.isnan(pb) else None, total_invested
                ))
                sell_month_done = True
                last_sell_month = year_month
                after_sell_cooldown = True

        # ---- 卖出 extreme (清仓信号, 每月最多1次) ----
        if sell_mode == 3 and shares > 0 and can_sell:
            peak_price = max(peak_price, price)
            if peak_price > 0:
                dd = (peak_price - price) / peak_price
                if dd >= dt:
                    sell_ratio = min(0.25 + dd * 0.3, 0.50)
                    s = shares * sell_ratio
                    if s > 0:
                        shares -= s
                        trades.append((
                            dt_str, "clear", -s * price, shares, price,
                            pct, pb if not np.isnan(pb) else None, total_invested
                        ))
                        sell_month_done = True
                        last_sell_month = year_month
                        after_sell_cooldown = True
                        sell_mode = 0
                        peak_price = 0

        # ---- 买入 (每周最多1次, 卖出当月不买) ----
        if sell_mode in (0, 1) and week_key != last_buy_week and not after_sell_cooldown:
            # FED 门控
            if fed_gate and not (np.isnan(fm) or np.isnan(fs) or np.isnan(fv)):
                if fv < fm + fd * fs:
                    continue
            # PB 否决
            if pb_veto and not np.isnan(pb) and pb >= vt:
                continue

            if pct < fl:
                mult = 3
            elif pct < lo:
                mult = 2
            elif pct < mi:
                mult = 1
            elif pct < hi:
                mult = 0.5
            else:
                mult = 0

            if mult > 0:
                amount = BASE_AMOUNT * mult
                shares += amount / price
                total_invested += amount
                trades.append((
                    dt_str, "buy", amount, shares, price,
                    pct, pb if not np.isnan(pb) else None, total_invested
                ))
                last_buy_week = week_key

    final_price = float(prices[-1]) if len(prices) > 0 else 0.0
    final_value = float(shares * final_price)
    buy_count = sum(1 for t in trades if t[1] == "buy")

    return {
        "total_invested": total_invested,
        "final_value": final_value,
        "trades": len(trades),
        "buys": buy_count,
        "cash_flows": trades,
        "params": params,
    }


# =========================== 运行 XIRR 序列 (高速版) ===========================
def compute_running_annualized(trades):
    """从 trades 计算每个交易点的运行年化收益率 (高速版).
    全程使用简单年化: annualized = simple_ret / (days/365.25)
    速度 O(trades)，XIRR 仅在最终验证时单独计算.
    返回 list of (date, annualized_return)
    """
    if len(trades) < 2:
        return []

    cum_invested = 0.0
    cum_cash = 0.0
    first_date = None
    result = []

    for t in trades:
        d, act, amt = str(t[0])[:10], t[1], float(t[2])
        sh = float(t[3])
        pr = float(t[4])

        if first_date is None:
            first_date = d

        if act == "buy":
            cum_invested += amt
        elif act in ("sell", "clear"):
            cum_cash += abs(amt)

        equity = sh * pr
        total = equity + cum_cash

        if cum_invested <= 0:
            result.append((d, 0.0))
            continue

        simple_ret = (total - cum_invested) / cum_invested
        days = max((pd.Timestamp(d) - pd.Timestamp(first_date)).days, 1)
        annualized = simple_ret / (days / 365.25)

        result.append((d, annualized))

    return result


def compute_final_xirr(trades):
    """计算最终 XIRR (仅用于验证/诊断)."""
    if len(trades) < 2:
        return 0.0

    cum_cf = []
    cum_invested = 0.0
    cum_cash = 0.0
    last_shares = 0.0
    last_price = 0.0
    last_date = None

    for t in trades:
        d, act, amt = str(t[0])[:10], t[1], float(t[2])
        sh = float(t[3])
        pr = float(t[4])
        if act == "buy":
            cum_invested += amt
            cum_cf.append((d, -amt))
        elif act in ("sell", "clear"):
            cum_cf.append((d, -amt))
            cum_cash += abs(amt)
        last_shares = sh
        last_price = pr
        last_date = d

    equity = last_shares * last_price
    if len(cum_cf) < 2 or equity <= 0:
        return 0.0
    return calc_xirr(cum_cf, str(last_date)[:10], equity)


# =========================== 评分 ===========================
def score_from_xirr(xirr_series, w1, w2):
    """从运行 XIRR 序列计算评分: avg * w1 - sigma * w2."""
    vals = np.array([v for _, v in xirr_series if v is not None and not np.isnan(v)])
    if len(vals) < 3:
        return 0.0, 0.0, 0.0
    avg = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=1))
    score = avg * w1 - sigma * w2
    return score, avg, sigma


def strategy_score_across_indices(per_index_results, w1, w2):
    """跨指数平均评分."""
    scores = []
    for _, info in per_index_results.items():
        s, a, g = score_from_xirr(info["xirr_series"], w1, w2)
        if not np.isnan(s):
            scores.append(s)
    return np.mean(scores) if scores else 0.0


# =========================== 主流程 ===========================
def main():
    parser = argparse.ArgumentParser(description="宽基统一策略网格搜索")
    parser.add_argument("--n", type=int, default=N_SAMPLE, help="每 (特征组合,窗口) 采样数")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-load", action="store_true", help="跳过数据加载 (使用缓存)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print(f"  宽基统一策略网格搜索 — {ts}")
    print("=" * 70)
    print(f"  宽基指数: {CODES}")
    print(f"  窗口年限: {WINDOWS}")
    print(f"  权重组合: {WEIGHTS}")
    print(f"  特征组合: {list(FEATURE_COMBOS)}")
    print(f"  采样数:   {args.n} / (特征组合,窗口)")
    print("=" * 70)

    # ---- Phase 0: 预加载数据 ----
    if not args.skip_load:
        print("\n[Phase 0] 预加载指数数据 ...")
        data_cache = preload_all_data()
        print(f"  可用 (指数,窗口) = {len(data_cache)}\n")
    else:
        print("\n[Phase 0] 跳过 (--skip-load)\n")
        data_cache = {}

    # ---- Phase 1: 运行回测 ----
    print("[Phase 1] 遍历回测...")
    all_results = []

    for w in WINDOWS:
        for feat_name, feat_cfg in FEATURE_COMBOS.items():
            params_list = sample_params(feat_name, n=args.n)
            t0 = time.time()
            valid_count = 0

            for p_idx, params in enumerate(params_list):
                per_index = {}
                active_codes = 0
                for code in CODES:
                    key = (code, w)
                    if key not in data_cache:
                        continue
                    df = data_cache[key]

                    result = run_one_combo(df, params, feat_cfg)
                    if result["trades"] < 2:
                        continue

                    ann_series = compute_running_annualized(result["cash_flows"])
                    if len(ann_series) < 2:
                        continue

                    final_xirr = compute_final_xirr(result["cash_flows"])

                    per_index[code] = {
                        "xirr_series": ann_series,
                        "final_xirr": final_xirr,
                        "total_invested": result["total_invested"],
                        "final_value": result["final_value"],
                        "trades": result["trades"],
                        "buys": result["buys"],
                    }
                    active_codes += 1

                if active_codes >= 3:
                    all_results.append({
                        "window": w,
                        "feature": feat_name,
                        "params": params,
                        "per_index": per_index,
                    })
                    valid_count += 1

                if (p_idx + 1) % 200 == 0:
                    print(f"  [{feat_name} w={w}] {p_idx+1}/{len(params_list)}", flush=True)

            elapsed = time.time() - t0
            print(f"  [{feat_name} w={w}] 完成 — {valid_count} 有效 ({elapsed:.0f}s)")

    total_runs = len(all_results)
    print(f"\n[Phase 1] 完成, 有效策略数: {total_runs}\n")

    # ---- Phase 2: 评分 & CSV 导出 ----
    print("[Phase 2] 评分 & CSV 导出...")

    for w1, w2 in WEIGHTS:
        label = f"w{str(w1).replace('.','_')}_{str(w2).replace('.','_')}"
        scored = []
        for r in all_results:
            s = strategy_score_across_indices(r["per_index"], w1, w2)
            scored.append((s, r))
        scored.sort(key=lambda x: x[0], reverse=True)

        top_rows = []
        for w in WINDOWS:
            for feat_name in FEATURE_COMBOS:
                group = [(s, r) for s, r in scored
                         if r["window"] == w and r["feature"] == feat_name]
                for rank, (s, r) in enumerate(group[:5], 1):
                    p = r["params"]
                    row = {
                        "window": w, "feature": feat_name, "w1": w1, "w2": w2,
                        "rank": rank, "strategy_score": round(s, 6),
                        "pe_buy_floor": p[0], "pe_buy_low": p[1],
                        "pe_buy_mid": p[2], "pe_buy_high": p[3],
                        "pe_sell_warn": p[4], "pe_sell_heavy": p[5],
                        "pe_sell_extreme": p[6],
                        "fed_buy_threshold": p[7],
                        "pb_veto_threshold": p[8],
                        "pb_confirm_threshold": p[9],
                        "drawdown_standard": p[10], "drawdown_tight": p[11],
                    }
                    for code in CODES:
                        info = r["per_index"].get(code)
                        if info:
                            xv = np.array([v for _, v in info["xirr_series"]
                                           if v is not None and not np.isnan(v)])
                            if len(xv) >= 3:
                                row[f"{code}_avg_ann"] = round(np.mean(xv), 6)
                                row[f"{code}_sigma_ann"] = round(np.std(xv, ddof=1), 6)
                            else:
                                row[f"{code}_avg_ann"] = None
                                row[f"{code}_sigma_ann"] = None
                            row[f"{code}_final_xirr"] = round(info.get("final_xirr", 0), 6)
                        else:
                            row[f"{code}_avg_ann"] = None
                            row[f"{code}_sigma_ann"] = None
                            row[f"{code}_final_xirr"] = None
                    top_rows.append(row)

        csv_path = OUTPUT_DIR / f"top5_{label}.csv"
        pd.DataFrame(top_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  top5_{label}.csv — {len(top_rows)} rows")

    # ---- Phase 3: 全量导出 ----
    print("\n[Phase 3] 全量结果导出...")
    for w1, w2 in WEIGHTS:
        label = f"w{str(w1).replace('.','_')}_{str(w2).replace('.','_')}"
        flat = []
        for r in all_results:
            s = strategy_score_across_indices(r["per_index"], w1, w2)
            flat.append({
                "window": r["window"], "feature": r["feature"],
                "strategy_score": round(s, 6),
                "pe_buy_floor": r["params"][0], "pe_buy_low": r["params"][1],
                "pe_buy_mid": r["params"][2], "pe_buy_high": r["params"][3],
                "pe_sell_warn": r["params"][4], "pe_sell_heavy": r["params"][5],
                "pe_sell_extreme": r["params"][6],
                "fed_buy_threshold": r["params"][7],
                "pb_veto_threshold": r["params"][8],
                "pb_confirm_threshold": r["params"][9],
                "drawdown_standard": r["params"][10], "drawdown_tight": r["params"][11],
            })
        flat.sort(key=lambda x: x["strategy_score"], reverse=True)
        pd.DataFrame(flat).to_csv(
            OUTPUT_DIR / f"all_{label}.csv", index=False, encoding="utf-8-sig"
        )
        print(f"  all_{label}.csv — {len(flat)} rows")

    # ---- Phase 4: 量级诊断 ----
    print("\n[Phase 4] 量级诊断...")
    all_avgs = []; all_sigs = []
    for r in all_results:
        for _, info in r["per_index"].items():
            xa = np.array([v for _, v in info["xirr_series"] if v is not None and not np.isnan(v)])
            if len(xa) >= 3:
                all_avgs.append(np.mean(xa))
                all_sigs.append(np.std(xa, ddof=1))
    if all_avgs:
        print(f"  avg_xirr  range: [{min(all_avgs):.6f}, {max(all_avgs):.6f}]")
        print(f"  sigma     range: [{min(all_sigs):.6f}, {max(all_sigs):.6f}]")
        print(f"  avg_xirr  mean:  {np.mean(all_avgs):.6f}")
        print(f"  sigma     mean:  {np.mean(all_sigs):.6f}")
        ratio = np.mean(all_sigs) / (np.mean(all_avgs) + 1e-9)
        print(f"  sigma/avg ratio: {ratio:.2f}  (接近1.0=量级可比)")
        if ratio > 10:
            print(f"  ⚠ sigma 量级大 {ratio:.0f}x, w2>0 将主导评分, 考虑归一化")

    # ---- Phase 5: HTML 报告 ----
    print("\n[Phase 5] HTML 报告...")
    html_path = OUTPUT_DIR / "report.html"
    _write_html_report(all_results, html_path)
    print(f"  {html_path}")

    elapsed_total = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  总耗时: {elapsed_total/60:.1f} 分钟")
    print(f"  有效策略: {total_runs}")
    print(f"  输出: {OUTPUT_DIR}/")
    print(f"{'='*70}")


# =========================== HTML 报告 ===========================
def _write_html_report(all_results, path):
    """生成可视化 HTML 报告."""

    # 取 top 策略数据 (w1=1.0, w2=0.0)
    top_for_html = {"w1_0_w0_0": [], "w0_8_w0_2": [], "w0_6_w0_4": []}
    for w1, w2 in WEIGHTS:
        lbl = f"w{str(w1).replace('.','_')}_{str(w2).replace('.','_')}"
        scored = [(strategy_score_across_indices(r["per_index"], w1, w2), r) for r in all_results]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_for_html[lbl] = scored[:30]

    # 选 w1=1.0 的第一名做跨指数展示
    best_list = top_for_html["w1_0_w0_0"]
    top1 = best_list[0][1] if best_list else None

    cross_index_data = {"codes": [], "names": [], "avgs": [], "sigmas": [], "trades": []}
    if top1:
        for code in CODES:
            info = top1["per_index"].get(code)
            if info:
                xa = np.array([v for _, v in info["xirr_series"] if v is not None and not np.isnan(v)])
                cross_index_data["codes"].append(code)
                cross_index_data["names"].append(CODE_NAMES.get(code, code))
                cross_index_data["avgs"].append(round(np.mean(xa), 6) if len(xa) >= 3 else 0)
                cross_index_data["sigmas"].append(round(np.std(xa, ddof=1), 6) if len(xa) >= 3 else 0)
                cross_index_data["trades"].append(info["trades"])

    # ---- 构建 score 分布数据 (按 feature 分组) ----
    dist_data = {}
    for w1, w2 in [(1.0, 0.0)]:
        for r in all_results:
            feat = r["feature"]
            w = r["window"]
            key = f"{feat}"
            if key not in dist_data:
                dist_data[key] = {"windows": set(), "scores": [], "params_6": [], "params_8": [], "params_10": []}
            s = strategy_score_across_indices(r["per_index"], w1, w2)
            dist_data[key]["scores"].append(s)
            dist_data[key]["windows"].add(f"w{w}")

    dist_json = json.dumps({k: {"count": len(v["scores"]),
                                  "max": round(max(v["scores"]), 6),
                                  "mean": round(np.mean(v["scores"]), 6)}
                             for k, v in dist_data.items()}, ensure_ascii=False)
    cross_json = json.dumps(cross_index_data, ensure_ascii=False)

    # ---- Top table JSON for tabs ----
    tab_data = {}
    for lbl, scored_list in top_for_html.items():
        for w in WINDOWS:
            for feat_name in FEATURE_COMBOS:
                group = [(s, r) for s, r in scored_list
                         if r["window"] == w and r["feature"] == feat_name][:5]
                if group:
                    key = f"{lbl}_w{w}_{feat_name}"
                    tab_data[key] = {
                        "label": lbl, "window": w, "feature": feat_name,
                        "rows": [{"rank": i+1, "score": round(s, 6),
                                  "floor": f"{r['params'][0]:.0%}",
                                  "low": f"{r['params'][1]:.0%}",
                                  "mid": f"{r['params'][2]:.0%}",
                                  "high": f"{r['params'][3]:.0%}",
                                  "warn": f"{r['params'][4]:.0%}",
                                  "heavy": f"{r['params'][5]:.0%}",
                                  "extreme": f"{r['params'][6]:.0%}",
                                  "fed": r['params'][7],
                                  "veto_cfm": f"{r['params'][8]:.0%}/{r['params'][9]:.0%}",
                                  "dd": f"{r['params'][10]:.2f}/{r['params'][11]:.2f}",}
                                 for i, (s, r) in enumerate(group, 1)],
                    }
    tab_json = json.dumps(tab_data, ensure_ascii=False)

    # ---- Build HTML ----
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>宽基统一策略 网格搜索报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:16px}}
h1{{font-size:22px;margin-bottom:4px;color:#f8fafc}}
h2{{font-size:16px;margin:28px 0 10px;color:#94a3b8;border-bottom:1px solid #334155;padding-bottom:6px}}
h3{{font-size:14px;margin:16px 0 8px;color:#cbd5e1}}
.meta{{color:#64748b;font-size:13px;margin-bottom:20px}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:20px}}
th{{background:#1e293b;color:#94a3b8;padding:8px 6px;text-align:left;border-bottom:2px solid #334155}}
td{{padding:6px;border-bottom:1px solid #1e293b;white-space:nowrap}}
tr:hover{{background:#1e293b}}
.good{{color:#4ade80}}
.bad{{color:#f87171}}
.hl{{background:#173554}}
.chart{{width:100%;height:360px;background:#1e293b;border-radius:8px;margin-bottom:12px;border:1px solid #334155}}
.tabs{{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}}
.tab{{padding:6px 14px;background:#1e293b;border:1px solid #334155;border-radius:6px;cursor:pointer;font-size:12px;color:#94a3b8}}
.tab.active{{background:#2563eb;color:#fff;border-color:#2563eb}}
.row2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:900px){{.row2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>宽基统一策略 网格搜索报告</h1>
<div class="meta">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &nbsp;|&nbsp;
宽基: {", ".join(CODES)} &nbsp;|&nbsp;
窗口: {", ".join(f"{w}yr" for w in WINDOWS)} &nbsp;|&nbsp;
样本: {N_SAMPLE}/组 &nbsp;|&nbsp;
有效策略: {len(all_results)}</div>

<h2>Top 1 策略 · 跨指数 avg_xirr / sigma</h2>
<div class="row2">
  <div class="chart" id="chart-avg"></div>
  <div class="chart" id="chart-sigma"></div>
</div>

<h2>特征组合得分分布</h2>
<div class="chart" id="chart-dist"></div>

<h2>各组合 Top5 详情</h2>
<div class="tabs" id="tabs"></div>
<div id="tab-content"></div>

<script>
(function() {{
    const CROSS = {cross_json};
    const DIST = {dist_json};
    const TAB = {tab_json};

    // ---- 跨指数 bar ----
    const c1 = echarts.init(document.getElementById('chart-avg'));
    c1.setOption({{
        title:{{text:'Top1 avg_xirr 各指数',left:'center',textStyle:{{color:'#e2e8f0',fontSize:14}}}},
        tooltip:{{trigger:'axis',formatter:p=>p[0]?p[0].name+'<br/>avg_xirr: '+(p[0].value*100).toFixed(2)+'%':'—'}},
        grid:{{left:80,right:20,top:45,bottom:70}},
        xAxis:{{type:'category',data:CROSS.names,axisLabel:{{color:'#94a3b8',rotate:30,fontSize:11}}}},
        yAxis:{{type:'value',axisLabel:{{color:'#94a3b8',formatter:v=>(v*100).toFixed(1)+'%'}}}},
        series:[{{type:'bar',data:CROSS.avgs,itemStyle:{{color:'#60a5fa'}},
          label:{{show:true,position:'top',color:'#94a3b8',fontSize:10,formatter:p=>(p.value*100).toFixed(2)+'%'}}}}],
    }});

    const c2 = echarts.init(document.getElementById('chart-sigma'));
    c2.setOption({{
        title:{{text:'Top1 sigma_xirr 各指数',left:'center',textStyle:{{color:'#e2e8f0',fontSize:14}}}},
        tooltip:{{trigger:'axis'}},
        grid:{{left:80,right:20,top:45,bottom:70}},
        xAxis:{{type:'category',data:CROSS.names,axisLabel:{{color:'#94a3b8',rotate:30,fontSize:11}}}},
        yAxis:{{type:'value',axisLabel:{{color:'#94a3b8'}}}},
        series:[{{type:'bar',data:CROSS.sigmas,itemStyle:{{color:'#f97316'}},
          label:{{show:true,position:'top',color:'#94a3b8',fontSize:10,formatter:p=>p.value.toFixed(4)}}}}],
    }});

    // ---- 得分分布 ----
    const featKeys = Object.keys(DIST);
    const c3 = echarts.init(document.getElementById('chart-dist'));
    c3.setOption({{
        title:{{text:'特征组合得分分布 (w1=1.0)',left:'center',textStyle:{{color:'#e2e8f0',fontSize:14}}}},
        tooltip:{{trigger:'axis',formatter:p=>p[0]?p[0].name+'<br/>max: '+(p[0].value*100).toFixed(2)+'%<br/>mean: '+(DIST[p[0].name]?.mean*100).toFixed(2)+'%':'—'}},
        grid:{{left:80,right:20,top:45,bottom:80}},
        xAxis:{{type:'category',data:featKeys,axisLabel:{{color:'#94a3b8',rotate:30,fontSize:11}}}},
        yAxis:{{type:'value',name:'max score',axisLabel:{{color:'#94a3b8'}}}},
        series:[
            {{type:'bar',name:'max',data:featKeys.map(k=>DIST[k].max),itemStyle:{{color:'#60a5fa'}},
              label:{{show:true,position:'top',color:'#94a3b8',fontSize:10,formatter:p=>(p.value*100).toFixed(2)+'%'}}}},
            {{type:'bar',name:'mean',data:featKeys.map(k=>DIST[k].mean),itemStyle:{{color:'#475569'}},
              label:{{show:true,position:'top',color:'#64748b',fontSize:9,formatter:p=>(p.value*100).toFixed(2)+'%'}}}},
        ],
    }});

    // ---- Tabs ----
    const keys = Object.keys(TAB);
    if (keys.length === 0) return;
    const tabsDiv = document.getElementById('tabs');
    const contentDiv = document.getElementById('tab-content');

    keys.forEach((k, i) => {{
        const d = TAB[k];
        const btn = document.createElement('span');
        btn.className = 'tab' + (i===0?' active':'');
        btn.textContent = d.label + ' W' + d.window + ' ' + d.feature;
        btn.onclick = () => showTab(k, btn);
        tabsDiv.appendChild(btn);
    }});

    function showTab(key, el) {{
        document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
        el.classList.add('active');
        const d = TAB[key];
        let h = '<h3>'+d.label+' · W='+d.window+'yr · '+d.feature+'</h3><table>';
        h += '<tr><th>#</th><th>Score</th><th>Floor</th><th>Low</th><th>Mid</th><th>High</th><th>Warn</th><th>Heavy</th><th>Extr</th><th>FED</th><th>Veto/Cfm</th><th>DD</th></tr>';
        d.rows.forEach((r, i) => {{
            h += '<tr class="'+(i===0?'hl':'')+'"><td>'+r.rank+'</td><td class="good">'+r.score+'</td>';
            h += '<td>'+r.floor+'</td><td>'+r.low+'</td><td>'+r.mid+'</td><td>'+r.high+'</td>';
            h += '<td>'+r.warn+'</td><td>'+r.heavy+'</td><td>'+r.extreme+'</td>';
            h += '<td>'+r.fed+'</td><td>'+r.veto_cfm+'</td><td>'+r.dd+'</td></tr>';
        }});
        h += '</table>';
        contentDiv.innerHTML = h;
    }}

    showTab(keys[0], document.querySelector('.tab'));
}})();
</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
