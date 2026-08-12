#!/usr/bin/env python3
"""
卖出参数遍历实验 — 2026-08-09
================================
简化卖出逻辑: 到达阈值即卖固定比例, 无步进系数.
遍历 heavy_ratio / extreme_ratio / dt 的最优组合.

固定买入: floor=10%, low=15%, mid=35%, high=70%, warn=70%, fed=-0.5
窗口: 8yr | 特征: PB_FED | 宽基: 6个
"""
import itertools, json, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "backtest"))
sys.path.insert(0, str(PROJECT_DIR))
from backtest import vec_rolling_pct, vec_rolling_mean_std, calc_xirr as _xirr

MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "merged"
PRICE_DIR = PROJECT_DIR / "data-store" / "parquet" / "index_price"
OUTPUT_DIR = PROJECT_DIR / "grid_search" / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CODES = ["000300", "000905", "000852", "000016", "399006", "399330"]
CODE_NAMES = {
    "000300": "沪深300", "000905": "中证500", "000852": "中证1000",
    "000016": "上证50", "399006": "创业板指", "399330": "深证100",
}
W = 8
BASE_AMOUNT = 1500

# 固定买入参数
FLOOR, LOW, MID, HIGH, WARN, FED = 0.10, 0.15, 0.35, 0.70, 0.70, -0.5

# 卖出遍历参数
HE_RANGE = [0.75, 0.80, 0.85]
EX_RANGE = [0.85, 0.90, 0.95]
DT_RANGE = [0.03, 0.04, 0.05]
H_RATIO_RANGE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35]
E_RATIO_RANGE = [0.10, 0.15, 0.20, 0.25, 0.35, 0.50]

# ───────────────────── 数据加载 ─────────────────────

def load_data(code):
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
    if total_years < W:
        return None
    rpy = len(bt_df) / max(total_years, 1)
    wr = int(W * rpy)
    pb_arr = bt_df["pb_val"].values.astype(float)
    fed_arr = bt_df["fed_val"].values.astype(float)
    bt_df["pb_pct"] = vec_rolling_pct(pb_arr, wr) if not np.isnan(pb_arr).all() else np.full(len(bt_df), np.nan)
    m, s = vec_rolling_mean_std(fed_arr, wr)
    bt_df["fed_mean"] = m; bt_df["fed_std"] = s
    return bt_df


# ───────────────────── 简化卖出回测 ─────────────────────

def run_simple_sell(df, he, ex, dt, heavy_ratio, extreme_ratio):
    """买入 + 简化卖出: 到达阈值即卖固定比例."""
    dates = df["date"].values
    prices = df["price"].values
    pcts = df["pb_pct"].values
    fed_vals = df["fed_val"].values
    fed_means = df["fed_mean"].values
    fed_stds = df["fed_std"].values

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
        fm, fs, fv = fed_means[i], fed_stds[i], fed_vals[i]

        dt_str = str(dates[i])[:10]
        parts = dt_str.split("-")
        year_month = int(parts[0]) * 12 + int(parts[1])
        cal_year = int(parts[0])
        iso_week = int(pd.Timestamp(dt_str).isocalendar()[1])
        week_key = cal_year * 53 + iso_week

        if year_month != last_sell_month:
            sell_month_done = False
            after_sell_cooldown = False

        # 止盈信号
        if pct >= ex:
            sell_mode = 3
        elif pct >= he:
            sell_mode = 2
        elif pct >= WARN:
            sell_mode = max(sell_mode, 1)
        else:
            sell_mode = 0

        can_sell = not sell_month_done

        # Heavy sell: 固定比例
        if sell_mode == 2 and shares > 0 and can_sell:
            s = shares * heavy_ratio
            if s > 0:
                shares -= s
                trades.append((dt_str, "sell", -s * price, shares, price, pct, total_invested))
                sell_month_done = True
                last_sell_month = year_month
                after_sell_cooldown = True

        # Extreme sell: 固定比例 (高于 heavy)
        if sell_mode == 3 and shares > 0 and can_sell:
            peak_price = max(peak_price, price)
            if peak_price > 0:
                dd = (peak_price - price) / peak_price
                if dd >= dt:
                    s = shares * extreme_ratio
                    if s > 0:
                        shares -= s
                        trades.append((dt_str, "clear", -s * price, shares, price, pct, total_invested))
                        sell_month_done = True
                        last_sell_month = year_month
                        after_sell_cooldown = True
                        sell_mode = 0
                        peak_price = 0

        # 买入 (与当前最优相同)
        if sell_mode in (0, 1) and week_key != last_buy_week and not after_sell_cooldown:
            if not (np.isnan(fm) or np.isnan(fs) or np.isnan(fv)):
                if fv < fm + FED * fs:
                    continue
            if pct >= WARN:
                continue
            if pct < FLOOR: mult = 3
            elif pct < LOW: mult = 2
            elif pct < MID: mult = 1
            elif pct < HIGH: mult = 0.5
            else: mult = 0
            if mult > 0:
                amount = BASE_AMOUNT * mult
                shares += amount / price
                total_invested += amount
                trades.append((dt_str, "buy", amount, shares, price, pct, total_invested))
                last_buy_week = week_key

    # 最终结算
    final_price = float(prices[-1])
    final_shares = float(shares)
    cum_cash = sum(abs(float(t[2])) for t in trades if t[1] in ("sell", "clear"))
    final_total = final_shares * final_price + cum_cash

    # XIRR
    xirr = 0.0
    if len(trades) >= 2 and final_shares * final_price > 0:
        cf = []
        for t in trades:
            d, act, amt = str(t[0])[:10], t[1], float(t[2])
            if act == "buy": cf.append((d, -amt))
            elif act in ("sell", "clear"): cf.append((d, -amt))
        xirr = _xirr(cf, str(dates[-1])[:10], final_shares * final_price)

    simple_ret = (final_total - total_invested) / total_invested if total_invested > 0 else 0
    buys = sum(1 for t in trades if t[1] == "buy")
    sells = len(trades) - buys

    return {
        "xirr": round(xirr, 6),
        "simple_return": round(simple_ret, 6),
        "trades": len(trades),
        "buys": buys,
        "sells": sells,
        "invested": round(total_invested, 0),
        "final_total": round(final_total, 0),
        "cash_flows": trades,
    }


# ───────────────────── 主流程 ─────────────────────

def main():
    t0 = time.time()

    # 生成有效参数组合
    valid_params = []
    for he, ex, dt in itertools.product(HE_RANGE, EX_RANGE, DT_RANGE):
        if he >= ex: continue
        for hr, er in itertools.product(H_RATIO_RANGE, E_RATIO_RANGE):
            if hr >= er: continue
            valid_params.append((he, ex, dt, hr, er))
    print("=" * 65)
    print(f"  卖出参数遍历 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  有效组合: {len(valid_params)}  宽基: {len(CODES)}  总回测: {len(valid_params)*len(CODES)}")
    print("=" * 65)

    # 预加载
    print("\n[0] 预加载数据...")
    data_cache = {}
    for code in CODES:
        df = load_data(code)
        if df is not None:
            data_cache[code] = df
            print(f"  loaded: {code} {CODE_NAMES[code]} rows={len(df)}")
        else:
            print(f"  skip:   {code}")

    # 遍历
    print(f"\n[1] 遍历 {len(valid_params)} 组参数...")
    results = []
    for pi, (he, ex, dt, hr, er) in enumerate(valid_params):
        scores = []
        row = {"he": he, "ex": ex, "dt": dt, "heavy_ratio": hr, "extreme_ratio": er}
        for code in CODES:
            if code not in data_cache:
                continue
            r = run_simple_sell(data_cache[code], he, ex, dt, hr, er)
            if r["trades"] < 2:
                continue
            scores.append(r["xirr"])
            row[f"{code}_xirr"] = r["xirr"]
            row[f"{code}_return"] = r["simple_return"]
            row[f"{code}_trades"] = r["trades"]
            row[f"{code}_buys"] = r["buys"]
            row[f"{code}_sells"] = r["sells"]

        if len(scores) >= 3:
            row["avg_xirr"] = round(np.mean(scores), 6)
            row["n_indices"] = len(scores)
            results.append(row)

        if (pi + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {pi+1}/{len(valid_params)} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\n  完成 {len(results)} 个有效策略 ({elapsed:.0f}s)")

    # 排序 & 输出
    results.sort(key=lambda x: x["avg_xirr"], reverse=True)

    print(f"\n[2] Top 10 策略:")
    for i, r in enumerate(results[:10]):
        print(f"  #{i+1} avg_xirr={(r['avg_xirr']*100):.2f}% "
              f"he={r['he']:.0%} ex={r['ex']:.0%} dt={r['dt']:.0%} "
              f"h_ratio={r['heavy_ratio']:.0%} e_ratio={r['extreme_ratio']:.0%} "
              f"n={r['n_indices']}")

    # 保存 CSV
    csv_path = OUTPUT_DIR / "sell_tune_results.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
    # 保存 JSON (供前端)
    json_path = OUTPUT_DIR / "sell_tune_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"\n  输出: {csv_path}")
    print(f"       {json_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
