#!/usr/bin/env python3
"""
PE-DCA 回测引擎 v2

优化: 预计算滚动百分位, 网格搜索 O(1) 读缓存。

用法:
    python backtest/backtest.py --code 000300 --pe all
"""

import argparse
import itertools
import os
import random
import sys
from datetime import datetime, date as dt_date

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_DIR = os.path.join(PROJECT_DIR, "data-store", "parquet", "merged")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "backtest", "output")

BASE_AMOUNT = 1500
WINDOW_YEARS_LIST = [3, 5, 8, 10]

PE_SOURCES = {"dj": "pe_ttm_dj", "csi": "pe_ttm_csi", "wind": "pe_ttm_wind"}

BROAD_INDICES = [
    "000300","000905","000852","000016","000688",
    "000015","000922","930955","930915",
    "930930","930931","931573","930939",
]
# 缺价格: 399006, 399330, HSI, HSTECH, NDX100, SPX500
# 零窗口: 000510
INDEX_NAMES = {"000300":"沪深300","000905":"中证500","000852":"中证1000",
    "000016":"上证50","000688":"科创50","399006":"创业板指","399330":"深证100","000510":"中证A500",
    "000015":"上证红利","000922":"中证红利","930955":"红利低波100",
    "930915":"港股通高股息","930930":"港股综合","930931":"港股通50",
    "931573":"港股通科技","930939":"中证质量成长"}

# ============================================================================
# 预计算滚动百分位 (向量化版本)
# ============================================================================

def vec_rolling_pct(series: np.ndarray, window_days: int, min_samples: int = 20):
    """
    向量化滚动百分位。
    对每个位置 i, 计算 series[i] 在 series[i-window:i+1] 中的百分位。
    """
    n = len(series)
    result = np.full(n, np.nan)
    clean = ~np.isnan(series)
    indices = np.arange(n)

    for i in range(n):
        if not clean[i]:
            continue
        start = max(0, i - window_days)
        window_clean = clean[start:i+1]
        if window_clean.sum() < min_samples:
            continue
        window_vals = series[start:i+1][window_clean]
        result[i] = (window_vals <= series[i]).sum() / len(window_vals)

    return result


def vec_rolling_mean_std(series: np.ndarray, window_days: int, min_samples: int = 20):
    """
    向量化滚动均值和标准差。
    """
    n = len(series)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    clean = ~np.isnan(series)

    for i in range(n):
        start = max(0, i - window_days)
        window_clean = clean[start:i+1]
        if window_clean.sum() < min_samples:
            continue
        window_vals = series[start:i+1][window_clean]
        mean[i] = window_vals.mean()
        std[i] = window_vals.std()

    return mean, std


# ============================================================================
# 参数生成
# ============================================================================

def gen_params_list():
    """预生成所有合法参数组合列表 (约 37K valid/window)，供随机取样。"""
    all_params = []
    floors  = [0.05, 0.10, 0.15, 0.20]
    lows    = [0.15, 0.20, 0.25, 0.30]
    mids    = [0.30, 0.35, 0.40, 0.45]
    highs   = [0.50, 0.55, 0.60, 0.65, 0.70]
    warns   = [0.65, 0.70, 0.75]
    heavys  = [0.75, 0.80, 0.85]
    extremes = [0.85, 0.90, 0.95]
    feds    = [-0.5, 0.0, 0.5, 1.0]
    vetos   = [0.50, 0.55, 0.60, 0.65]
    confirms = [0.70, 0.75, 0.80]
    dd_stds = [0.06, 0.08, 0.10, 0.12]
    dd_tights = [0.03, 0.04, 0.05]

    for fl, lo, mi, hi in itertools.product(floors, lows, mids, highs):
        if not (fl < lo < mi < hi):
            continue
        for wa, he, ex in itertools.product(warns, heavys, extremes):
            if not (wa < he < ex):
                continue
            for fd in feds:
                for vt in vetos:
                    for cf in confirms:
                        for ds in dd_stds:
                            for dt in dd_tights:
                                all_params.append((fl, lo, mi, hi, wa, he, ex, fd, vt, cf, ds, dt))
    return all_params


# ============================================================================
# XIRR 计算
# ============================================================================

def calc_xirr(cashflows, final_date, final_value):
    """
    计算 XIRR (年化内部收益率), 二分法求解。
    cashflows: [(date_str, amount), ...], 买=负(流出), 卖=正(流入)
    final_value: 最终持仓市值
    """
    if len(cashflows) < 3 or final_value <= 0:
        return 0.0

    dates = [pd.Timestamp(d) for d, a in cashflows]
    # 至少 30 天数据
    if (dates[-1] - dates[0]).days < 30:
        return 0.0

    amounts = [a for _, a in cashflows]  # 已正确: 买=负, 卖=正

    # 最终市值 = 现金流入
    dates.append(pd.Timestamp(final_date))
    amounts.append(final_value)

    t0 = dates[0]
    years = np.array([(d - t0).days / 365.25 for d in dates])
    amt_arr = np.array(amounts)

    def npv(rate):
        return np.sum(amt_arr / (1 + rate) ** years)

    # 二分法, 窄限 [-0.8, 1.0] 避免假解
    lo, hi = -0.8, 1.0
    if npv(lo) * npv(hi) > 0:
        return 0.0  # 无解

    for _ in range(60):
        mid = (lo + hi) / 2
        v = npv(mid)
        if abs(v) < 0.1:
            return round(mid, 4)
        if npv(lo) * v < 0:
            hi = mid
        else:
            lo = mid

    mid = (lo + hi) / 2
    return round(max(-0.5, min(mid, 1.0)), 4)


# ============================================================================
# 单次回测
# ============================================================================

def run_one(df: pd.DataFrame, params: tuple, w_idx: int, pe_col_idx: int) -> dict:
    """
    单次回测。
    df 列结构:
      date, price, pe_pct_w{0..3}, pb_pct_w{0..3}, fed_mean_w{0..3}, fed_std_w{0..3}
    params: (fl, lo, mi, hi, wa, he, ex, fd, vt, cf, ds, dt)
    """
    fl, lo, mi, hi, wa, he, ex, fd, vt, cf, ds, dt = params
    w = WINDOW_YEARS_LIST[w_idx]

    dates = df["date"].values
    prices = df["price"].values
    pe_pcts = df[f"pe_pct_w{w}"].values
    pb_pcts = df[f"pb_pct_w{w}"].values
    fed_means = df[f"fed_mean_w{w}"].values
    fed_stds = df[f"fed_std_w{w}"].values
    fed_vals = df["fed_val"].values

    n = len(df)
    shares = 0.0
    total_invested = 0.0
    peak_price = 0.0
    sell_mode = 0  # 0=none, 1=warn, 2=sell, 3=extreme
    trades = []

    # 交易频率控制
    last_buy_week = -1   # 每周最多买一次
    last_sell_month = -1  # 每月最多卖一次
    sell_month_done = False  # 当月已卖过
    after_sell_cooldown = False  # 卖后当月不买

    for i in range(n):
        pe = pe_pcts[i]
        if np.isnan(pe) or pe < 0 or pe > 1:
            continue

        price = prices[i]
        if np.isnan(price):
            continue
        pb = pb_pcts[i]
        fm = fed_means[i]
        fs = fed_stds[i]
        fv = fed_vals[i]

        # 月份/周识别
        dt_str = str(dates[i])[:10]
        date_parts = dt_str.split('-')
        year_month = int(date_parts[0]) * 12 + int(date_parts[1])
        # ISO week: simple approximation using date ordinal
        cal_week = dt_date(*[int(x) for x in date_parts]).isocalendar()[1]
        cal_year = int(date_parts[0])
        week_key = cal_year * 53 + cal_week

        # Monthly cooldown reset
        if year_month != last_sell_month:
            sell_month_done = False
            after_sell_cooldown = False

        # ---- 止盈 ----
        if pe >= ex:
            sell_mode = 3
        elif pe >= he:
            sell_mode = 2
        elif pe >= wa:
            sell_mode = max(sell_mode, 1)
        else:
            sell_mode = 0

        # ---- 卖出交易 (每月最多1次) ----
        can_sell = not sell_month_done
        if sell_mode == 2 and shares > 0 and can_sell:
            over = pe - he
            sell_ratio = min(over * 0.10, 0.25)  # 每月最多卖 5%~25%
            s = shares * sell_ratio
            if s > 0:
                shares -= s
                trades.append((dt_str, "sell", -s * price, shares, price, pe,
                               pb if not np.isnan(pb) else None, total_invested))
                sell_month_done = True
                last_sell_month = year_month
                after_sell_cooldown = True

        if sell_mode == 3 and shares > 0 and can_sell:
            peak_price = max(peak_price, price)
            if peak_price > 0:
                dd = (peak_price - price) / peak_price
                if dd >= dt:
                    sell_ratio = min(0.25 + dd * 0.3, 0.50)  # 极端回撤卖 25%~50%
                    s = shares * sell_ratio
                    if s > 0:
                        shares -= s
                        trades.append((dt_str, "clear", -s * price, shares, price, pe,
                                       pb if not np.isnan(pb) else None, total_invested))
                        sell_month_done = True
                        last_sell_month = year_month
                        after_sell_cooldown = True
                        sell_mode = 0
                        peak_price = 0

        # ---- 买入交易 (每周最多1次, 卖出当月不买) ----
        if sell_mode in (0, 1) and week_key != last_buy_week and not after_sell_cooldown:
            # FED 否决
            if not (np.isnan(fm) or np.isnan(fs) or np.isnan(fv)):
                if fv < fm + fd * fs:
                    continue
            # PB 否决
            if not np.isnan(pb) and pb >= vt:
                continue

            if pe < fl:
                mult = 3
            elif pe < lo:
                mult = 2
            elif pe < mi:
                mult = 1
            elif pe < hi:
                mult = 0.5
            else:
                mult = 0

            if mult > 0:
                amount = BASE_AMOUNT * mult
                new_s = amount / price
                shares += new_s
                total_invested += amount
                trades.append((dt_str, "buy", amount, shares, price, pe,
                               pb if not np.isnan(pb) else None, total_invested))
                last_buy_week = week_key

    # 最终结算
    final_price = prices[-1] if len(prices) > 0 else 0
    final_value = shares * final_price
    final_return = (final_value - total_invested) / total_invested if total_invested > 0 else 0

    # 指标
    if len(trades) >= 2 and total_invested > 0:
        peak = 0.0
        max_dd = 0.0
        cur_shares = 0.0
        eq_curve = []
        for d, act, amt, sh, pr, *_ in trades:
            if act == "buy":
                cur_shares = sh
            else:
                cur_shares = max(0, sh)
            eq = cur_shares * pr
            eq_curve.append(eq)
            peak = max(peak, eq)
            if peak > 0:
                max_dd = min(max_dd, (eq - peak) / peak)

        d1 = pd.Timestamp(trades[0][0])
        d2 = pd.Timestamp(trades[-1][0])
        yrs = max((d2 - d1).days / 365.25, 0.5)

        cagr = (1 + final_return) ** (1 / yrs) - 1

        returns = []
        for j in range(1, len(eq_curve)):
            if eq_curve[j-1] > 0:
                returns.append((eq_curve[j] - eq_curve[j-1]) / eq_curve[j-1])
        rets = np.array(returns)
        sharpe = 0.0
        if len(rets) > 1 and rets.std() > 0:
            sharpe = (rets.mean() * 52 - 0.02) / (rets.std() * np.sqrt(52))
        calmar = cagr / abs(max_dd) if max_dd < 0 and cagr > 0 else 0

        buy_count = sum(1 for t in trades if t[1] == "buy")
    else:
        cagr, max_dd, sharpe, calmar = 0, 0, 0, 0
        buy_count = 0

    # XIRR 计算
    xirr_val = 0.0
    if len(trades) >= 2 and final_value > 0:
        cashflows = []
        for t in trades:
            d, act, amt = t[0], t[1], t[2]
            if act == "buy":
                cashflows.append((d, -amt))  # 买入 = 现金流出
            elif act in ("sell", "clear"):
                cashflows.append((d, -amt))  # amt 本身为负, 取负 = 正 = 现金流入
        xirr_val = calc_xirr(cashflows, str(dates[-1])[:10], final_value)

    return {
        "window_years": w,
        "final_return": round(final_return, 4),
        "xirr": round(xirr_val, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(abs(max_dd), 4),
        "sharpe": round(sharpe, 4),
        "calmar": round(calmar, 4),
        "total_invested": round(total_invested, 0),
        "final_value": round(final_value, 0),
        "trades": len(trades),
        "buys": buy_count,
        "cash_flows": trades,  # 每条交易明细
        "pe_buy_floor": fl, "pe_buy_low": lo, "pe_buy_mid": mi, "pe_buy_high": hi,
        "pe_sell_warn": wa, "pe_sell_heavy": he, "pe_sell_extreme": ex,
        "fed_buy_threshold": fd, "pb_veto_threshold": vt, "pb_confirm_threshold": cf,
        "drawdown_standard": ds, "drawdown_tight": dt,
    }


# ============================================================================
# 网格搜索
# ============================================================================

def grid_search(code: str, pe_source: str, pe_col: str, output_dir: str):
    name = INDEX_NAMES.get(code, code)
    path = os.path.join(MERGED_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        return

    df0 = pd.read_parquet(path)
    df0["date"] = pd.to_datetime(df0["date"])
    df0 = df0.sort_values("date").reset_index(drop=True)

    if pe_col not in df0.columns or df0[pe_col].isna().sum() >= len(df0):
        print(f"  [{name}] {pe_source}: 无数据")
        return

    # 获取有效数据（该PE源非空的行）
    valid_mask = df0[pe_col].notna()
    df = df0[valid_mask].copy().reset_index(drop=True)
    print(f"  [{name}] {pe_source}: {len(df)} 有效行")

    if len(df) < 100:
        print(f"    数据太少, 跳过")
        return

    # 准备基础列
    fed_col = "fed_csi" if pe_col == "pe_ttm_csi" else ("fed_dj" if pe_col == "pe_ttm_dj" else None)
    has_price = "index_price" in df0.columns and df0["index_price"].notna().sum() > 0
    if has_price:
        # 对价格做 ffill（日频数据覆盖周频缺口）
        df0["index_price"] = df0["index_price"].ffill()

    df["price"] = df0.loc[valid_mask, "index_price"].values if has_price else df[pe_col].values
    if not has_price:
        print(f"    无明显价格数据, 用PE代理 (结果仅供方向性参考)")
    df["fed_val"] = df0.loc[valid_mask, fed_col].values if fed_col and fed_col in df0.columns else np.nan
    df["pb_val"] = df0.loc[valid_mask, "pb_dj"].values if "pb_dj" in df0.columns else np.nan

    dates_arr = df["date"].values
    pe_arr = df0.loc[valid_mask, pe_col].values.astype(float)  # PE值用于计算百分位
    pb_arr = df["pb_val"].values.astype(float) if "pb_dj" in df.columns else np.full(len(df), np.nan)
    fed_arr = df["fed_val"].values.astype(float) if fed_col else np.full(len(df), np.nan)

    # 预计算各窗口的滚动百分位
    # 根据数据频率计算实际窗口行数
    total_days = (df["date"].max() - df["date"].min()).days
    total_years = total_days / 365.25
    rows_per_year = len(df) / total_years  # 实际每年有多少行

    print(f"    预计算滚动指标 (每年 {rows_per_year:.0f} 行)...")
    for wi, w in enumerate(WINDOW_YEARS_LIST):
        window_rows = int(w * rows_per_year)
        if window_rows < 20:
            window_rows = 20

        days = w * 365
        df[f"pe_pct_w{w}"] = vec_rolling_pct(pe_arr, window_rows)
        if not np.isnan(pb_arr).all():
            df[f"pb_pct_w{w}"] = vec_rolling_pct(pb_arr, window_rows)
        else:
            df[f"pb_pct_w{w}"] = np.full(len(df), np.nan)
        if not np.isnan(fed_arr).all():
            m, s = vec_rolling_mean_std(fed_arr, window_rows)
            df[f"fed_mean_w{w}"] = m
            df[f"fed_std_w{w}"] = s
        else:
            df[f"fed_mean_w{w}"] = np.full(len(df), np.nan)
            df[f"fed_std_w{w}"] = np.full(len(df), np.nan)

    # 网格搜索: 每窗口独立保留 top 5
    all_results = []

    for wi, w in enumerate(WINDOW_YEARS_LIST):
        window_rows = int(w * rows_per_year)
        # 数据日期跨度必须 >= 窗口年数，否则跳过
        if total_years < w:
            print(f"      跳过 w={w}yr (数据仅 {total_years:.1f} 年)")
            continue
        if len(df) < window_rows * 0.5:
            continue

        count = 0
        top_w = []
        all_params = gen_params_list()
        random.shuffle(all_params)
        max_samples = min(3000, len(all_params))

        for params in all_params:
            count += 1
            if count > max_samples:
                break

            result = run_one(df, params, wi, 0)
            if result["trades"] < 2:
                continue
            score = result["final_return"]
            keep = 5  # 每窗口保留 top 5
            if len(top_w) < keep:
                top_w.append((score, result))
                top_w.sort(key=lambda x: x[0], reverse=True)
            elif score > top_w[-1][0]:
                top_w[-1] = (score, result)
                top_w.sort(key=lambda x: x[0], reverse=True)

        print(f"      w={w} 完成 {count} 组, top5:")
        for _, r in top_w[:5]:
            print(f"        ret={r['final_return']:.2%} cagr={r['cagr']:.2%} mdd={r['max_drawdown']:.2%} "
                  f"sharpe={r['sharpe']:.2f} floor={r['pe_buy_floor']:.0%} high={r['pe_buy_high']:.0%} "
                  f"fed={r['fed_buy_threshold']:.1f} veto={r['pb_veto_threshold']:.0%}")

        all_results.extend([r for _, r in top_w])

    if not all_results:
        print(f"    无有效结果")
        return

    results_df = pd.DataFrame(all_results)

    code_dir = os.path.join(output_dir, f"{code}_{name}")
    os.makedirs(code_dir, exist_ok=True)
    csv_path = os.path.join(code_dir, f"{pe_source}_top20.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"    汇总 (每窗口 top 5):")
    for w in sorted(results_df["window_years"].unique()):
        sub = results_df[results_df["window_years"] == w]
        best = sub.iloc[0]
        print(f"      w={int(w)}yr: best ret={best['final_return']:.2%} cagr={best['cagr']:.2%} "
              f"mdd={best['max_drawdown']:.2%} floor={best['pe_buy_floor']:.0%}")
    print(f"    结果: {csv_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="PE-DCA 回测引擎")
    parser.add_argument("--code", type=str, default=None)
    parser.add_argument("--pe", type=str, default="all", choices=["all","dj","csi","wind"])
    args = parser.parse_args()

    codes = [args.code] if args.code else BROAD_INDICES
    sources = [args.pe] if args.pe != "all" else ["dj", "csi", "wind"]

    out_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nPE-DCA 回测 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"指数: {len(codes)} 个, PE源: {sources}\n")

    for code in codes:
        name = INDEX_NAMES.get(code, code)
        for src in sources:
            pe_col = PE_SOURCES[src]
            grid_search(code, src, pe_col, out_dir)
        print()

    print(f"\n结果: {out_dir}/")


if __name__ == "__main__":
    main()
