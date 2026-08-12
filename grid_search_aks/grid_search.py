#!/usr/bin/env python3
"""
akshare 网格搜索回测引擎 v2

跨指数统一评分：每组参数同时回测 沪深300/上证50/中证500，
取 min(XIRR) 作为统一评分（防止过拟合某单指数）。

用法:
  python grid_search_aks/grid_search.py                     # 统一搜索
  python grid_search_aks/grid_search.py --code 000300       # 单指数
  python grid_search_aks/grid_search.py --sample 1000       # 随机取样
"""

import argparse
import itertools
import json
import os
import random as _random
import sys
import time
from datetime import datetime, date as dt_date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
MERGED_DIR = PROJECT_DIR / "data-store" / "parquet" / "aks_merged"
OUTPUT_DIR = PROJECT_DIR / "grid_search_aks" / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_AMOUNT = 1500
WINDOW_YEARS = [3, 5, 10]

# ── 参数范围 ──────────────────────────────────────

PE_BUY_FLOORS = [0.10, 0.15]
PE_BUY_LOWS = [0.20, 0.25]
PE_BUY_MIDS = [0.35, 0.40]
PE_BUY_HIGHS = [0.55, 0.65]

PE_SELL_HEAVYS = [0.75, 0.80]
PE_SELL_EXTREMES = [0.85, 0.90, 0.95]

FED_GATES = [None, -0.5, 0.0]
PB_VETOS = [None, 0.60]
PB_SELLS = [None]

INDICES_ALL = {
    "000300": "沪深300",
    "000016": "上证50",
    "000905": "中证500",
}

# ── XIRR ──────────────────────────────────────────

def calc_xirr(cashflows, final_date, final_value):
    if len(cashflows) < 3 or final_value <= 0:
        return 0.0
    dates = [pd.Timestamp(d) for d, a in cashflows]
    if (dates[-1] - dates[0]).days < 30:
        return 0.0
    amounts = [a for _, a in cashflows]
    dates.append(pd.Timestamp(final_date))
    amounts.append(final_value)
    t0 = dates[0]
    years = np.array([(d - t0).days / 365.25 for d in dates])
    amt_arr = np.array(amounts)

    def npv(rate):
        return np.sum(amt_arr / (1 + rate) ** years)

    lo, hi = -0.8, 1.0
    if npv(lo) * npv(hi) > 0:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        v = npv(mid)
        if abs(v) < 0.1:
            return round(mid, 4)
        if npv(lo) * v < 0:
            hi = mid
        else:
            lo = mid
    return round(max(-0.5, min((lo + hi) / 2, 1.0)), 4)


# ── 单次回测 ─────────────────────────────────────

def run_backtest(df, params, w, base_amount=1500, max_net_principal=None,
                  idle_cash_rate=0.0, min_trades=5):
    bf, bl, bm, bh = params["buy_floor"], params["buy_low"], params["buy_mid"], params["buy_high"]
    sh, se = params["sell_heavy"], params["sell_extreme"]
    fed_gate, pb_veto, pb_sell = params["fed_gate"], params["pb_veto"], params["pb_sell"]

    pe_pct_col = f"pe_pct_w{w}"
    pb_pct_col = f"pb_pct_w{w}"
    fed_pct_col = f"fed_pct_w{w}"

    has_pb = pb_pct_col in df.columns and df[pb_pct_col].notna().any()
    has_fed = fed_pct_col in df.columns and df[fed_pct_col].notna().any()

    dates = df["date"].values
    prices = df["price"].values
    pe_pcts = df[pe_pct_col].values
    pb_pcts = df[pb_pct_col].values if has_pb else np.full(len(df), np.nan)
    fed_pcts = df[fed_pct_col].values if has_fed else np.full(len(df), np.nan)

    n = len(df)
    shares = 0.0
    total_invested = 0.0
    total_cash_in = 0.0
    peak_price = 0.0
    cash_balance = 0.0
    interest_earned = 0.0
    daily_rate = idle_cash_rate / 252.0
    trades = []

    last_buy_week = -1
    last_sell_month = -1
    sell_month_done = False
    after_sell_cooldown = False

    for i in range(n):
        pe_pct = pe_pcts[i]
        if np.isnan(pe_pct) or pe_pct < 0 or pe_pct > 1:
            continue
        price = prices[i]
        if np.isnan(price):
            continue
        pb_pct = pb_pcts[i] if has_pb else np.nan
        fed_pct_val = fed_pcts[i] if has_fed else np.nan

        dt_str = str(dates[i])[:10]
        parts = dt_str.split("-")
        year_month = int(parts[0]) * 12 + int(parts[1])
        cal_week = dt_date(*[int(x) for x in parts]).isocalendar()[1]
        cal_year = int(parts[0])
        week_key = cal_year * 53 + cal_week

        # 闲置现金生息（卖出现金未再投入的部分）
        if cash_balance > 0:
            interest_earned += cash_balance * daily_rate
            cash_balance += cash_balance * daily_rate

        if year_month != last_sell_month:
            sell_month_done = False
            after_sell_cooldown = False

        # 卖出
        sell_mode = 0
        if pe_pct >= se:
            sell_mode = 3
        elif pe_pct >= sh:
            sell_mode = 2
        if pb_sell is not None and has_pb and not np.isnan(pb_pct) and pb_pct >= pb_sell:
            sell_mode = max(sell_mode, 2)

        can_sell = not sell_month_done

        if sell_mode >= 2 and shares > 0 and can_sell:
            ratio = 0.20
            if sell_mode == 3:
                peak_price = max(peak_price, price)
                if peak_price > 0:
                    dd = (peak_price - price) / peak_price
                    if dd >= 0.05:
                        ratio = min(0.25 + dd * 0.3, 0.50)
                    else:
                        continue
            s = shares * ratio
            cash_in = s * price
            shares -= s
            total_cash_in += cash_in
            cash_balance += cash_in
            act = "clear" if sell_mode == 3 else "sell"
            trades.append((dt_str, act, -cash_in, shares, price, float(pe_pct),
                           float(pb_pct) if not np.isnan(pb_pct) else None, total_invested))
            sell_month_done = True
            last_sell_month = year_month
            after_sell_cooldown = True
            if sell_mode == 3:
                peak_price = 0

        # 买入
        can_buy = (week_key != last_buy_week) and not after_sell_cooldown
        if can_buy:
            if fed_gate is not None and has_fed and not np.isnan(fed_pct_val):
                if fed_pct_val < fed_gate:
                    continue
            if pb_veto is not None and has_pb and not np.isnan(pb_pct):
                if pb_pct >= pb_veto:
                    continue

            if pe_pct < bf:
                mult = 3
            elif pe_pct < bl:
                mult = 2
            elif pe_pct < bm:
                mult = 1
            elif pe_pct < bh:
                mult = 0.5
            else:
                mult = 0

            if mult > 0:
                amt = base_amount * mult
                if max_net_principal is not None:
                    remaining = max_net_principal - (total_invested - total_cash_in)
                    if remaining <= 0:
                        continue
                    amt = min(amt, remaining)
                s = amt / price
                shares += s
                total_invested += amt
                cash_balance -= amt
                trades.append((dt_str, "buy", amt, shares, price, float(pe_pct),
                               float(pb_pct) if not np.isnan(pb_pct) else None, total_invested))
                last_buy_week = week_key

    pos_value = shares * prices[-1] if shares > 0 else 0
    final_value = pos_value + max(cash_balance, 0)
    invested_base = total_invested if total_invested > 0 else 1
    final_return = (final_value - total_invested) / invested_base

    cashflows = [(t[0], -t[2] if t[1] == "buy" else abs(t[2])) for t in trades]
    xirr = calc_xirr(cashflows, str(dates[-1])[:10], final_value) if len(cashflows) >= 3 else 0.0

    buys = sum(1 for t in trades if t[1] == "buy")
    sells = sum(1 for t in trades if t[1] in ("sell", "clear"))

    return {
        "xirr": xirr, "final_return": round(final_return, 4),
        "total_invested": round(total_invested, 0),
        "total_cash_in": round(total_cash_in, 0),
        "net_principal": round(total_invested - total_cash_in, 0),
        "final_value": round(final_value, 0),
        "position_value": round(pos_value, 0),
        "idle_cash": round(max(cash_balance, 0), 0),
        "interest_earned": round(interest_earned, 2),
        "trades": len(trades), "buys": buys, "sells": sells,
        "cash_flows": trades,
    }


# ── 参数生成 ─────────────────────────────────────

def gen_param_combos(strict=False):
    """生成参数组合。strict=True 使用更严格的买入区间。"""
    bf_list = [0.05, 0.08] if strict else PE_BUY_FLOORS
    bl_list = [0.12, 0.15] if strict else PE_BUY_LOWS
    bm_list = [0.22, 0.28] if strict else PE_BUY_MIDS
    bh_list = [0.40, 0.50] if strict else PE_BUY_HIGHS
    combos = []
    for bf, bl, bm, bh in itertools.product(bf_list, bl_list, bm_list, bh_list):
        if not (bf < bl < bm < bh):
            continue
        for sh, se in itertools.product(PE_SELL_HEAVYS, PE_SELL_EXTREMES):
            if not (sh < se):
                continue
            for fed in FED_GATES:
                for pv in PB_VETOS:
                    for ps in PB_SELLS:
                        combos.append({
                            "buy_floor": bf, "buy_low": bl, "buy_mid": bm, "buy_high": bh,
                            "sell_heavy": sh, "sell_extreme": se,
                            "fed_gate": fed, "pb_veto": pv, "pb_sell": ps,
                        })
    return combos


# ── 跨指数统一评分搜索 ───────────────────────────
def unified_grid_search(codes, names, n_sample=0, w_list=None, base_amount=1500,
                         max_net_principal=None, idle_cash_rate=0.0, min_trades=5,
                         strict=False):
    """跨指数统一评分。"""
    if w_list is None:
        w_list = WINDOW_YEARS

    all_combos = gen_param_combos(strict=strict)
    if n_sample and n_sample < len(all_combos):
        all_combos = _random.sample(all_combos, n_sample)

    print(f"\n{'='*60}")
    print(f"跨指数统一搜索: {', '.join(names)}")
    print(f"{len(all_combos)} 参数组合 × {len(w_list)} 窗口")
    print(f"评分 = min(各指数XIRR)")
    print(f"{'='*60}")

    # 加载所有指数数据
    dfs = {}
    for code in codes:
        dfs[code] = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
        dfs[code]["date"] = pd.to_datetime(dfs[code]["date"])

    all_results = {}

    for w in w_list:
        print(f"\n--- 窗口 {w}yr ---")
        window_results = []
        t0 = time.time()

        for ci, params in enumerate(all_combos):
            code_results = {}
            xirrs = []
            all_ok = True

            for code in codes:
                try:
                    r = run_backtest(dfs[code], params, w, base_amount=base_amount,
                                     max_net_principal=max_net_principal,
                                     idle_cash_rate=idle_cash_rate, min_trades=min_trades)
                    if r["trades"] >= min_trades and r["final_value"] > 0:
                        code_results[code] = r
                        xirrs.append(r["xirr"])
                    else:
                        all_ok = False
                        break
                except Exception:
                    all_ok = False
                    break

            if all_ok and len(xirrs) == len(codes):
                min_xirr = min(xirrs)
                avg_xirr = sum(xirrs) / len(xirrs)
                window_results.append({
                    **params,
                    "unified_xirr": round(min_xirr, 4),
                    "avg_xirr": round(avg_xirr, 4),
                    **{f"{code}_xirr": round(code_results[code]["xirr"], 4) for code in codes},
                    **{f"{code}_return": round(code_results[code]["final_return"], 4) for code in codes},
                    **{f"{code}_trades": code_results[code]["trades"] for code in codes},
                    "total_trades": sum(code_results[code]["trades"] for code in codes),
                })

            if ci % 200 == 0 and ci > 0:
                elapsed = time.time() - t0
                sys.stdout.write(f"\r    进度: {ci}/{len(all_combos)} ({elapsed:.1f}s)")
                sys.stdout.flush()

        sys.stdout.write(f"\r    进度: {len(all_combos)}/{len(all_combos)} ({time.time()-t0:.1f}s) 有效={len(window_results)}\n")

        if not window_results:
            continue

        window_results.sort(key=lambda x: x["unified_xirr"], reverse=True)
        top = window_results[:30]

        print(f"  Top5 统一策略 (min XIRR):")
        for r in top[:5]:
            xirrs_str = " | ".join(f"{names.get(c,c)}={(r[f'{c}_xirr']*100):.1f}%" for c in codes)
            fed_str = f"FED={r['fed_gate']}" if r['fed_gate'] is not None else "FED=off"
            print(f"    统一XIRR={r['unified_xirr']*100:.2f}% avg={r['avg_xirr']*100:.2f}% "
                  f"B{r['buy_floor']}/{r['buy_low']}/{r['buy_mid']}/{r['buy_high']} "
                  f"S{r['sell_heavy']}/{r['sell_extreme']} {fed_str} PBv={r['pb_veto']}")
            print(f"       ({xirrs_str})")

        all_results[f"w{w}"] = {
            "codes": codes,
            "top": top,
        }

    return all_results


def single_grid_search(code, name, n_sample=0):
    """单指数网格搜索。"""
    df = pd.read_parquet(MERGED_DIR / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    all_combos = gen_param_combos()
    if n_sample and n_sample < len(all_combos):
        all_combos = _random.sample(all_combos, n_sample)

    print(f"\n{'='*60}")
    print(f"{name}({code}) — {len(all_combos)} 参数组合")
    results = {}

    for w in WINDOW_YEARS:
        pe_col = f"pe_pct_w{w}"
        if pe_col not in df.columns:
            continue
        valid = df[df[pe_col].notna()]
        if len(valid) < 100:
            continue

        t0 = time.time()
        window_results = []
        for params in all_combos:
            try:
                r = run_backtest(df, params, w)
                if r["trades"] >= 5 and r["final_value"] > 0:
                    window_results.append({**params, **{k: v for k, v in r.items() if k != "cash_flows"}})
            except Exception:
                continue

        window_results.sort(key=lambda x: x["xirr"], reverse=True)
        top = window_results[:30]

        elapsed = time.time() - t0
        print(f"  窗口{w}yr: {len(window_results)}有效 ({elapsed:.1f}s)")
        for r in top[:3]:
            print(f"    XIRR={r['xirr']*100:.2f}% B{r['buy_floor']}/{r['buy_low']}/{r['buy_mid']}/{r['buy_high']} S{r['sell_heavy']}/{r['sell_extreme']}")

        valid_df = df[df[pe_col].notna()]
        results[f"w{w}"] = {
            "trade_start": str(valid_df["date"].iloc[0])[:10],
            "trade_end": str(valid_df["date"].iloc[-1])[:10],
            "valid_days": len(valid_df),
            "top": top,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=str, default=None)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--unified", action="store_true", default=True,
                        help="跨指数统一搜索(默认)")
    parser.add_argument("--capped", action="store_true", default=False,
                        help="30万净值本金限制模式")
    parser.add_argument("--capped-base", type=int, default=500,
                        help="净值本金限制模式下的单笔买入基准 [default: 500]")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="严格买入模式（不限本金+闲置现金2%收益+min 10笔）")
    parser.add_argument("--strict-base", type=int, default=500,
                        help="严格买入模式下的单笔买入基准 [default: 500]")
    args = parser.parse_args()

    if args.code:
        r = single_grid_search(args.code, INDICES_ALL.get(args.code, args.code), args.sample)
        all_data = {args.code: r} if r else {}
    else:
        codes = ["000300", "000016", "000905"]
        names = {c: INDICES_ALL[c] for c in codes}
        if args.strict:
            r = unified_grid_search(codes, names, args.sample, w_list=[10],
                                    base_amount=args.strict_base, idle_cash_rate=0.02,
                                    min_trades=10, strict=True)
        elif args.capped:
            r = unified_grid_search(codes, names, args.sample, w_list=[10],
                                    base_amount=args.capped_base, max_net_principal=300_000)
        else:
            r = unified_grid_search(codes, names, args.sample)
        all_data = {"unified": r} if r else {}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.strict:
        suffix = "_strict" if args.strict_base == 500 else f"_strict{args.strict_base}"
    elif args.capped:
        suffix = "_capped" if args.capped_base == 500 else f"_capped{args.capped_base}"
    else:
        suffix = ""
    out_path = OUTPUT_DIR / f"grid_results{suffix}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False, default=str)
    latest_path = OUTPUT_DIR / f"latest{suffix}.json"
    with open(latest_path, "w") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果保存: {out_path} / {latest_path}")


if __name__ == "__main__":
    main()
