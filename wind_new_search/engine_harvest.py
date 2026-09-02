#!/usr/bin/env python3
"""
均衡策略 v2026-09-01 增量引擎 — 独立实现, 不修改 engine.py

在均衡策略基础上新增两条规则:
  规则1 (盈利收割卖出): 当 总资产 > 净占用本金×(1+profit_ratio) 时, 强制卖出
        超额利润的 profit_frac 比例, 把利润抽出来分配其他理财。
        - 不受"每月卖1次"约束
        - 触发后当月不再按 PE/PB 估值卖出 (本月互斥)
        - 卖出的现金记为 withdrawn (已抽出, 不再计入本金池终值)
  规则2 (底仓保护): 任何卖出后持仓市值不得低于 净占用本金×floor_ratio (底仓),
        保证大牛市不丢光筹码。

仅 import engine.py 的纯函数 (calc_xirr/mult_for/分位/归一化), 不改动原引擎。
数据源: 只读 data-store/parquet/wind_new_merged (与 test_balanced 同源)。
输出:  独立 JSON, 不覆盖任何现有文件。

用法:
  python wind_new_search/engine_harvest.py --code 000300   # 单指数回测(打印明细)
  python wind_new_search/test_260901.py                     # 全量测试集 + 对比 balanced
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import (
    calc_xirr, mult_for, _expensive_pct, normalize_params,
    _principal_pct_max_scale, _principal_min_mult, prep_df,
)


def run_harvest(df, params, base_amount=1000, commission_rate=0.0005,
                min_commission=5.0, lot_size=0,
                principal_threshold=None, principal_cap=None, principal_pool=None,
                buy_mults=None, profit_ratio=0.30, profit_frac=0.20,
                floor_ratio=0.20, exec_price=None):
    """新增规则的完整回测. 返回与 run_backtest 同结构的 dict (额外含 withdrawn/harvests).

    profit_ratio: 总资产相对净占用本金超额比例阈值 (默认0.30=超额30%触发)
    profit_frac : 收割卖出比例 (超额利润的 fraction, 默认0.20)
    floor_ratio : 底仓 = 净占用本金×floor_ratio 市值永不卖 (默认0.20)
    """
    params = normalize_params(params)
    if buy_mults is None:
        buy_mults = (8.0, 4.0, 2.0, 0.0)
    buy_signal = params["buy_signal"]
    buy_gate = params.get("buy_gate")
    buy_gate_cap = params.get("buy_gate_cap")
    sell_signal = params["sell_signal"]
    sell_gate = params.get("sell_gate")
    sell_gate_floor = params.get("sell_gate_floor")

    def _as_list(v):
        if v is None:
            return []
        return list(v) if isinstance(v, (list, tuple)) else [v]
    buy_gates = _as_list(buy_gate)
    buy_gate_caps = _as_list(buy_gate_cap)
    sell_gates = _as_list(sell_gate)
    sell_gate_floors = _as_list(sell_gate_floor)
    bf, bl, bm, bh = params["buy_floor"], params["buy_low"], params["buy_mid"], params["buy_high"]
    sh, se = params["sell_heavy"], params["sell_extreme"]

    df = df.sort_values("date").reset_index(drop=True)
    df = prep_df(df)
    dates = df["date"].values
    prices = df["price"].values.astype(float)
    if exec_price is None:
        exec_prices = prices
    else:
        exec_prices = np.asarray(exec_price, dtype=float)
    pe_pcts = df["pe_pct"].values.astype(float)
    pb_pcts = df["pb_pct"].values.astype(float)
    fed_pcts = df["fed_pct"].values.astype(float)
    ym_arr = df["_ym"].values
    wk_arr = df["_wk"].values

    buy_exp = _expensive_pct(buy_signal, pe_pcts, pb_pcts, fed_pcts)
    gate_exps = [_expensive_pct(g, pe_pcts, pb_pcts, fed_pcts) for g in buy_gates]
    sell_exp = _expensive_pct(sell_signal, pe_pcts, pb_pcts, fed_pcts)
    sell_gate_exps = [_expensive_pct(g, pe_pcts, pb_pcts, fed_pcts) for g in sell_gates]

    shares = 0.0
    total_invested = 0.0
    total_cash_in = 0.0
    withdrawn = 0.0          # 已抽出的利润 (分配其他理财, 不计入本金池)
    trades = []
    first_tradable = None
    last_buy_week = -1
    last_sell_month = -1
    sell_month_done = False      # 本月已按估值卖出
    harvest_month_done = False   # 本月已收割 (触发后本月不再估值卖出)

    n = len(df)
    for i in range(n):
        pe_pct = pe_pcts[i]
        if np.isnan(pe_pct) or pe_pct < 0 or pe_pct > 1:
            continue
        if first_tradable is None:
            first_tradable = dates[i]
        price = prices[i]
        if np.isnan(price) or price <= 0:
            continue
        px = exec_prices[i]
        if np.isnan(px) or px <= 0:
            continue
        pb_pct = pb_pcts[i]
        dt_str = str(dates[i])[:10]
        year_month = ym_arr[i]
        week_key = wk_arr[i]

        if year_month != last_sell_month:
            sell_month_done = False
            harvest_month_done = False

        net_invested = total_invested - total_cash_in   # 净占用本金
        floor_value = net_invested * floor_ratio         # 底仓市值 (永不卖穿)
        eq = shares * px                                  # 当前持仓市值

        # ============ 规则1: 盈利收割卖出 ============
        # 触发: 持仓市值 > 净占用本金×(1+profit_ratio)  (持仓超额盈利达标)
        # 收割: 卖 超额持仓利润 的 profit_frac, 抽出分配其他理财
        # 不受月度限制; 触发后当月不再按估值卖出, 但不抑制买入
        can_harvest = (not harvest_month_done) and shares > 0 and total_invested > 0
        if can_harvest:
            excess = eq - net_invested * (1.0 + profit_ratio)
            if excess > 0:
                sell_amount = excess * profit_frac
                # 底仓保护: 卖出后持仓市值不得低于 floor_value
                max_sell = eq - floor_value
                if max_sell <= 0:
                    sell_amount = 0.0
                else:
                    sell_amount = min(sell_amount, max_sell)
                if sell_amount > 0:
                    s = sell_amount / px
                    if lot_size > 0:
                        s = int(s / lot_size) * lot_size
                    if s > 0:
                        gross = s * px
                        comm = max(gross * commission_rate, min_commission) if commission_rate > 0 else 0.0
                        cash_in = gross - comm
                        shares -= s
                        total_cash_in += cash_in
                        withdrawn += cash_in                    # 利润抽出, 不再留在池内
                        trades.append((dt_str, "harvest", -cash_in, shares, px,
                                       float(pe_pct),
                                       float(pb_pct) if not np.isnan(pb_pct) else None,
                                       total_invested))
                        harvest_month_done = True
                        last_sell_month = year_month

        # ============ 规则2 + 原估值卖出 (月度1次, 触发收割当月互斥) ============
        sell_mode = 0
        sp = sell_exp[i]
        if not np.isnan(sp):
            if sp >= se:
                sell_mode = 3
            elif sp >= sh:
                sell_mode = 2
        if sell_mode >= 2 and sell_gates:
            for gp, floor in zip(sell_gate_exps, sell_gate_floors):
                if floor is None:
                    continue
                gv = gp[i]
                if np.isnan(gv) or gv < floor:
                    sell_mode = 0
                    break
        can_sell = (not sell_month_done) and (not harvest_month_done)
        if sell_mode >= 2 and shares > 0 and can_sell:
            ratio = 0.20
            if sell_mode == 3:
                ratio = 0.50
            # 底仓保护: 卖出后市值不得低于 floor_value
            sell_value_limit = eq - floor_value
            if sell_value_limit <= 0:
                sell_mode = 0
            if sell_mode >= 2:
                s = shares * ratio
                if lot_size > 0:
                    s = int(s / lot_size) * lot_size
                if s <= 0 or s * px > sell_value_limit + 1e-9:
                    s = max(0.0, int(sell_value_limit / px / (lot_size if lot_size > 0 else 1)) * (lot_size if lot_size > 0 else 1))
            if sell_mode >= 2 and s > 0:
                gross = s * px
                comm = max(gross * commission_rate, min_commission) if commission_rate > 0 else 0.0
                cash_in = gross - comm
                shares -= s
                total_cash_in += cash_in
                act = "clear" if sell_mode == 3 else "sell"
                trades.append((dt_str, act, -cash_in, shares, px, float(pe_pct),
                               float(pb_pct) if not np.isnan(pb_pct) else None,
                               total_invested))
                sell_month_done = True
                last_sell_month = year_month

        # ============ 买入 (每周≤1, 估值卖出当月不买; 收割当月仍可买) ============
        can_buy = (week_key != last_buy_week) and (not sell_month_done)
        if can_buy:
            net_invested2 = total_invested - total_cash_in
            mult = mult_for(buy_exp[i], bf, bl, bm, bh, buy_mults)
            if mult > 0 and buy_gates:
                for gp, cap in zip(gate_exps, buy_gate_caps):
                    if cap is None:
                        continue
                    gv = gp[i]
                    if np.isnan(gv) or gv > cap:
                        mult = 0
                        break
            if mult > 0 and principal_threshold:
                min_mult = _principal_min_mult(net_invested2, principal_threshold)
                if mult < min_mult:
                    mult = 0
            if mult > 0:
                amt = base_amount * mult
                if lot_size > 0:
                    s = int(amt / px / lot_size) * lot_size
                    cost = s * px
                else:
                    s = amt / px
                    cost = amt
                if s <= 0:
                    last_buy_week = week_key
                    continue
                comm = max(cost * commission_rate, min_commission) if commission_rate > 0 else 0.0
                if principal_cap and (net_invested2 + cost + comm) > principal_cap:
                    last_buy_week = week_key
                    continue
                shares += s
                total_invested += cost + comm
                trades.append((dt_str, "buy", cost + comm, shares, px, float(pe_pct),
                               float(pb_pct) if not np.isnan(pb_pct) else None,
                               total_invested))
                last_buy_week = week_key

    final_price = float(exec_prices[-1]) if len(exec_prices) else 0
    pos_value = shares * final_price if shares > 0 else 0
    pool_cash = total_cash_in - withdrawn
    total_value = pos_value + pool_cash   # 池内总资产 (不含已抽出的利润)
    invested_base = total_invested if total_invested > 0 else 1
    final_return = (total_value - total_invested) / invested_base

    principal_final = None
    principal_return = None
    principal_annual = None
    if principal_pool and principal_pool > 0:
        # 固定口径: 本金池 + 池内总资产 − 累计投入; 已抽出的利润(withdrawn)计入收益
        principal_final = principal_pool + total_value - total_invested + withdrawn
        principal_return = (principal_final - principal_pool) / principal_pool
        if first_tradable is not None:
            years = (pd.Timestamp(dates[-1]) - pd.Timestamp(first_tradable)).days / 365.25
        else:
            years = 0.0
        if years > 0 and principal_final > 0:
            principal_annual = (principal_final / principal_pool) ** (1.0 / years) - 1.0
        else:
            principal_annual = 0.0

    cashflows = [(t[0], -t[2]) for t in trades]
    # XIRR 终值 = 全部资产 (持仓 + 池内现金 + 已抽走的利润), 反映完整收益
    xirr = calc_xirr(cashflows, str(dates[-1])[:10], pos_value + total_cash_in) \
        if len(cashflows) >= 3 else 0.0

    buys = sum(1 for t in trades if t[1] == "buy")
    sells = sum(1 for t in trades if t[1] in ("sell", "clear"))
    harvests = sum(1 for t in trades if t[1] == "harvest")

    return {
        "xirr": xirr,
        "final_return": round(final_return, 4),
        "total_invested": round(total_invested, 0),
        "total_cash_in": round(total_cash_in, 0),
        "withdrawn": round(withdrawn, 0),
        "final_value": round(total_value, 0),
        "position_value": round(pos_value, 0),
        "trades": len(trades), "buys": buys, "sells": sells, "harvests": harvests,
        "cash_flows": trades,
        "principal_final": round(principal_final, 0) if principal_final is not None else None,
        "principal_return": round(principal_return, 4) if principal_return is not None else None,
        "principal_annual": round(principal_annual, 4) if principal_annual is not None else None,
    }


def build_curve_harvest(df, params, base_amount=1000, commission_rate=0.0005,
                        min_commission=5.0, lot_size=0,
                        principal_threshold=None, principal_cap=None, principal_pool=None,
                        buy_mults=None, profit_ratio=0.60, profit_frac=0.20,
                        floor_ratio=0.20, exec_price=None):
    """构建前端所需的每日曲线 (meta + daily + trades), 口径与 run_harvest 一致.

    返回 daily 额外含 principal/annualized/daily_return (供画资金曲线+算夏普).
    """
    df = df.sort_values("date").reset_index(drop=True)
    r = run_harvest(df, params, base_amount=base_amount, commission_rate=commission_rate,
                    min_commission=min_commission, lot_size=lot_size,
                    principal_threshold=principal_threshold, principal_cap=principal_cap,
                    principal_pool=principal_pool, buy_mults=buy_mults,
                    profit_ratio=profit_ratio, profit_frac=profit_frac,
                    floor_ratio=floor_ratio, exec_price=exec_price)
    flows = r["cash_flows"]
    flow_by_date = {}
    for t in flows:
        flow_by_date.setdefault(t[0], []).append(t)

    dates = df["date"].dt.strftime("%Y-%m-%d").values
    if exec_price is None:
        prices = df["price"].values.astype(float)
    else:
        prices = np.asarray(exec_price, dtype=float)
    pe = df["pe"].values.astype(float)
    pb = df["pb"].values.astype(float)
    fed = df["fed"].values.astype(float)
    pe_pct = df["pe_pct"].values.astype(float)
    pb_pct = df["pb_pct"].values.astype(float)
    fed_pct = df["fed_pct"].values.astype(float)

    daily = []
    shares = 0.0
    cum_invested = 0.0
    cash = 0.0
    withdrawn = 0.0
    cf = []
    last_xirr = None
    first_invest = None
    prev_principal = None
    first_tradable = None

    for i in range(len(df)):
        d = dates[i]
        if first_tradable is None and not np.isnan(pe_pct[i]) and 0 <= pe_pct[i] <= 1:
            first_tradable = d
        buy_amt = sell_amt = harvest_amt = 0.0
        if d in flow_by_date:
            for t in flow_by_date[d]:
                act, amt = t[1], float(t[2])
                if act == "buy":
                    cf.append((d, -amt))
                    cum_invested += amt
                    buy_amt += amt
                elif act == "harvest":
                    cf.append((d, -amt))
                    cash += abs(amt)
                    withdrawn += abs(amt)
                    harvest_amt += abs(amt)
                else:
                    cf.append((d, -amt))
                    cash += abs(amt)
                    sell_amt += abs(amt)
                shares = float(t[3])
            if len(cf) >= 3 and shares * prices[i] > 0:
                last_xirr = calc_xirr(cf, d, shares * prices[i] + cash)
        if first_invest is None and cum_invested > 0:
            first_invest = d

        eq = shares * prices[i] if shares > 0 else 0.0
        total_value = eq + (cash - withdrawn)   # 池内总资产 (不含已抽出利润)
        ret = (total_value + withdrawn - cum_invested) / cum_invested * 100 if cum_invested > 0 else 0.0
        row = {
            "date": d, "price": round(float(prices[i]), 2),
            "pe": None if np.isnan(pe[i]) else round(float(pe[i]), 2),
            "pb": None if np.isnan(pb[i]) else round(float(pb[i]), 2),
            "fed": None if np.isnan(fed[i]) else round(float(fed[i]), 2),
            "pe_pct": None if np.isnan(pe_pct[i]) else round(float(pe_pct[i]), 4),
            "pb_pct": None if np.isnan(pb_pct[i]) else round(float(pb_pct[i]), 4),
            "fed_pct": None if np.isnan(fed_pct[i]) else round(float(fed_pct[i]), 4),
            "cum_invested": round(cum_invested, 0),
            "equity": round(eq, 0), "cash": round(cash, 0), "withdrawn": round(withdrawn, 0),
            "total_value": round(total_value, 0),
            "return_pct": round(ret, 2), "xirr": last_xirr,
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
            "harvest_amount": round(harvest_amt, 0) if harvest_amt > 0 else 0,
        }
        if principal_pool:
            # 固定口径: 本金池 + (池内总资产 + 已抽出) − 累计投入
            principal = principal_pool + (total_value + withdrawn) - cum_invested
            if first_tradable is not None:
                years = (pd.Timestamp(d) - pd.Timestamp(first_tradable)).days / 365.25
            else:
                years = 0.0
            if years > 0 and principal > 0:
                annualized = (principal / principal_pool) ** (1.0 / years) - 1.0
            else:
                annualized = 0.0
            daily_return = (principal - prev_principal) / prev_principal * 100.0 if prev_principal else 0.0
            prev_principal = principal
            row["principal"] = round(principal, 0)
            row["annualized"] = round(annualized, 4)
            row["daily_return"] = round(daily_return, 4)
        daily.append(row)

    trades = []
    for t in flows:
        d, act, amt, sh, pr, ppct, ppb, inv = t
        trades.append({
            "date": d, "action": act, "amount": round(float(amt), 0),
            "shares": round(float(sh), 4), "price": round(float(pr), 2),
            "pe_pct": round(float(ppct), 4) if ppct is not None and not np.isnan(ppct) else None,
            "pb_pct": round(float(ppb), 4) if ppb is not None and not np.isnan(ppb) else None,
            "cum_invested": round(float(inv), 0),
        })

    meta = {
        "signal_mode": params.get("buy_signal", "PB"), "params": params,
        "total_invested": r["total_invested"], "final_value": r["final_value"],
        "position_value": r["position_value"], "total_cash_in": r["total_cash_in"],
        "withdrawn": r["withdrawn"],
        "trades": r["trades"], "buys": r["buys"], "sells": r["sells"], "harvests": r["harvests"],
        "xirr": r["xirr"], "final_return": r["final_return"],
        "first_invest": first_invest, "first_tradable": first_tradable,
        "principal_threshold": principal_threshold, "principal_cap": principal_cap,
        "principal_pool": principal_pool,
        "principal_final": r.get("principal_final"),
        "principal_return": r.get("principal_return"),
        "principal_annual": r.get("principal_annual"),
        "profit_ratio": profit_ratio, "profit_frac": profit_frac, "floor_ratio": floor_ratio,
    }
    return {"meta": meta, "daily": daily, "trades": trades}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="000300")
    ap.add_argument("--profit-ratio", type=float, default=0.30)
    ap.add_argument("--profit-frac", type=float, default=0.20)
    ap.add_argument("--floor-ratio", type=float, default=0.20)
    args = ap.parse_args()

    MERGED = PROJECT_DIR / "data-store" / "parquet" / "wind_new_merged"
    df = pd.read_parquet(MERGED / f"{args.code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    PARAMS = {
        "buy_signal": "PB", "buy_gate": "FED", "buy_gate_cap": 0.55,
        "sell_signal": "PE", "sell_gate": None, "sell_gate_floor": None,
        "buy_floor": 0.10, "buy_low": 0.15, "buy_mid": 0.25, "buy_high": 0.70,
        "sell_heavy": 0.85, "sell_extreme": 0.95,
    }
    r = run_harvest(df, PARAMS, base_amount=1000, commission_rate=0.0005,
                    min_commission=5.0, lot_size=0,
                    principal_threshold=200_000, principal_cap=300_000,
                    principal_pool=300_000, buy_mults=(8, 4, 2, 0),
                    profit_ratio=args.profit_ratio, profit_frac=args.profit_frac,
                    floor_ratio=args.floor_ratio)
    print(f"{args.code}: 固定年化 {r['principal_annual']*100:.2f}%  XIRR {r['xirr']*100:.2f}%  "
          f"买{r['buys']}/卖{r['sells']}/收割{r['harvests']}  withdrawn ¥{r['withdrawn']:,.0f}  "
          f"终值 ¥{r['principal_final']:,.0f}")
