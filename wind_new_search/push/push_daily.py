#!/usr/bin/env python3
"""
推送系统主脚本 — 逐指数判断买卖信号并推送到企业微信群。

流程:
  1. 读取 config.json (webhook/策略参数/指数列表/账本路径)
  2. 读取 push/data/{code}.parquet (build_push_data.py 生成)
  3. 用引擎相同逻辑判断每个指数"本周该买/本月该卖"
  4. 结合持仓账本 ledger.json 计算具体买卖金额
  5. 有动作才推送; test_mode=True 时前缀标注

用法:
  python wind_new_search/push/push_daily.py            # 正常执行
  python wind_new_search/push/push_daily.py --dry-run  # 只打印不推送
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from wind_new_search.engine import mult_for, prep_df

PUSH_DIR = Path(__file__).resolve().parent
DATA_DIR = PUSH_DIR / "data"
LEDGER_PATH = PUSH_DIR / "ledger.json"
ETF_DIR = PROJECT_DIR / "data-store" / "parquet" / "etf"

WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send"


def load_config():
    with open(PUSH_DIR / "config.json") as f:
        return json.load(f)


def load_ledger():
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {}


def save_ledger(ledger):
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)


def load_index_data(code):
    path = DATA_DIR / f"{code}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def current_signal(df, params, strat, ledger_entry=None):
    """判断最新一行数据对应的买入/卖出信号。

    返回: dict(action, buy_mult, sell_ratio, pe_pct, pb_pct, fed_pct, price,
                buy_reason, sell_reason)
    复用引擎逻辑: mult_for 分档 + 均线制动, 卖出 heavy/extreme + 闸门确认。
    """
    row = df.iloc[-1]
    pe_pct = float(row["pe_pct"]) if not np.isnan(row["pe_pct"]) else None
    pb_pct = float(row["pb_pct"]) if "pb_pct" in row and not np.isnan(row["pb_pct"]) else None
    fed_pct = float(row["fed_pct"]) if "fed_pct" in row and not np.isnan(row["fed_pct"]) else None
    price = float(row["price"])

    out = {
        "action": "none", "buy_mult": 0.0, "sell_ratio": 0.0,
        "pe_pct": pe_pct, "pb_pct": pb_pct, "fed_pct": fed_pct, "price": price,
        "buy_reason": "", "sell_reason": "",
    }

    # ---- 卖出 (引擎 engine.py:428-506 同逻辑) ----
    sh, se = params["sell_heavy"], params["sell_extreme"]
    sell_mode = 0
    sell_signal = params.get("sell_signal", "PE")
    if sell_signal == "PE" and pe_pct is not None:
        sp = pe_pct
    elif sell_signal == "PB" and pb_pct is not None:
        sp = pb_pct
    else:
        sp = None
    if sp is not None:
        if sp >= se:
            sell_mode = 3
        elif sp >= sh:
            sell_mode = 2
    # 卖出闸门确认 (引擎: 任一闸门未到下限则不确认)
    sell_gates = params.get("sell_gate") or []
    sell_gate_floors = params.get("sell_gate_floor") or []
    if sell_mode >= 2 and sell_gates:
        gates = sell_gates if isinstance(sell_gates, list) else [sell_gates]
        floors = sell_gate_floors if isinstance(sell_gate_floors, list) else [sell_gate_floors]
        for g, floor in zip(gates, floors):
            gv = {"PE": pe_pct, "PB": pb_pct, "FED": fed_pct}.get(g)
            if gv is None or gv < floor:
                sell_mode = 0
                break
    if sell_mode >= 2:
        ratio = 0.20
        if sell_mode == 3:
            ratio = 0.50  # 无持仓期峰值, 保守取上限
        out["sell_ratio"] = ratio
        out["sell_reason"] = (f"{'PE' if sell_signal=='PE' else sell_signal}%={sp*100:.0f}%"
                              f" 触发{'极端卖出' if sell_mode==3 else '分批卖出'} (比例{ratio*100:.0f}%)")
        out["action"] = "sell"

    # ---- 买入 (引擎 engine.py:553-630 同逻辑) ----
    buy_signal = params.get("buy_signal", "PB")
    if buy_signal == "PE":
        be = pe_pct
    elif buy_signal == "PB":
        be = pb_pct
    else:
        be = fed_pct
    mult = mult_for(be, params["buy_floor"], params["buy_low"],
                    params["buy_mid"], params["buy_high"], strat["buy_mults"])
    # 买入闸门 (FED<=cap 且 PE<=cap)
    buy_gates = params.get("buy_gate") or []
    buy_gate_caps = params.get("buy_gate_cap") or []
    if mult > 0 and buy_gates:
        gates = buy_gates if isinstance(buy_gates, list) else [buy_gates]
        caps = buy_gate_caps if isinstance(buy_gate_caps, list) else [buy_gate_caps]
        for g, cap in zip(gates, caps):
            gv = {"PE": pe_pct, "PB": pb_pct, "FED": fed_pct}.get(g)
            if gv is None or gv > cap:
                mult = 0
                break
    # 均线软制动 (ma_window)
    ma_window = strat.get("ma_window")
    ma_below = strat.get("ma_below", 0.0)
    ma_above = strat.get("ma_above", 1.0)
    if mult > 0 and ma_window:
        sma = df["price"].rolling(ma_window, min_periods=ma_window).mean()
        s_ma = sma.iloc[-1]
        if not np.isnan(s_ma):
            if price < s_ma:
                mult = mult * ma_below
            else:
                mult = mult * ma_above
    if mult > 0:
        out["buy_mult"] = mult
        out["buy_reason"] = (f"{buy_signal}%={be*100:.0f}% 进入买入区 "
                             f"(档位倍数{mult:.1f}x)")
        out["action"] = "buy"

    return out


def calc_amounts(code, info, sig, ledger, etf_price=None):
    """结合账本计算具体买入/卖出金额与份数.

    买入: 倍数×1000 (现金投入, 不依赖价格)
    卖出: 仅当账本有持仓才有意义; 金额 = 份额 × ETF价 × 比例
    """
    if sig["action"] == "buy":
        return sig["buy_mult"] * 1000, 0.0
    # sell
    held = ledger.get(code, {})
    shares = held.get("shares", 0.0)
    px = etf_price if etf_price else sig["price"]
    sell_value = shares * px * sig["sell_ratio"]
    return 0.0, sell_value


def _fmt_pct(v):
    return f"{v*100:.0f}%" if v is not None and not np.isnan(v) else "—"


def holding_stats(code, info, df, ledger_entry):
    """持仓统计: 持仓金额, 持仓收益%, 持仓收益年化夏普.

    持仓金额 = 份额 × ETF最新收盘价   (账本记的是ETF份额, 成本也是ETF价格)
    持仓收益% = (ETF最新价 / 平均成本 - 1) * 100   (无持仓或无成本时 None)
    持仓收益夏普 = 基于ETF价格序列(近1年)月频收益计算的年化夏普
    """
    etf_code = info.get("etf")
    etf_price = None
    etf_series = None
    if etf_code:
        epath = ETF_DIR / f"{etf_code}.parquet"
        if epath.exists():
            etf = pd.read_parquet(epath)
            etf["date"] = pd.to_datetime(etf["date"])
            etf = etf.sort_values("date").reset_index(drop=True)
            etf_series = etf.set_index("date")["close"]
            etf_price = float(etf["close"].iloc[-1])

    shares = float(ledger_entry.get("shares", 0.0))
    avg_cost = float(ledger_entry.get("avg_cost", 0.0))
    value = shares * etf_price if etf_price else 0.0
    ret_pct = None
    if shares > 0 and avg_cost > 0 and etf_price:
        ret_pct = (etf_price / avg_cost - 1.0) * 100.0
    # 持仓收益夏普: 仅当有持仓且有足够ETF价格历史(近1年)
    sharpe = None
    if shares > 0 and etf_series is not None and len(etf_series) > 60:
        s = etf_series
        if len(s) > 252:
            s = s[s.index >= s.index[-1] - pd.Timedelta(days=365)]
        m = s.resample("ME").last().dropna()
        r = m.pct_change().dropna()
        if len(r) > 5 and r.std() > 0:
            sharpe = float(r.mean() / r.std() * np.sqrt(12))
    return {"value": value, "ret_pct": ret_pct, "sharpe": sharpe,
            "etf_price": etf_price, "etf_code": etf_code}


def distance_to_signals(sig, params, strat):
    """计算当前估值距各买卖阈值还差多少百分位.

    返回 dict: dist_buy (距买入区, 负值=已在区内), dist_sell (距卖出区),
              in_buy/in_sell 布尔, buy_key/sell_key 信号列名.
    """
    buy_signal = params.get("buy_signal", "PB")
    sell_signal = params.get("sell_signal", "PE")
    bv = {"PE": sig["pe_pct"], "PB": sig["pb_pct"], "FED": sig["fed_pct"]}.get(buy_signal)
    sv = {"PE": sig["pe_pct"], "PB": sig["pb_pct"], "FED": sig["fed_pct"]}.get(sell_signal)
    buy_thresh = params.get("buy_mid", 0.25)  # 进入买入区(2x档)的边界
    sell_thresh = params["sell_heavy"]         # 进入卖出区(heavy)的边界
    out = {
        "buy_key": buy_signal, "sell_key": sell_signal,
        "buy_thresh": buy_thresh, "sell_thresh": sell_thresh,
        "dist_buy": None, "dist_sell": None,
        "in_buy": False, "in_sell": False,
    }
    if bv is not None:
        out["dist_buy"] = (bv - buy_thresh) * 100.0
        out["in_buy"] = bv <= buy_thresh
    if sv is not None:
        out["dist_sell"] = (sell_thresh - sv) * 100.0
        out["in_sell"] = sv >= sell_thresh
    return out


def strategy_intro(strat, params):
    """策略介绍文本."""
    buy_signal = params.get("buy_signal", "PB")
    sell_signal = params.get("sell_signal", "PE")
    mults = strat["buy_mults"]
    buy_desc = (f"{buy_signal}%<{params['buy_floor']*100:.0f}% →{mults[0]}x | "
                f"<{params['buy_low']*100:.0f}% →{mults[1]}x | "
                f"<{params['buy_mid']*100:.0f}% →{mults[2]}x")
    gates = params.get("buy_gate")
    gate_txt = ""
    if gates:
        caps = params.get("buy_gate_cap")
        if isinstance(gates, list):
            gs = "且".join(f"{g}≤{c*100:.0f}%" for g, c in zip(gates, caps))
        else:
            gs = f"{gates}≤{caps*100:.0f}%"
        gate_txt = f" 闸门:{gs}"
    sell_desc = (f"{sell_signal}%≥{params['sell_heavy']*100:.0f}% →减20% | "
                 f"≥{params['sell_extreme']*100:.0f}% →减50%")
    principal = (f"本金{strat['principal_threshold']/10000:.0f}万软收缩/"
                 f"{strat['principal_cap']/10000:.0f}万封顶")
    return [
        f"📋 策略: {strat['name']}",
        f"  买入: {buy_desc}{gate_txt}",
        f"  卖出: {sell_desc}",
        f"  约束: {principal}",
    ]


def build_message(cfg, results):
    """完整决策看板: 策略介绍 + 今日信号 + 各指数估值/距买卖差 + 持仓概览."""
    strat = cfg["strategy"]
    params = strat["params"]
    lines = []
    lines.append("【理财助手 · 每日看板】")
    if cfg.get("test_mode"):
        lines.append("⚠️ 测试模式")
    lines.append("")
    lines.extend(strategy_intro(strat, params))

    # ---- 今日信号 ----
    lines.append("")
    buys = [r for r in results if r["action"] == "buy"]
    sells = [r for r in results if r["action"] == "sell"]
    if not buys and not sells:
        lines.append("📈 今日信号: 无买卖动作 ✅")
    for r in buys:
        lines.append(f"📈 {r['name']} 建议买入 ¥{r['amount']:,.0f} "
                     f"({r['buy_reason']})")
    for r in sells:
        lines.append(f"📉 {r['name']} 建议卖出 ¥{r['amount']:,.0f} "
                     f"({r['sell_reason']})")

    # ---- 各指数估值 + 距买卖区距离 ----
    lines.append("")
    lines.append("📊 各宽基估值(百分位/距买卖区):")
    for r in sorted(results, key=lambda x: x["code"]):
        d = r.get("dist", {})
        bk, sk = d.get("buy_key", "?"), d.get("sell_key", "?")
        bv = {"PE": r["pe_pct"], "PB": r["pb_pct"], "FED": r["fed_pct"]}.get(bk)
        sv = {"PE": r["pe_pct"], "PB": r["pb_pct"], "FED": r["fed_pct"]}.get(sk)
        # 距离文字
        if d.get("in_buy"):
            buy_txt = f"{bk}{_fmt_pct(bv)}·已入买区"
        elif d.get("dist_buy") is not None:
            buy_txt = f"{bk}{_fmt_pct(bv)}·距买还差{d['dist_buy']:.0f}%"
        else:
            buy_txt = f"{bk}—"
        if d.get("in_sell"):
            sell_txt = f"{sk}{_fmt_pct(sv)}·已入卖区"
        elif d.get("dist_sell") is not None:
            sell_txt = f"{sk}{_fmt_pct(sv)}·距卖还差{d['dist_sell']:.0f}%"
        else:
            sell_txt = f"{sk}—"
        h = r.get("holding")
        hold_txt = ""
        if h and h["value"] > 0:
            ret = f"{h['ret_pct']:+.1f}%" if h["ret_pct"] is not None else "—"
            sharpe = f"夏普{h['sharpe']:.2f}" if h["sharpe"] is not None else ""
            hold_txt = f" | 💰¥{h['value']:,.0f}({ret}{sharpe})"
        lines.append(f"  {r['name']}: {buy_txt} | {sell_txt}{hold_txt}")

    # ---- 持仓概览 ----
    lines.append("")
    held = [r for r in results if r.get("holding") and r["holding"]["value"] > 0]
    if held:
        total_value = sum(r["holding"]["value"] for r in held)
        total_cost = 0.0
        ledger = load_ledger()
        for r in held:
            total_cost += float(ledger.get(r["code"], {}).get("avg_cost", 0.0)) * \
                float(ledger.get(r["code"], {}).get("shares", 0.0))
        total_ret = (total_value / total_cost - 1) * 100 if total_cost > 0 else None
        ret_txt = f" 总收益{total_ret:+.1f}%" if total_ret is not None else ""
        lines.append(f"💼 持仓概览: {len(held)}只 | 市值¥{total_value:,.0f}{ret_txt}")
        for r in held:
            h = r["holding"]
            ret = f"{h['ret_pct']:+.1f}%" if h["ret_pct"] is not None else "—"
            sharpe = f" | 夏普{h['sharpe']:.2f}" if h["sharpe"] is not None else ""
            lines.append(f"  {r['name']}: ¥{h['value']:,.0f} ({ret}{sharpe})")
    else:
        lines.append("💼 持仓概览: 暂无持仓(账本未录入)")
    return "\n".join(lines)


def send_webhook(cfg, text):
    url = cfg["webhook_url"]
    r = requests.post(url, json={"msgtype": "text", "text": {"content": text}}, timeout=15)
    resp = r.json()
    if resp.get("errcode") != 0:
        print(f"推送失败: {resp}")
    return resp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印不推送")
    args = parser.parse_args()

    cfg = load_config()
    ledger = load_ledger()
    strat = cfg["strategy"]
    params = strat["params"]

    # 频率限制状态: 本周/本月 key (引擎口径: 买入每周≤1, 卖出每月≤1)
    latest_date = None
    for code, info in cfg["indices"].items():
        df = load_index_data(code)
        if df is not None and len(df):
            d = df["date"].max()
            latest_date = d if latest_date is None else max(latest_date, d)
    if latest_date is None:
        print("无数据")
        return
    this_week = f"{latest_date.isocalendar().year}-{latest_date.isocalendar().week}"
    this_month = f"{latest_date.year}-{latest_date.month}"

    results = []
    for code, info in cfg["indices"].items():
        df = load_index_data(code)
        if df is None or len(df) < 30:
            print(f"[SKIP] {code} 数据不足")
            continue
        entry = ledger.get(code, {})
        # 本周已买过 -> 不再提示买入
        if entry.get("last_buy_week") == this_week:
            pass
        # 本月已卖过 -> 不再提示卖出
        if entry.get("last_sell_month") == this_month:
            pass

        sig = current_signal(df, params, strat, entry)
        # 频率限制
        if sig["action"] == "buy" and entry.get("last_buy_week") == this_week:
            sig["action"] = "none"
            sig["buy_reason"] = ""
        if sig["action"] == "sell" and entry.get("last_sell_month") == this_month:
            sig["action"] = "none"
            sig["sell_reason"] = ""
        sig["holding"] = holding_stats(code, info, df, entry)
        amount, sell_value = calc_amounts(code, info, sig, ledger,
                                          etf_price=sig["holding"]["etf_price"])
        # 无持仓的卖出信号无意义 -> 作废
        if sig["action"] == "sell" and (amount + sell_value) <= 0:
            sig["action"] = "none"
            sig["sell_reason"] = ""
        sig["code"] = code
        sig["name"] = info["name"]
        sig["amount"] = amount if sig["action"] == "buy" else sell_value
        sig["dist"] = distance_to_signals(sig, params, strat)
        results.append(sig)
        if sig["action"] != "none":
            kind = "买入" if sig["action"] == "buy" else "卖出"
            print(f"[SIG] {info['name']} {kind}: ¥{sig['amount']:,.0f} "
                  f"({sig['buy_reason'] or sig['sell_reason']})")

    msg = build_message(cfg, results)
    print("\n" + "=" * 40)
    print(msg)

    if args.dry_run:
        print("\n[dry-run] 未推送")
        return

    resp = send_webhook(cfg, msg)
    if resp.get("errcode") == 0:
        print("\n✅ 已推送")
    else:
        print(f"\n❌ 推送失败: {resp}")


if __name__ == "__main__":
    main()
