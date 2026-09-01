#!/usr/bin/env python3
"""
新版 Wind 回测引擎 — 不对称「主信号 + 闸门」框架

买入: 由 buy_signal (PE/PB/FED) 的昂贵度百分位分档 (3x/2x/1x/0.5x)
      可选闸门 buy_gate (PE/PB/FED) + buy_gate_cap: 该指标昂贵度超过上限则本次买入作废
卖出: 由 sell_signal (PE/PB/FED) 的昂贵度百分位触发 (heavy/extreme)
分档: 每周最多 1 次买入, base=1000; 卖出每月最多 1 次

昂贵度百分位: 低=便宜, 高=贵。PE/PB 直接用其百分位; FED 高=便宜, 故取 1 - fed_pct。
旧版 signal_mode (PE_only/PB_only/FED_only) 由 normalize_params 映射为
  buy_signal=X, buy_gate=None, sell_signal=PE, 行为与原引擎完全一致。
"""

import numpy as np
import pandas as pd
from datetime import date as dt_date

try:
    from scipy.optimize import brentq
except Exception:  # pragma: no cover - 无 scipy 环境回退到二分法
    brentq = None

BASE_AMOUNT = 1000


def sharpe_annual(daily, ft, rf_annual=0.0):
    """本金曲线 (principal) 的夏普, 口径与既有页面一致。

    夏普 = (mean(收益) - 周期无风险利率) / std(收益) × √(每年周期数)
    自动按数据频率估算每年周期数 (周频≈52, 日频≈252), 无风险利率按周期换算。
    rf_annual: 年化无风险利率 (如 0.013 = 1.3%)。默认 0 表示"毛夏普"。
    """
    dts = pd.to_datetime([d["date"] for d in daily])
    mask = np.asarray(dts >= pd.Timestamp(ft)) if ft else np.ones(len(daily), dtype=bool)
    pr = np.array([d["principal"] for d, m in zip(daily, mask) if m and d.get("principal") is not None], dtype=float)
    if len(pr) < 3:
        return None
    r = pr[1:] / pr[:-1] - 1.0
    s = r.std(ddof=0)
    if s <= 0 or not np.isfinite(s):
        return None
    gap = (dts[1] - dts[0]).days if len(dts) >= 2 else 7
    ppy = 365.25 / max(float(gap), 1.0)
    rf_per = rf_annual / ppy
    return float((r.mean() - rf_per) / s * np.sqrt(ppy))


def calc_xirr(cashflows, final_date, final_value):
    """XIRR 年化收益率 (稳健求根). cashflows: [(date, amount)] 买=负 卖=正.

    在 -50%~+10000% 区间内按网格寻找 NPV 符号翻转, 用 brentq 求根。
    避免旧二分法在利率接近 -1 时 NPV 发散导致的误判(返回 0)。
    """
    if len(cashflows) < 3 or final_value < 0:
        return 0.0
    dates = [pd.Timestamp(d) for d, _ in cashflows]
    if (dates[-1] - dates[0]).days < 30:
        return 0.0
    amounts = [a for _, a in cashflows]
    dates.append(pd.Timestamp(final_date))
    amounts.append(final_value)
    t0 = dates[0]
    years = np.array([(d - t0).days / 365.25 for d in dates])
    amt_arr = np.array(amounts)

    def npv(rate):
        denom = (1.0 + rate) ** years
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return np.sum(amt_arr / denom)

    grid = [-0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    for i in range(len(grid) - 1):
        lo, hi = grid[i], grid[i + 1]
        flo = npv(lo)
        fhi = npv(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)):
            continue
        if flo == 0.0:
            return round(lo, 4)
        if fhi == 0.0:
            return round(hi, 4)
        if flo * fhi < 0:
            if brentq is not None:
                try:
                    root = brentq(npv, lo, hi, xtol=1e-6, maxiter=100)
                    return round(float(root), 4)
                except ValueError:
                    continue
            else:
                a, b, fa = lo, hi, flo
                for _ in range(200):
                    mid = 0.5 * (a + b)
                    fmid = npv(mid)
                    if not np.isfinite(fmid):
                        break
                    if fmid == 0.0 or (b - a) < 1e-9:
                        return round(float(mid), 4)
                    if fa * fmid < 0:
                        b = mid
                    else:
                        a, fa = mid, fmid
                return round(float(0.5 * (a + b)), 4)
    return 0.0


def mult_for(pct, bf, bl, bm, bh, mults=(3, 2, 1, 0.5)):
    """分档倍数: 越便宜(昂贵度越低)倍数越高. mults = [<bf, <bl, <bm, <bh] 对应倍数."""
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return 0
    if pct < bf:
        return mults[0]
    if pct < bl:
        return mults[1]
    if pct < bm:
        return mults[2]
    if pct < bh:
        return mults[3]
    return 0


def mult_exp(pct, A, k, pct_max, floor=0.0):
    """指数曲线买入倍数 + 中位地板: 越便宜(昂贵度越低)倍数越高, pct_max 处归零(超出不买).

    mult(pct) = floor + A * (e^(-k*pct) - e^(-k*pct_max)) / (1 - e^(-k*pct_max))
    pct=0 -> floor + A (峰值), pct=pct_max -> floor (地板), pct>=pct_max -> 0.
    floor 让中位区间(接近 pct_max)仍保留最小买入力度, 服务横盘市(如300).
    """
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return 0.0
    if pct_max is None or pct >= pct_max:
        return 0.0
    denom = 1.0 - float(np.exp(-k * pct_max))
    if denom <= 0:
        return float(floor)
    exp_part = A * (float(np.exp(-k * pct)) - float(np.exp(-k * pct_max))) / denom
    return float(floor) + exp_part


def _principal_pct_max_scale(total_invested, threshold):
    """指数曲线下本金阈值收缩: 本金越高, 买入上限 pct_max 越往回收(更挑剔).

      < threshold           -> 1.00 (原样)
      [threshold, 1.2x)     -> 0.88
      [1.2x, 1.4x]          -> 0.75
      > 1.4x                -> 0.60
    """
    if not threshold or threshold <= 0:
        return 1.0
    if total_invested < threshold:
        return 1.0
    if total_invested < threshold * 1.2:
        return 0.88
    if total_invested <= threshold * 1.4:
        return 0.75
    return 0.60


def _principal_min_mult(total_invested, threshold):
    """本金阈值: 累计买入本金越高, 只保留更强(更高倍数)档位, 返回允许的最小倍数.

    规则:
      < threshold           -> 0.5x/1x/2x/3x  (原样)
      [threshold, 1.2x)     -> 1x/2x/3x       (去掉 0.5x)
      [1.2x, 1.4x]          -> 2x/3x          (去掉 1x)
      > 1.4x                -> 仅 3x          (去掉 2x)
    """
    if not threshold or threshold <= 0:
        return 0.5
    if total_invested < threshold:
        return 0.5
    if total_invested < threshold * 1.2:
        return 1.0
    if total_invested <= threshold * 1.4:
        return 2.0
    return 3.0


def _principal_band(total_invested, threshold):
    """本金所处档位描述 (用于交易触发说明)."""
    if not threshold or threshold <= 0:
        return ""
    w = threshold / 10000.0
    if total_invested < threshold:
        return f"本金<{w:.0f}万: 0.5x/1x/2x/3x"
    if total_invested < threshold * 1.2:
        return f"本金{w:.0f}~{w * 1.2:.0f}万: 1x/2x/3x"
    if total_invested <= threshold * 1.4:
        return f"本金{w * 1.2:.0f}~{w * 1.4:.0f}万: 2x/3x"
    return f"本金>{w * 1.4:.0f}万: 仅3x"


def _expensive_pct(signal, pe, pb, fed):
    """各信号的'昂贵度百分位'(低=便宜, 高=贵)."""
    if signal == "PE":
        return pe
    if signal == "PB":
        return pb
    if signal == "FED":
        return 1.0 - fed  # 高 FED = 便宜, 故昂贵度取反
    return None


def normalize_params(params):
    """兼容旧格式 (signal_mode) 与新格式 (buy_signal/buy_gate/buy_gate_cap/sell_signal)."""
    p = dict(params)
    if "buy_signal" not in p:
        mode = p.pop("signal_mode", "PB_only")
        m = {"PE_only": "PE", "PB_only": "PB", "FED_only": "FED"}.get(mode, "PB")
        p["buy_signal"] = m
        p["buy_gate"] = None
        p["buy_gate_cap"] = None
        p["sell_signal"] = "PE"
    p.setdefault("buy_gate", None)
    p.setdefault("buy_gate_cap", None)
    p.setdefault("sell_signal", "PE")
    p.setdefault("sell_gate", None)
    p.setdefault("sell_gate_floor", None)
    return p


def prep_df(df):
    """预计算年月/周 key 数组, 避免回测内重复解析日期."""
    d = df["date"]
    df = df.copy()
    df["_ym"] = (d.dt.year * 12 + d.dt.month).values
    iso = d.dt.isocalendar()
    df["_wk"] = (iso["week"] + d.dt.year * 53).values
    return df


def hurst_rs(x):
    """单段 R/S 法估计赫斯特指数 H∈(0,1). 输入为对数价格序列.

    H>0.5 趋势有持续性; H<0.5 均值回归/震荡; H=0.5 随机游走.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return np.nan
    x = x - x.mean()
    z = np.cumsum(x)
    R = float(z.max() - z.min())
    S = float(x.std(ddof=0))
    if S <= 0 or R <= 0:
        return 0.5
    H = float(np.log(R / S) / np.log(n))
    return float(np.clip(H, 0.0, 1.0))


def run_backtest(df, params, base_amount=BASE_AMOUNT, min_trades=0,
                 exec_price=None, commission_rate=0.0, min_commission=0.0, lot_size=0,
                 principal_threshold=None, principal_cap=None, principal_pool=None,
                 buy_mults=None, buy_curve=None, ma_window=None,
                 ma_below=0.0, ma_above=1.0, ma_sell_ratio=None,
                 hurst_window=None, hurst_discount=None, hurst_boost=None,
                 hurst_sell_up=None, hurst_sell_down=None,
                 hurst_pct_window=None, hurst_pct_lo=0.4, hurst_pct_mid=0.6, hurst_pct_hi=0.8,
                 hurst_f_lo=1.15, hurst_f_mid_hi=0.85, hurst_f_hi=0.7,
                 trail_stop=None, trail_stop_ratio=1.0,
                 vol_window=None, vol_target=None,
                 equity_stop=None, equity_stop_ratio=1.0):
    """
    单次回测。df 需含列: date, price, pe_pct, pb_pct, fed_pct (可选 _ym/_wk)
    params: dict(buy_signal, buy_gate, buy_gate_cap, sell_signal,
                 buy_floor, buy_low, buy_mid, buy_high, sell_heavy, sell_extreme)

    ETF 变体参数:
      exec_price: 与 df 等长的执行价数组(如 ETF 后复权价), 缺省用 df["price"]
      commission_rate: 佣金率(成交额比例), 默认 0 表示不收
      min_commission: 单笔最低佣金(元), 默认 0
      lot_size: 整手股数(如 100), 默认 0 表示允许零股(指数口径)
      principal_threshold: 本金阈值(元), 净占用本金(累计买入-卖出回笼)越高只保留更强档位; None 表示不启用
      principal_cap: 本金硬封顶(元), 净占用本金(累计买入-卖出回笼, 含费)达该值后停止买入; None 表示不启用
      principal_pool: 固定本金池(元), 设值后额外输出固定口径的账户终值/回报/年化
      ma_window: 买入侧趋势制动阀均线窗口(周). 设值后按 price 与 SMA(ma_window) 关系调制买入倍数;
                 前 ma_window-1 周无均线值按不制动处理(warm-up 照常交易). None 表示不启用
      ma_below: price < SMA 时买入倍数缩放系数 (0.0=全刹不买, 0~1 软刹, 默认 0.0 保持旧行为)
      ma_above: price >= SMA 时买入倍数缩放系数 (1.0=不加力, >1 企稳加力, 默认 1.0)
      ma_sell_ratio: 卖出侧趋势退出比例. 设值后 price < SMA 时每月减仓该比例 (1.0=清仓);
                     与估值卖出互斥, 趋势退出优先. None 表示不启用(默认)
      hurst_window: 赫斯特指数滚动窗口(周). 设值后按 H>0.5 二值触发对买入/卖出倍数调制 (信心指数); None 表示不启用
      hurst_discount: 下跌趋势(H>0.5 且方向跌)时买入倍数打折系数 (0<值<=1); None 表示不打折
      hurst_boost: 上涨趋势(H>0.5 且方向涨)时买入倍数加码系数 (值>=1); None 表示不加码
      hurst_sell_up: 上涨趋势(H>0.5 且方向涨)时卖出比例缩放系数 (0<=值<=1, 抑制卖飞); None 表示不调制
      hurst_sell_down: 下跌趋势(H>0.5 且方向跌)时卖出比例缩放系数 (值>=1, 加速卖出); None 表示不调制
      hurst_pct_window: H 的滚动百分位回看窗口(周). 设值后按当前 H 在其自身历史中的分位调制买入倍数; None 表示不启用
      hurst_pct_lo/mid/hi: H 百分位的下/中/上分带阈值 (默认 0.4/0.6/0.8)
      hurst_f_lo / hurst_f_mid_hi / hurst_f_hi: H 百分位 <lo / (mid,hi] / >hi 时的买入倍数系数 (默认 1.15 / 0.85 / 0.7)
      (H<=0.5 震荡/均值回归时不调制, 维持原档位)
      trail_stop: 回撤止损比例. 持仓期间跟踪持有期最高价, 价格自最高价回撤 ≥ trail_stop
                  即卖出 (默认触发时全清, 可用 trail_stop_ratio 指定卖出比例). None 表示不启用
      trail_stop_ratio: 回撤止损触发时的卖出比例 (1.0=清仓, 0~1 部分减仓)
      vol_window / vol_target: 波动率目标买入调制. 以 price 近 vol_window 日日收益标准差为
                  已实现波动率(年化 sqrt(252)), 买入倍数乘 clamp(vol_target/realized_vol, 0.3, 2.5);
                  高波动少买、低波动多买, 平滑组合波动. 均设值才启用
      equity_stop: 组合净值回撤止损比例. 以 (持仓市值+已实现现金) 组合净值峰值为基准,
                  回撤 ≥ equity_stop 即卖出 equity_stop_ratio 比例. None 表示不启用
      equity_stop_ratio: 组合净值回撤触发时的卖出比例
    """
    params = normalize_params(params)
    if buy_mults is None:
        buy_mults = (3.0, 2.0, 1.0, 0.5)
    buy_signal = params["buy_signal"]
    buy_gate = params.get("buy_gate")
    buy_gate_cap = params.get("buy_gate_cap")
    sell_signal = params["sell_signal"]
    sell_gate = params.get("sell_gate")
    sell_gate_floor = params.get("sell_gate_floor")
    # 闸门支持多个信号: 字符串或数组统一为列表, 每个闸门都要满足 (AND 关系)
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

    dates = df["date"].values
    prices = df["price"].values
    if exec_price is None:
        exec_prices = prices
    else:
        exec_prices = np.asarray(exec_price, dtype=float)
    pe_pcts = df["pe_pct"].values.astype(float)
    pb_pcts = df["pb_pct"].values.astype(float)
    fed_pcts = df["fed_pct"].values.astype(float)

    # 买入侧趋势制动阀: 收盘价(指数价)的 ma_window 周简单均线
    if ma_window:
        sma = pd.Series(prices).rolling(ma_window, min_periods=ma_window).mean().values
    else:
        sma = None

    # 赫斯特指数信心指数: 滚动窗口内 H + 趋势方向 d
    n = len(df)
    if hurst_window:
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(prices > 0, np.log(prices), np.nan)
        hurst_h = np.full(n, np.nan)
        hurst_d = np.full(n, np.nan)
        for i in range(hurst_window - 1, n):
            hurst_h[i] = hurst_rs(logp[i - hurst_window + 1: i + 1])
            p0 = prices[i - hurst_window + 1]
            p1 = prices[i]
            if np.isfinite(p0) and np.isfinite(p1):
                hurst_d[i] = 1.0 if p1 >= p0 else -1.0
    else:
        hurst_h = None
        hurst_d = None

    # H 的滚动百分位: 当前 H 在过去 hurst_pct_window 个 H 值中的分位 (相对趋势强度)
    if hurst_window and hurst_pct_window:
        hurst_pct = np.full(n, np.nan)
        min_hist = max(20, hurst_pct_window // 2)
        for i in range(hurst_window - 1 + hurst_pct_window, n):
            w = hurst_h[i - hurst_pct_window: i]
            w = w[~np.isnan(w)]
            if len(w) < min_hist:
                continue
            hurst_pct[i] = float((w < hurst_h[i]).sum() / len(w))
    else:
        hurst_pct = None

    # 已实现波动率 (波动率目标买入调制): 日收益滚动标准差, 年化 sqrt(252)
    if vol_window and vol_target:
        rets = np.diff(prices) / prices[:-1]
        vol_ts = pd.Series(rets).rolling(vol_window, min_periods=vol_window).std().values
        realized_vol = np.full(n, np.nan)
        realized_vol[1:] = vol_ts * np.sqrt(252.0)
    else:
        realized_vol = None

    buy_exp = _expensive_pct(buy_signal, pe_pcts, pb_pcts, fed_pcts)
    gate_exps = [_expensive_pct(g, pe_pcts, pb_pcts, fed_pcts) for g in buy_gates]
    sell_exp = _expensive_pct(sell_signal, pe_pcts, pb_pcts, fed_pcts)
    sell_gate_exps = [_expensive_pct(g, pe_pcts, pb_pcts, fed_pcts) for g in sell_gates]

    if "_ym" in df.columns:
        ym_arr = df["_ym"].values
        wk_arr = df["_wk"].values
    else:
        d = df["date"]
        ym_arr = (d.dt.year * 12 + d.dt.month).values
        iso = d.dt.isocalendar()
        wk_arr = (iso["week"] + d.dt.year * 53).values

    shares = 0.0
    total_invested = 0.0
    total_cash_in = 0.0
    peak_price = 0.0
    trail_peak = 0.0  # 回撤止损的持有期最高价 (仅 trail_stop 启用时使用)
    peak_equity = 0.0  # 组合净值回撤止损的净值峰值 (仅 equity_stop 启用时使用)
    trades = []
    first_tradable = None  # 回测区间第一天 (首个 pe_pct 有效日)

    last_buy_week = -1
    last_sell_month = -1
    sell_month_done = False
    after_sell_cooldown = False

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
            after_sell_cooldown = False

        # ---- 卖出 (基于 sell_signal 昂贵度, 每月最多 1 次, 可选确认闸门) ----
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
                    sell_mode = 0  # 任一闸门未到昂贵度下限, 不确认卖出
                    break

        can_sell = not sell_month_done
        trend_exit = False
        if sma is not None and ma_sell_ratio is not None:
            s_ma = sma[i]
            if not np.isnan(s_ma) and price < s_ma:
                trend_exit = True

        if trend_exit and shares > 0 and can_sell:
            s = shares * ma_sell_ratio
            if lot_size > 0:
                s = int(s / lot_size) * lot_size
            if s > 0:
                gross = s * px
                comm = max(gross * commission_rate, min_commission) if commission_rate > 0 else 0.0
                cash_in = gross - comm
                shares -= s
                total_cash_in += cash_in
                trades.append((dt_str, "sell", -cash_in, shares, px, float(pe_pct),
                               float(pb_pct) if not np.isnan(pb_pct) else None, total_invested))
                sell_month_done = True
                last_sell_month = year_month
                after_sell_cooldown = True
        elif sell_mode >= 2 and shares > 0 and can_sell:
            ratio = 0.20
            if sell_mode == 3:
                peak_price = max(peak_price, px)
                if peak_price > 0:
                    dd = (peak_price - px) / peak_price
                    if dd >= 0.05:
                        ratio = min(0.25 + dd * 0.3, 0.50)
                    else:
                        sell_mode = 0  # 等回撤
            if sell_mode >= 2 and hurst_h is not None:
                h = hurst_h[i]
                d = hurst_d[i]
                if not np.isnan(h) and not np.isnan(d) and h > 0.5:
                    if d >= 0:
                        if hurst_sell_up is not None:
                            ratio = ratio * hurst_sell_up
                    else:
                        if hurst_sell_down is not None:
                            ratio = ratio * hurst_sell_down
            if sell_mode >= 2:
                s = shares * ratio
                if lot_size > 0:
                    s = int(s / lot_size) * lot_size
                if s <= 0:
                    sell_mode = 0
            if sell_mode >= 2:
                gross = s * px
                comm = max(gross * commission_rate, min_commission) if commission_rate > 0 else 0.0
                cash_in = gross - comm
                shares -= s
                total_cash_in += cash_in
                act = "clear" if sell_mode == 3 else "sell"
                trades.append((dt_str, act, -cash_in, shares, px, float(pe_pct),
                               float(pb_pct) if not np.isnan(pb_pct) else None, total_invested))
                sell_month_done = True
                last_sell_month = year_month
                after_sell_cooldown = True
                if sell_mode == 3:
                    peak_price = 0

        # ---- 回撤止损 (trailing stop): 自持仓期最高价回撤 trail_stop 即卖出 ----
        if trail_stop and trail_stop > 0 and shares > 0 and can_sell:
            if px > trail_peak:
                trail_peak = px
            elif trail_peak > 0 and px <= trail_peak * (1 - trail_stop):
                s = shares * trail_stop_ratio
                if lot_size > 0:
                    s = int(s / lot_size) * lot_size
                if s > 0:
                    gross = s * px
                    comm = max(gross * commission_rate, min_commission) if commission_rate > 0 else 0.0
                    cash_in = gross - comm
                    shares -= s
                    total_cash_in += cash_in
                    trades.append((dt_str, "sell", -cash_in, shares, px, float(pe_pct),
                                   float(pb_pct) if not np.isnan(pb_pct) else None, total_invested))
                    sell_month_done = True
                    last_sell_month = year_month
                    after_sell_cooldown = True
                    if trail_stop_ratio >= 1.0:
                        trail_peak = 0

        # ---- 组合净值回撤止损 (equity trailing stop) ----
        if equity_stop and equity_stop > 0 and shares > 0 and can_sell:
            eq = shares * px + total_cash_in
            if eq > peak_equity:
                peak_equity = eq
            elif peak_equity > 0 and eq <= peak_equity * (1 - equity_stop):
                s = shares * equity_stop_ratio
                if lot_size > 0:
                    s = int(s / lot_size) * lot_size
                if s > 0:
                    gross = s * px
                    comm = max(gross * commission_rate, min_commission) if commission_rate > 0 else 0.0
                    cash_in = gross - comm
                    shares -= s
                    total_cash_in += cash_in
                    trades.append((dt_str, "sell", -cash_in, shares, px, float(pe_pct),
                                   float(pb_pct) if not np.isnan(pb_pct) else None, total_invested))
                    sell_month_done = True
                    last_sell_month = year_month
                    after_sell_cooldown = True
                    if equity_stop_ratio >= 1.0:
                        peak_equity = 0

        # ---- 买入 (每周最多 1 次, 卖出当月不买) ----
        can_buy = (week_key != last_buy_week) and not after_sell_cooldown
        if can_buy:
            net_invested = total_invested - total_cash_in  # 净占用本金(卖出回笼现金可再买入)
            if buy_curve is not None:
                pct = buy_exp[i]
                scale = _principal_pct_max_scale(net_invested, principal_threshold) if principal_threshold else 1.0
                eff_max = buy_curve["pct_max"] * scale
                mult = mult_exp(pct, buy_curve["A"], buy_curve["k"], eff_max,
                                buy_curve.get("floor", 0.0))
            else:
                mult = mult_for(buy_exp[i], bf, bl, bm, bh, buy_mults)
            if mult > 0 and buy_gates:
                for gp, cap in zip(gate_exps, buy_gate_caps):
                    if cap is None:
                        continue
                    gv = gp[i]
                    if np.isnan(gv) or gv > cap:
                        mult = 0  # 任一闸门昂贵度超上限, 本次买入作废
                        break
            if mult > 0 and buy_curve is None and principal_threshold:
                min_mult = _principal_min_mult(net_invested, principal_threshold)
                if mult < min_mult:
                    mult = 0  # 净占用本金已达阈值, 该档位被禁用
            if mult > 0 and sma is not None:
                s = sma[i]
                if not np.isnan(s):
                    if price < s:
                        mult = mult * ma_below
                    else:
                        mult = mult * ma_above
            if mult > 0 and hurst_h is not None:
                h = hurst_h[i]
                d = hurst_d[i]
                if not np.isnan(h) and not np.isnan(d):
                    if h > 0.5:
                        if d >= 0:
                            if hurst_boost is not None:
                                mult = mult * hurst_boost
                        else:
                            if hurst_discount is not None:
                                mult = mult * hurst_discount
            if mult > 0 and hurst_pct is not None:
                p = hurst_pct[i]
                if not np.isnan(p):
                    if p > hurst_pct_hi:
                        mult = mult * hurst_f_hi
                    elif p > hurst_pct_mid:
                        mult = mult * hurst_f_mid_hi
                    elif p < hurst_pct_lo:
                        mult = mult * hurst_f_lo
            if mult > 0 and realized_vol is not None:
                rv = realized_vol[i]
                if not np.isnan(rv) and rv > 0:
                    scale = float(np.clip(vol_target / rv, 0.3, 2.5))
                    mult = mult * scale
            if mult > 0:
                amt = base_amount * mult
                if lot_size > 0:
                    s = int(amt / px / lot_size) * lot_size
                    if s <= 0:
                        s = 0.0
                    cost = s * px
                else:
                    s = amt / px
                    cost = amt
                if s <= 0:
                    last_buy_week = week_key
                    continue
                comm = max(cost * commission_rate, min_commission) if commission_rate > 0 else 0.0
                if principal_cap and (net_invested + cost + comm) > principal_cap:
                    last_buy_week = week_key
                    continue  # 本金硬封顶: 净占用本金(含费)超上限, 本次买入作废
                shares += s
                total_invested += cost + comm
                trades.append((dt_str, "buy", cost + comm, shares, px, float(pe_pct),
                               float(pb_pct) if not np.isnan(pb_pct) else None, total_invested))
                last_buy_week = week_key

    final_price = float(exec_prices[-1]) if len(exec_prices) else 0
    pos_value = shares * final_price if shares > 0 else 0
    total_value = pos_value + total_cash_in  # 总资产(持仓 + 已实现现金)
    invested_base = total_invested if total_invested > 0 else 1
    final_return = (total_value - total_invested) / invested_base

    principal_final = None
    principal_return = None
    principal_annual = None
    if principal_pool and principal_pool > 0:
        principal_final = principal_pool + total_value - total_invested
        principal_return = (principal_final - principal_pool) / principal_pool
        if first_tradable is not None:
            years = (pd.Timestamp(dates[-1]) - pd.Timestamp(first_tradable)).days / 365.25
        else:
            years = 0.0
        if years > 0 and principal_final > 0:
            principal_annual = (principal_final / principal_pool) ** (1.0 / years) - 1.0
        else:
            principal_annual = 0.0

    # 现金流: 买=流出(负), 卖=流入(正); 终值=剩余持仓市值 (已实现现金已在现金流中)
    cashflows = [(t[0], -t[2]) for t in trades]
    xirr = calc_xirr(cashflows, str(dates[-1])[:10], pos_value) if len(cashflows) >= 3 else 0.0

    buys = sum(1 for t in trades if t[1] == "buy")
    sells = sum(1 for t in trades if t[1] in ("sell", "clear"))

    return {
        "xirr": xirr,
        "final_return": round(final_return, 4),
        "total_invested": round(total_invested, 0),
        "total_cash_in": round(total_cash_in, 0),
        "final_value": round(total_value, 0),
        "position_value": round(pos_value, 0),
        "trades": len(trades),
        "buys": buys,
        "sells": sells,
        "cash_flows": trades,
        "principal_final": round(principal_final, 0) if principal_final is not None else None,
        "principal_return": round(principal_return, 4) if principal_return is not None else None,
        "principal_annual": round(principal_annual, 4) if principal_annual is not None else None,
    }


def _pct_cmp(signal, op, value):
    """把'昂贵度'上的比较转成该信号原生百分位口径的可读文本.

    PE/PB: 昂贵度 == 百分位, 方向不变。
    FED : 昂贵度 = 1 - 百分位, 需反转方向与比较符。
    op ∈ {'<', '<=', '>', '>='}
    """
    disp = {"<": "<", "<=": "≤", ">": ">", ">=": "≥"}
    if signal == "FED":
        flip = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
        return f"{signal}%{disp[flip[op]]}{1 - value:.0%}"
    return f"{signal}%{disp[op]}{value:.0%}"


def _reason(params, act, pe_pct, pb_pct, principal_before=None, principal_threshold=None):
    """生成交易触发说明."""
    params = normalize_params(params)
    buy_signal = params["buy_signal"]
    buy_gate = params.get("buy_gate")
    buy_gate_cap = params.get("buy_gate_cap")
    sell_signal = params["sell_signal"]
    sell_gate = params.get("sell_gate")
    sell_gate_floor = params.get("sell_gate_floor")
    def _gates(v, cap):
        if v is None:
            return []
        return list(zip(list(v), list(cap))) if isinstance(v, (list, tuple)) else [(v, cap)]
    bf, bl, bm, bh = params["buy_floor"], params["buy_low"], params["buy_mid"], params["buy_high"]
    sh, se = params["sell_heavy"], params["sell_extreme"]
    if act == "buy":
        if buy_signal == "FED":
            s = f"{buy_signal}% 进入买入区 (>{1-bf:.0%}/{1-bl:.0%}/{1-bm:.0%}/{1-bh:.0%})"
        else:
            s = f"{buy_signal}% 进入买入区 (<{bf:.0%}/{bl:.0%}/{bm:.0%}/{bh:.0%})"
        for g, c in _gates(buy_gate, buy_gate_cap):
            if c is not None:
                s += " 且 " + _pct_cmp(g, "<=", c)
        if principal_threshold and principal_before is not None:
            s += f" · {_principal_band(principal_before, principal_threshold)}"
        return s
    if act == "sell":
        s = _pct_cmp(sell_signal, ">=", sh) + " → 卖出"
        for g, f in _gates(sell_gate, sell_gate_floor):
            if f is not None:
                s += " 且 " + _pct_cmp(g, ">=", f)
        return s
    return _pct_cmp(sell_signal, ">=", se) + " 回撤 → 清仓"


def build_curve(df, params, base_amount=BASE_AMOUNT,
                exec_price=None, commission_rate=0.0, min_commission=0.0, lot_size=0,
                principal_threshold=None, principal_cap=None, principal_pool=None,
                buy_mults=None, buy_curve=None, ma_window=None,
                ma_below=0.0, ma_above=1.0, ma_sell_ratio=None,
                hurst_window=None, hurst_discount=None, hurst_boost=None,
                hurst_sell_up=None, hurst_sell_down=None,
                hurst_pct_window=None, hurst_pct_lo=0.4, hurst_pct_mid=0.6, hurst_pct_hi=0.8,
                hurst_f_lo=1.15, hurst_f_mid_hi=0.85, hurst_f_hi=0.7,
                trail_stop=None, trail_stop_ratio=1.0,
                vol_window=None, vol_target=None,
                equity_stop=None, equity_stop_ratio=1.0):
    """构建前端所需的回测曲线 (meta + daily + trades).

    ETF 变体: 传入 exec_price(ETF 后复权价数组) + 佣金/整手参数, 执行价与展示价改用 ETF 价。
    principal_threshold: 本金阈值(元), None 表示不启用。
    principal_cap: 本金硬封顶(元), None 表示不启用。
    principal_pool: 固定本金池(元), 设值后 daily 额外输出本金曲线/固定口径年化/日收益。
    """
    df = df.sort_values("date").reset_index(drop=True)
    params = normalize_params(params)
    r = run_backtest(df, params, base_amount=base_amount, exec_price=exec_price,
                     commission_rate=commission_rate, min_commission=min_commission,
                     lot_size=lot_size, principal_threshold=principal_threshold,
                     principal_cap=principal_cap, principal_pool=principal_pool,
                     buy_mults=buy_mults, buy_curve=buy_curve, ma_window=ma_window,
                     ma_below=ma_below, ma_above=ma_above, ma_sell_ratio=ma_sell_ratio,
                     hurst_window=hurst_window, hurst_discount=hurst_discount,
                     hurst_boost=hurst_boost, hurst_sell_up=hurst_sell_up,
                     hurst_sell_down=hurst_sell_down,
                     hurst_pct_window=hurst_pct_window, hurst_pct_lo=hurst_pct_lo,
                     hurst_pct_mid=hurst_pct_mid, hurst_pct_hi=hurst_pct_hi,
                     hurst_f_lo=hurst_f_lo, hurst_f_mid_hi=hurst_f_mid_hi,
                     hurst_f_hi=hurst_f_hi,
                     trail_stop=trail_stop, trail_stop_ratio=trail_stop_ratio,
                     vol_window=vol_window, vol_target=vol_target,
                     equity_stop=equity_stop, equity_stop_ratio=equity_stop_ratio)
    flows = r["cash_flows"]

    # 按日期聚合交易
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
    cf = []
    last_xirr = None
    first_invest = None
    prev_principal = None
    first_tradable = None  # 回测区间第一天 (首个 pe_pct 有效日)

    for i in range(len(df)):
        d = dates[i]
        if first_tradable is None and not np.isnan(pe_pct[i]) and 0 <= pe_pct[i] <= 1:
            first_tradable = d
        buy_amt = 0.0
        sell_amt = 0.0
        if d in flow_by_date:
            for t in flow_by_date[d]:
                act, amt = t[1], float(t[2])
                if act == "buy":
                    cf.append((d, -amt))
                    cum_invested += amt
                    buy_amt += amt
                else:
                    cf.append((d, -amt))  # -amt = +现金流入
                    cash += abs(amt)
                    sell_amt += abs(amt)
                shares = float(t[3])
            if len(cf) >= 3 and shares * prices[i] > 0:
                last_xirr = calc_xirr(cf, d, shares * prices[i])
        if first_invest is None and cum_invested > 0:
            first_invest = d

        eq = shares * prices[i] if shares > 0 else 0.0
        total_value = eq + cash
        ret = (total_value - cum_invested) / cum_invested * 100 if cum_invested > 0 else 0.0
        row = {
            "date": d, "price": round(float(prices[i]), 2),
            "pe": None if np.isnan(pe[i]) else round(float(pe[i]), 2),
            "pb": None if np.isnan(pb[i]) else round(float(pb[i]), 2),
            "fed": None if np.isnan(fed[i]) else round(float(fed[i]), 2),
            "pe_pct": None if np.isnan(pe_pct[i]) else round(float(pe_pct[i]), 4),
            "pb_pct": None if np.isnan(pb_pct[i]) else round(float(pb_pct[i]), 4),
            "fed_pct": None if np.isnan(fed_pct[i]) else round(float(fed_pct[i]), 4),
            "cum_invested": round(cum_invested, 0),
            "equity": round(eq, 0), "cash": round(cash, 0), "total_value": round(total_value, 0),
            "return_pct": round(ret, 2), "xirr": last_xirr,
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
        }
        if principal_pool:
            principal = principal_pool + total_value - cum_invested
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
        principal_before = (inv - amt) if act == "buy" else None
        trades.append({
            "date": d, "action": act, "amount": round(float(amt), 0),
            "shares": round(float(sh), 4), "price": round(float(pr), 2),
            "pe_pct": round(float(ppct), 4) if ppct is not None and not np.isnan(ppct) else None,
            "pb_pct": round(float(ppb), 4) if ppb is not None and not np.isnan(ppb) else None,
            "cum_invested": round(float(inv), 0),
            "reason": _reason(params, act, ppct, ppb,
                              principal_before=principal_before,
                              principal_threshold=principal_threshold),
        })

    meta = {
        "signal_mode": params["buy_signal"], "params": params,
        "total_invested": r["total_invested"], "final_value": r["final_value"],
        "position_value": r["position_value"], "total_cash_in": r["total_cash_in"],
        "trades": r["trades"], "buys": r["buys"], "sells": r["sells"],
        "xirr": r["xirr"], "final_return": r["final_return"],
        "first_invest": first_invest,
        "first_tradable": first_tradable,
        "principal_threshold": principal_threshold,
        "principal_cap": principal_cap,
        "principal_pool": principal_pool,
        "principal_final": r.get("principal_final"),
        "principal_return": r.get("principal_return"),
        "principal_annual": r.get("principal_annual"),
    }
    return {"meta": meta, "daily": daily, "trades": trades}
