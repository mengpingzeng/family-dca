#!/usr/bin/env python3
"""PE-DCA 可视化服务 — FastAPI + ECharts"""

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# 滚动百分位（内联避免循环导入）
def _rolling_pct(series, window_rows, min_samples=20):
    import numpy as np
    n = len(series)
    result = np.full(n, np.nan)
    arr = np.asarray(series, dtype=float)
    clean = ~np.isnan(arr)
    for i in range(n):
        if not clean[i]:
            continue
        start = max(0, i - window_rows)
        wc = clean[start:i+1]
        if wc.sum() < min_samples:
            continue
        w = arr[start:i+1][wc]
        result[i] = (w <= arr[i]).sum() / len(w)
    return result

app = FastAPI(title="PE-DCA 指标数据库")

BASE = Path(__file__).parent
MERGED_DIR = BASE / "data-store" / "parquet" / "merged"
WIND_DIR = BASE / "data-store" / "parquet" / "wind_source"
ALIGNED_DIR = BASE / "data-store" / "parquet" / "aligned_source"
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

INDEX_NAMES = {
    "000300": ("沪深300", "宽基"), "000905": ("中证500", "宽基"),
    "000852": ("中证1000", "宽基"), "000016": ("上证50", "宽基"),
    "000688": ("科创50", "宽基"), "000510": ("中证A500", "宽基"),
    "399006": ("创业板指", "宽基"), "399330": ("深证100", "宽基"),
    "000015": ("上证红利", "红利"), "000922": ("中证红利", "红利"),
    "930955": ("红利低波100", "红利"), "930915": ("港股通高股息", "港股通"),
    "930930": ("港股综合", "港股通"), "930931": ("港股通50", "港股通"),
    "931573": ("港股通科技", "港股通"), "930939": ("中证质量成长", "质量"),
    "HSI": ("恒生指数", "港股"), "HSTECH": ("恒生科技", "港股"),
    "NDX100": ("纳斯达克100", "美股"), "SPX500": ("标普500", "美股"),
}


def _read_index(code: str) -> pd.DataFrame:
    path = MERGED_DIR / f"{code}.parquet"
    if not path.exists():
        raise HTTPException(404, f"指数 {code} 不存在")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.where(pd.notna(df), None)


# ============================================================================
# 页面
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def index_page():
    with open(BASE / "templates" / "index.html") as f:
        return f.read()


@app.get("/detail/{code}", response_class=HTMLResponse)
def detail_page(code: str):
    path = MERGED_DIR / f"{code}.parquet"
    if not path.exists():
        raise HTTPException(404, "指数不存在")
    with open(BASE / "templates" / "detail.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)


@app.get("/grid-search", response_class=HTMLResponse)
def grid_search_page():
    with open(BASE / "templates" / "grid_search.html") as f:
        return f.read()


@app.get("/grid-search/demo/{code}", response_class=HTMLResponse)
def strategy_demo_page(code: str):
    """最优策略 PB_FED W=8yr 回测曲线可视化."""
    path = MERGED_DIR / f"{code}.parquet"
    if not path.exists():
        raise HTTPException(404, "指数不存在")
    with open(BASE / "templates" / "strategy_demo.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)


@app.get("/grid-search/buy-only", response_class=HTMLResponse)
def buy_only_page():
    with open(BASE / "templates" / "buy_only.html") as f:
        return f.read()


@app.get("/grid-search/sell-tune", response_class=HTMLResponse)
def sell_tune_page():
    with open(BASE / "templates" / "sell_tune.html") as f:
        return f.read()


@app.get("/grid-search/sell-tune/{code}", response_class=HTMLResponse)
def sell_tune_demo_page(code: str):
    path = MERGED_DIR / f"{code}.parquet"
    if not path.exists():
        raise HTTPException(404, "指数不存在")
    with open(BASE / "templates" / "sell_tune_demo.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)


# ── akshare 网格搜索 ──

@app.get("/aks/grid-search", response_class=HTMLResponse)
def aks_grid_search_page():
    with open(BASE / "templates" / "aks_grid_search.html") as f:
        return f.read()


@app.get("/api/aks/grid-search")
def api_aks_grid_search():
    import json as _json
    path = BASE / "grid_search_aks" / "output" / "latest.json"
    if not path.exists():
        return {"error": "未生成网格搜索结果, 请先运行 grid_search_aks/grid_search.py"}
    with open(path) as f:
        return _json.load(f)


@app.get("/aks/grid-search/{code}", response_class=HTMLResponse)
def aks_detail_page(code: str):
    """akshare 回测详情页面（曲线图+交易明细）"""
    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")
    with open(BASE / "templates" / "aks_detail.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)


@app.get("/api/aks/grid-search/{code}")
def api_aks_detail(code: str, w: int = 5,
                   bf: float = 0.10, bl: float = 0.20, bm: float = 0.35, bh: float = 0.65,
                   sh: float = 0.75, se: float = 0.85,
                   fed: str = "0.0", pv: str = "0.6", ps: str = "None"):
    """运行 akshare 回测，返回每日指标 + 交易明细"""
    import sys as _sys
    _sys.path.insert(0, str(BASE / "grid_search_aks"))
    from grid_search import run_backtest, calc_xirr as _xirr

    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")

    df = pd.read_parquet(merged_path)
    df["date"] = pd.to_datetime(df["date"])

    fed_val = float(fed) if fed != "None" else None
    pv_val = float(pv) if pv != "None" else None
    ps_val = float(ps) if ps != "None" else None

    params = {
        "buy_floor": bf, "buy_low": bl, "buy_mid": bm, "buy_high": bh,
        "sell_heavy": sh, "sell_extreme": se,
        "fed_gate": fed_val, "pb_veto": pv_val, "pb_sell": ps_val,
    }

    result = run_backtest(df, params, int(w))
    flows = result.get("cash_flows", [])
    if not flows:
        return {"error": "无有效交易数据", "daily": [], "trades": [], "meta": {}}

    pe_pct_col = f"pe_pct_w{w}"
    pb_pct_col = f"pb_pct_w{w}"
    fed_pct_col = f"fed_pct_w{w}"

    # 构建每日指标 (XIRR 只在每月第一日计算一次，避免性能问题)
    daily = []
    cf = []
    last_shares = 0.0; last_cum = 0.0; total_cash = 0.0; flow_idx = 0
    price_vals = df["price"].values
    date_vals = df["date"].values
    last_xirr = 0.0; last_xirr_month = -1

    for i in range(len(df)):
        cur_date = str(date_vals[i])[:10]
        cur_price = price_vals[i] if not pd.isna(price_vals[i]) else None
        buy_amt = 0.0; sell_amt = 0.0
        parts = cur_date.split("-") if cur_date else []
        cur_month = int(parts[0])*12 + int(parts[1]) if len(parts) >= 2 else -1

        while flow_idx < len(flows):
            fd, fa, famt = flows[flow_idx][0], flows[flow_idx][1], flows[flow_idx][2]
            if fd <= cur_date:
                if fa == "buy":
                    cf.append((fd, -float(famt))); last_cum += float(famt); buy_amt += float(famt)
                elif fa in ("sell", "clear"):
                    cf.append((fd, -float(famt))); sell_amt += abs(float(famt)); total_cash += abs(float(famt))
                last_shares = float(flows[flow_idx][3]); flow_idx += 1
            else:
                break

        if cur_price is None:
            daily.append({"date": cur_date, "cum_invested": 0, "equity": 0, "net_principal": 0,
                          "total_value": 0, "return_pct": 0, "xirr": 0.0,
                          "buy_amount": 0, "sell_amount": 0})
            continue

        eq = last_shares * cur_price if last_shares > 0 else 0
        total_value = eq + total_cash
        net_principal = max(last_cum - total_cash, 0)
        ret = (total_value - last_cum) / last_cum * 100 if last_cum > 0 else 0

        # 仅在月份变化时计算一次 XIRR
        dx = last_xirr
        if cur_month != last_xirr_month and len(cf) >= 3 and eq > 0:
            dx = _xirr(cf, cur_date, eq)
            last_xirr = dx
            last_xirr_month = cur_month

        # 读取当日百分位值（如果有）
        pe_pct = float(df.iloc[i][pe_pct_col]) if pe_pct_col in df.columns and not pd.isna(df.iloc[i].get(pe_pct_col)) else None
        pb_pct = float(df.iloc[i][pb_pct_col]) if pb_pct_col in df.columns and not pd.isna(df.iloc[i].get(pb_pct_col)) else None
        fed_pct_v = float(df.iloc[i][fed_pct_col]) if fed_pct_col in df.columns and not pd.isna(df.iloc[i].get(fed_pct_col)) else None
        pe_raw = float(df.iloc[i]["pe"]) if "pe" in df.columns and not pd.isna(df.iloc[i].get("pe")) else None
        pb_raw = float(df.iloc[i]["pb"]) if "pb" in df.columns and not pd.isna(df.iloc[i].get("pb")) else None

        daily.append({
            "date": cur_date,
            "price": round(cur_price, 2),
            "cum_invested": round(last_cum, 0),
            "equity": round(eq, 0),
            "net_principal": round(net_principal, 0),
            "total_value": round(total_value, 0),
            "return_pct": round(ret, 2),
            "xirr": round(dx, 4),
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
            "pe_raw": round(pe_raw, 2) if pe_raw else None,
            "pb_raw": round(pb_raw, 4) if pb_raw else None,
            "pe_pct": round(pe_pct, 4) if pe_pct else None,
            "pb_pct": round(pb_pct, 4) if pb_pct else None,
            "fed_pct": round(fed_pct_v, 4) if fed_pct_v else None,
        })

    # 交易明细
    trades_detail = []
    for t in flows:
        d, act, amt, sh, pr, pct, pbv, inv = t[0], t[1], float(t[2]), float(t[3]), float(t[4]), t[5], t[6], t[7]
        pct_str = f"PE%={(pct*100):.1f}%"
        if act == "buy":
            if pct < bf: reason = f"{pct_str}<{bf*100:.0f}%→3x"
            elif pct < bl: reason = f"{pct_str}<{bl*100:.0f}%→2x"
            elif pct < bm: reason = f"{pct_str}<{bm*100:.0f}%→1x"
            elif pct < bh: reason = f"{pct_str}<{bh*100:.0f}%→0.5x"
            else: reason = ""
        elif act == "sell":
            reason = f"{pct_str}≥{sh*100:.0f}%→卖20%"
        elif act == "clear":
            reason = f"{pct_str}≥{se*100:.0f}%→极端卖出"
        else:
            reason = ""
        trades_detail.append({
            "date": str(d)[:10], "action": act,
            "amount": round(amt, 0), "shares": round(sh, 4),
            "price": round(pr, 2), "pe_pct": round(float(pct), 4),
            "pb_pct": round(float(pbv), 4) if pbv is not None else None,
            "cum_invested": round(float(inv), 0), "reason": reason,
        })

    meta = {
        "code": code, "window_years": w,
        "params": params,
        "total_invested": result["total_invested"],
        "final_value": result["final_value"],
        "xirr": result["xirr"],
        "final_return": result["final_return"],
        "trades": result["trades"],
        "buys": result["buys"],
        "sells": result["sells"],
    }

    return {"meta": meta, "daily": daily, "trades": trades_detail}


# ── akshare 网格搜索（净值本金30W限制） ──

@app.get("/aks-capped/grid-search", response_class=HTMLResponse)
def aks_capped_grid_page():
    with open(BASE / "templates" / "aks_capped_grid.html") as f:
        return f.read()


@app.get("/api/aks-capped/grid-search")
def api_aks_capped_grid():
    import json as _json
    path = BASE / "grid_search_aks" / "output" / "latest_capped.json"
    if not path.exists():
        return {"error": "未生成净值本金限制版结果，请先运行 grid_search_aks/grid_search.py --capped"}
    with open(path) as f:
        return _json.load(f)


@app.get("/aks-capped/grid-search/{code}", response_class=HTMLResponse)
def aks_capped_detail_page(code: str):
    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")
    with open(BASE / "templates" / "aks_capped_detail.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)


@app.get("/api/aks-capped/grid-search/{code}")
def api_aks_capped_detail(code: str, w: int = 10,
                          bf: float = 0.10, bl: float = 0.20, bm: float = 0.35, bh: float = 0.65,
                          sh: float = 0.75, se: float = 0.85,
                          fed: str = "0.0", pv: str = "0.6", ps: str = "None"):
    import sys as _sys
    _sys.path.insert(0, str(BASE / "grid_search_aks"))
    from grid_search import run_backtest, calc_xirr as _xirr

    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")

    df = pd.read_parquet(merged_path)
    df["date"] = pd.to_datetime(df["date"])

    fed_val = float(fed) if fed != "None" else None
    pv_val = float(pv) if pv != "None" else None
    ps_val = float(ps) if ps != "None" else None

    params = {
        "buy_floor": bf, "buy_low": bl, "buy_mid": bm, "buy_high": bh,
        "sell_heavy": sh, "sell_extreme": se,
        "fed_gate": fed_val, "pb_veto": pv_val, "pb_sell": ps_val,
    }

    result = run_backtest(df, params, int(w), base_amount=500, max_net_principal=300_000)
    flows = result.get("cash_flows", [])
    if not flows:
        return {"error": "无有效交易数据", "daily": [], "trades": [], "meta": {}}

    pe_pct_col = f"pe_pct_w{w}"
    pb_pct_col = f"pb_pct_w{w}"
    fed_pct_col = f"fed_pct_w{w}"

    daily = []
    cf = []
    last_shares = 0.0; last_cum = 0.0; total_cash = 0.0; flow_idx = 0
    price_vals = df["price"].values
    date_vals = df["date"].values
    last_xirr = 0.0; last_xirr_month = -1

    for i in range(len(df)):
        cur_date = str(date_vals[i])[:10]
        cur_price = price_vals[i] if not pd.isna(price_vals[i]) else None
        buy_amt = 0.0; sell_amt = 0.0
        parts = cur_date.split("-") if cur_date else []
        cur_month = int(parts[0])*12 + int(parts[1]) if len(parts) >= 2 else -1

        while flow_idx < len(flows):
            fd, fa, famt = flows[flow_idx][0], flows[flow_idx][1], flows[flow_idx][2]
            if fd <= cur_date:
                if fa == "buy":
                    cf.append((fd, -float(famt))); last_cum += float(famt); buy_amt += float(famt)
                elif fa in ("sell", "clear"):
                    cf.append((fd, -float(famt))); sell_amt += abs(float(famt)); total_cash += abs(float(famt))
                last_shares = float(flows[flow_idx][3]); flow_idx += 1
            else:
                break

        if cur_price is None:
            daily.append({"date": cur_date, "cum_invested": 0, "equity": 0, "net_principal": 0,
                          "total_value": 0, "return_pct": 0, "xirr": 0.0,
                          "buy_amount": 0, "sell_amount": 0})
            continue

        eq = last_shares * cur_price if last_shares > 0 else 0
        total_value = eq + total_cash
        net_principal = max(last_cum - total_cash, 0)
        ret = (total_value - last_cum) / last_cum * 100 if last_cum > 0 else 0

        dx = last_xirr
        if cur_month != last_xirr_month and len(cf) >= 3 and eq > 0:
            dx = _xirr(cf, cur_date, eq)
            last_xirr = dx
            last_xirr_month = cur_month

        pe_pct = float(df.iloc[i][pe_pct_col]) if pe_pct_col in df.columns and not pd.isna(df.iloc[i].get(pe_pct_col)) else None
        pb_pct = float(df.iloc[i][pb_pct_col]) if pb_pct_col in df.columns and not pd.isna(df.iloc[i].get(pb_pct_col)) else None
        fed_pct_v = float(df.iloc[i][fed_pct_col]) if fed_pct_col in df.columns and not pd.isna(df.iloc[i].get(fed_pct_col)) else None
        pe_raw = float(df.iloc[i]["pe"]) if "pe" in df.columns and not pd.isna(df.iloc[i].get("pe")) else None
        pb_raw = float(df.iloc[i]["pb"]) if "pb" in df.columns and not pd.isna(df.iloc[i].get("pb")) else None

        daily.append({
            "date": cur_date,
            "price": round(cur_price, 2),
            "cum_invested": round(last_cum, 0),
            "equity": round(eq, 0),
            "net_principal": round(net_principal, 0),
            "total_value": round(total_value, 0),
            "return_pct": round(ret, 2),
            "xirr": round(dx, 4),
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
            "pe_raw": round(pe_raw, 2) if pe_raw else None,
            "pb_raw": round(pb_raw, 4) if pb_raw else None,
            "pe_pct": round(pe_pct, 4) if pe_pct else None,
            "pb_pct": round(pb_pct, 4) if pb_pct else None,
            "fed_pct": round(fed_pct_v, 4) if fed_pct_v else None,
        })

    trades_detail = []
    for t in flows:
        d, act, amt, sh, pr, pct, pbv, inv = t[0], t[1], float(t[2]), float(t[3]), float(t[4]), t[5], t[6], t[7]
        pct_str = f"PE%={(pct*100):.1f}%"
        if act == "buy":
            if pct < bf: reason = f"{pct_str}<{bf*100:.0f}%→3x"
            elif pct < bl: reason = f"{pct_str}<{bl*100:.0f}%→2x"
            elif pct < bm: reason = f"{pct_str}<{bm*100:.0f}%→1x"
            elif pct < bh: reason = f"{pct_str}<{bh*100:.0f}%→0.5x"
            else: reason = ""
        elif act == "sell":
            reason = f"{pct_str}≥{sh*100:.0f}%→卖20%"
        elif act == "clear":
            reason = f"{pct_str}≥{se*100:.0f}%→极端卖出"
        else:
            reason = ""
        trades_detail.append({
            "date": str(d)[:10], "action": act,
            "amount": round(amt, 0), "shares": round(sh, 4),
            "price": round(pr, 2), "pe_pct": round(float(pct), 4),
            "pb_pct": round(float(pbv), 4) if pbv is not None else None,
            "cum_invested": round(float(inv), 0), "reason": reason,
        })

    meta = {
        "code": code, "window_years": w,
        "params": params,
        "total_invested": result["total_invested"],
        "total_cash_in": result["total_cash_in"],
        "net_principal": result["net_principal"],
        "final_value": result["final_value"],
        "xirr": result["xirr"],
        "final_return": result["final_return"],
        "trades": result["trades"],
        "buys": result["buys"],
        "sells": result["sells"],
    }

    return {"meta": meta, "daily": daily, "trades": trades_detail}


# ── akshare 网格搜索（净值本金30W + ¥750/笔） ──

@app.get("/aks-capped750/grid-search", response_class=HTMLResponse)
def aks_capped750_grid_page():
    with open(BASE / "templates" / "aks_capped750_grid.html") as f:
        return f.read()


@app.get("/api/aks-capped750/grid-search")
def api_aks_capped750_grid():
    import json as _json
    path = BASE / "grid_search_aks" / "output" / "latest_capped750.json"
    if not path.exists():
        return {"error": "未生成, 请先运行 grid_search_aks/grid_search.py --capped --capped-base 750"}
    with open(path) as f:
        return _json.load(f)


@app.get("/aks-capped750/grid-search/{code}", response_class=HTMLResponse)
def aks_capped750_detail_page(code: str):
    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")
    with open(BASE / "templates" / "aks_capped750_detail.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)


@app.get("/api/aks-capped750/grid-search/{code}")
def api_aks_capped750_detail(code: str, w: int = 10,
                              bf: float = 0.15, bl: float = 0.20, bm: float = 0.35, bh: float = 0.55,
                              sh: float = 0.80, se: float = 0.85,
                              fed: str = "None", pv: str = "None", ps: str = "None"):
    import sys as _sys
    _sys.path.insert(0, str(BASE / "grid_search_aks"))
    from grid_search import run_backtest, calc_xirr as _xirr

    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")

    df = pd.read_parquet(merged_path)
    df["date"] = pd.to_datetime(df["date"])

    fed_val = float(fed) if fed != "None" else None
    pv_val = float(pv) if pv != "None" else None
    ps_val = float(ps) if ps != "None" else None

    params = {
        "buy_floor": bf, "buy_low": bl, "buy_mid": bm, "buy_high": bh,
        "sell_heavy": sh, "sell_extreme": se,
        "fed_gate": fed_val, "pb_veto": pv_val, "pb_sell": ps_val,
    }

    result = run_backtest(df, params, int(w), base_amount=750, max_net_principal=300_000)
    flows = result.get("cash_flows", [])
    if not flows:
        return {"error": "无有效交易数据", "daily": [], "trades": [], "meta": {}}

    pe_pct_col = f"pe_pct_w{w}"
    pb_pct_col = f"pb_pct_w{w}"
    fed_pct_col = f"fed_pct_w{w}"

    daily = []
    cf = []
    last_shares = 0.0; last_cum = 0.0; total_cash = 0.0; flow_idx = 0
    price_vals = df["price"].values
    date_vals = df["date"].values
    last_xirr = 0.0; last_xirr_month = -1

    for i in range(len(df)):
        cur_date = str(date_vals[i])[:10]
        cur_price = price_vals[i] if not pd.isna(price_vals[i]) else None
        buy_amt = 0.0; sell_amt = 0.0
        parts = cur_date.split("-") if cur_date else []
        cur_month = int(parts[0])*12 + int(parts[1]) if len(parts) >= 2 else -1

        while flow_idx < len(flows):
            fd, fa, famt = flows[flow_idx][0], flows[flow_idx][1], flows[flow_idx][2]
            if fd <= cur_date:
                if fa == "buy":
                    cf.append((fd, -float(famt))); last_cum += float(famt); buy_amt += float(famt)
                elif fa in ("sell", "clear"):
                    cf.append((fd, -float(famt))); sell_amt += abs(float(famt)); total_cash += abs(float(famt))
                last_shares = float(flows[flow_idx][3]); flow_idx += 1
            else:
                break

        if cur_price is None:
            daily.append({"date": cur_date, "cum_invested": 0, "equity": 0, "net_principal": 0,
                          "total_value": 0, "return_pct": 0, "xirr": 0.0,
                          "buy_amount": 0, "sell_amount": 0})
            continue

        eq = last_shares * cur_price if last_shares > 0 else 0
        total_value = eq + total_cash
        net_principal = max(last_cum - total_cash, 0)
        ret = (total_value - last_cum) / last_cum * 100 if last_cum > 0 else 0

        dx = last_xirr
        if cur_month != last_xirr_month and len(cf) >= 3 and eq > 0:
            dx = _xirr(cf, cur_date, eq)
            last_xirr = dx
            last_xirr_month = cur_month

        pe_pct = float(df.iloc[i][pe_pct_col]) if pe_pct_col in df.columns and not pd.isna(df.iloc[i].get(pe_pct_col)) else None
        pb_pct = float(df.iloc[i][pb_pct_col]) if pb_pct_col in df.columns and not pd.isna(df.iloc[i].get(pb_pct_col)) else None
        fed_pct_v = float(df.iloc[i][fed_pct_col]) if fed_pct_col in df.columns and not pd.isna(df.iloc[i].get(fed_pct_col)) else None
        pe_raw = float(df.iloc[i]["pe"]) if "pe" in df.columns and not pd.isna(df.iloc[i].get("pe")) else None
        pb_raw = float(df.iloc[i]["pb"]) if "pb" in df.columns and not pd.isna(df.iloc[i].get("pb")) else None

        daily.append({
            "date": cur_date,
            "price": round(cur_price, 2),
            "cum_invested": round(last_cum, 0),
            "equity": round(eq, 0),
            "net_principal": round(net_principal, 0),
            "total_value": round(total_value, 0),
            "return_pct": round(ret, 2),
            "xirr": round(dx, 4),
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
            "pe_raw": round(pe_raw, 2) if pe_raw else None,
            "pb_raw": round(pb_raw, 4) if pb_raw else None,
            "pe_pct": round(pe_pct, 4) if pe_pct else None,
            "pb_pct": round(pb_pct, 4) if pb_pct else None,
            "fed_pct": round(fed_pct_v, 4) if fed_pct_v else None,
        })

    trades_detail = []
    for t in flows:
        d, act, amt, sh, pr, pct, pbv, inv = t[0], t[1], float(t[2]), float(t[3]), float(t[4]), t[5], t[6], t[7]
        pct_str = f"PE%={(pct*100):.1f}%"
        if act == "buy":
            if pct < bf: reason = f"{pct_str}<{bf*100:.0f}%→3x"
            elif pct < bl: reason = f"{pct_str}<{bl*100:.0f}%→2x"
            elif pct < bm: reason = f"{pct_str}<{bm*100:.0f}%→1x"
            elif pct < bh: reason = f"{pct_str}<{bh*100:.0f}%→0.5x"
            else: reason = ""
        elif act == "sell":
            reason = f"{pct_str}≥{sh*100:.0f}%→卖20%"
        elif act == "clear":
            reason = f"{pct_str}≥{se*100:.0f}%→极端卖出"
        else:
            reason = ""
        trades_detail.append({
            "date": str(d)[:10], "action": act,
            "amount": round(amt, 0), "shares": round(sh, 4),
            "price": round(pr, 2), "pe_pct": round(float(pct), 4),
            "pb_pct": round(float(pbv), 4) if pbv is not None else None,
            "cum_invested": round(float(inv), 0), "reason": reason,
        })

    meta = {
        "code": code, "window_years": w,
        "params": params,
        "total_invested": result["total_invested"],
        "total_cash_in": result["total_cash_in"],
        "net_principal": result["net_principal"],
        "final_value": result["final_value"],
        "xirr": result["xirr"],
        "final_return": result["final_return"],
        "trades": result["trades"],
        "buys": result["buys"],
        "sells": result["sells"],
    }

    return {"meta": meta, "daily": daily, "trades": trades_detail}


# ── akshare 网格搜索（严格买入+闲置收益） ──

@app.get("/aks-strict/grid-search", response_class=HTMLResponse)
def aks_strict_grid_page():
    with open(BASE / "templates" / "aks_strict_grid.html") as f:
        return f.read()


@app.get("/api/aks-strict/grid-search")
def api_aks_strict_grid():
    import json as _json
    path = BASE / "grid_search_aks" / "output" / "latest_strict.json"
    if not path.exists():
        return {"error": "未生成, 请先运行 grid_search_aks/grid_search.py --strict"}
    with open(path) as f:
        return _json.load(f)


@app.get("/aks-strict/grid-search/{code}", response_class=HTMLResponse)
def aks_strict_detail_page(code: str):
    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")
    with open(BASE / "templates" / "aks_strict_detail.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)


@app.get("/api/aks-strict/grid-search/{code}")
def api_aks_strict_detail(code: str, w: int = 10,
                          bf: float = 0.05, bl: float = 0.12, bm: float = 0.22, bh: float = 0.50,
                          sh: float = 0.80, se: float = 0.85,
                          fed: str = "None", pv: str = "None", ps: str = "None"):
    import sys as _sys
    _sys.path.insert(0, str(BASE / "grid_search_aks"))
    from grid_search import run_backtest, calc_xirr as _xirr

    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")

    df = pd.read_parquet(merged_path)
    df["date"] = pd.to_datetime(df["date"])

    fed_val = float(fed) if fed != "None" else None
    pv_val = float(pv) if pv != "None" else None
    ps_val = float(ps) if ps != "None" else None

    params = {
        "buy_floor": bf, "buy_low": bl, "buy_mid": bm, "buy_high": bh,
        "sell_heavy": sh, "sell_extreme": se,
        "fed_gate": fed_val, "pb_veto": pv_val, "pb_sell": ps_val,
    }

    result = run_backtest(df, params, int(w), base_amount=500,
                          idle_cash_rate=0.02, min_trades=10)
    flows = result.get("cash_flows", [])
    if not flows:
        return {"error": "无有效交易数据", "daily": [], "trades": [], "meta": {}}

    pe_pct_col = f"pe_pct_w{w}"
    pb_pct_col = f"pb_pct_w{w}"
    fed_pct_col = f"fed_pct_w{w}"

    daily = []
    cf = []
    last_shares = 0.0; last_cum = 0.0; total_cash = 0.0; flow_idx = 0
    price_vals = df["price"].values
    date_vals = df["date"].values
    last_xirr = 0.0; last_xirr_month = -1

    for i in range(len(df)):
        cur_date = str(date_vals[i])[:10]
        cur_price = price_vals[i] if not pd.isna(price_vals[i]) else None
        buy_amt = 0.0; sell_amt = 0.0
        parts = cur_date.split("-") if cur_date else []
        cur_month = int(parts[0])*12 + int(parts[1]) if len(parts) >= 2 else -1

        while flow_idx < len(flows):
            fd, fa, famt = flows[flow_idx][0], flows[flow_idx][1], flows[flow_idx][2]
            if fd <= cur_date:
                if fa == "buy":
                    cf.append((fd, -float(famt))); last_cum += float(famt); buy_amt += float(famt)
                elif fa in ("sell", "clear"):
                    cf.append((fd, -float(famt))); sell_amt += abs(float(famt)); total_cash += abs(float(famt))
                last_shares = float(flows[flow_idx][3]); flow_idx += 1
            else:
                break

        if cur_price is None:
            daily.append({"date": cur_date, "cum_invested": 0, "equity": 0, "net_principal": 0,
                          "total_value": 0, "return_pct": 0, "xirr": 0.0,
                          "buy_amount": 0, "sell_amount": 0})
            continue

        eq = last_shares * cur_price if last_shares > 0 else 0
        total_value = eq + total_cash
        net_principal = max(last_cum - total_cash, 0)
        ret = (total_value - last_cum) / last_cum * 100 if last_cum > 0 else 0

        dx = last_xirr
        if cur_month != last_xirr_month and len(cf) >= 3 and eq > 0:
            dx = _xirr(cf, cur_date, eq)
            last_xirr = dx
            last_xirr_month = cur_month

        pe_pct = float(df.iloc[i][pe_pct_col]) if pe_pct_col in df.columns and not pd.isna(df.iloc[i].get(pe_pct_col)) else None
        pb_pct = float(df.iloc[i][pb_pct_col]) if pb_pct_col in df.columns and not pd.isna(df.iloc[i].get(pb_pct_col)) else None
        fed_pct_v = float(df.iloc[i][fed_pct_col]) if fed_pct_col in df.columns and not pd.isna(df.iloc[i].get(fed_pct_col)) else None
        pe_raw = float(df.iloc[i]["pe"]) if "pe" in df.columns and not pd.isna(df.iloc[i].get("pe")) else None
        pb_raw = float(df.iloc[i]["pb"]) if "pb" in df.columns and not pd.isna(df.iloc[i].get("pb")) else None

        daily.append({
            "date": cur_date,
            "price": round(cur_price, 2),
            "cum_invested": round(last_cum, 0),
            "equity": round(eq, 0),
            "net_principal": round(net_principal, 0),
            "total_value": round(total_value, 0),
            "return_pct": round(ret, 2),
            "xirr": round(dx, 4),
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
            "pe_raw": round(pe_raw, 2) if pe_raw else None,
            "pb_raw": round(pb_raw, 4) if pb_raw else None,
            "pe_pct": round(pe_pct, 4) if pe_pct else None,
            "pb_pct": round(pb_pct, 4) if pb_pct else None,
            "fed_pct": round(fed_pct_v, 4) if fed_pct_v else None,
        })

    trades_detail = []
    for t in flows:
        d, act, amt, sh, pr, pct, pbv, inv = t[0], t[1], float(t[2]), float(t[3]), float(t[4]), t[5], t[6], t[7]
        pct_str = f"PE%={(pct*100):.1f}%"
        if act == "buy":
            if pct < bf: reason = f"{pct_str}<{bf*100:.0f}%→3x"
            elif pct < bl: reason = f"{pct_str}<{bl*100:.0f}%→2x"
            elif pct < bm: reason = f"{pct_str}<{bm*100:.0f}%→1x"
            elif pct < bh: reason = f"{pct_str}<{bh*100:.0f}%→0.5x"
            else: reason = ""
        elif act == "sell":
            reason = f"{pct_str}≥{sh*100:.0f}%→卖20%"
        elif act == "clear":
            reason = f"{pct_str}≥{se*100:.0f}%→极端卖出"
        else:
            reason = ""
        trades_detail.append({
            "date": str(d)[:10], "action": act,
            "amount": round(amt, 0), "shares": round(sh, 4),
            "price": round(pr, 2), "pe_pct": round(float(pct), 4),
            "pb_pct": round(float(pbv), 4) if pbv is not None else None,
            "cum_invested": round(float(inv), 0), "reason": reason,
        })

    meta = {
        "code": code, "window_years": w,
        "params": params,
        "total_invested": result["total_invested"],
        "total_cash_in": result["total_cash_in"],
        "net_principal": result["net_principal"],
        "final_value": result["final_value"],
        "position_value": result.get("position_value", 0),
        "idle_cash": result.get("idle_cash", 0),
        "interest_earned": result.get("interest_earned", 0),
        "max_equity": max((d.get("equity", 0) for d in daily), default=0),
        "xirr": result["xirr"],
        "final_return": result["final_return"],
        "trades": result["trades"],
        "buys": result["buys"],
        "sells": result["sells"],
    }

    return {"meta": meta, "daily": daily, "trades": trades_detail}


# ── akshare 严格买入 ¥750/笔 ──

@app.get("/aks-strict750/grid-search", response_class=HTMLResponse)
def aks_strict750_grid_page():
    with open(BASE / "templates" / "aks_strict750_grid.html") as f:
        return f.read()

@app.get("/api/aks-strict750/grid-search")
def api_aks_strict750_grid():
    import json as _json
    path = BASE / "grid_search_aks" / "output" / "latest_strict750.json"
    if not path.exists():
        return {"error": "未生成, 请先运行 grid_search_aks/grid_search.py --strict --strict-base 750"}
    with open(path) as f:
        return _json.load(f)

@app.get("/aks-strict750/grid-search/{code}", response_class=HTMLResponse)
def aks_strict750_detail_page(code: str):
    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")
    with open(BASE / "templates" / "aks_strict750_detail.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)

@app.get("/api/aks-strict750/grid-search/{code}")
def api_aks_strict750_detail(code: str, w: int = 10,
                              bf: float = 0.08, bl: float = 0.12, bm: float = 0.22, bh: float = 0.40,
                              sh: float = 0.75, se: float = 0.85,
                              fed: str = "None", pv: str = "None", ps: str = "None"):
    import sys as _sys
    _sys.path.insert(0, str(BASE / "grid_search_aks"))
    from grid_search import run_backtest, calc_xirr as _xirr
    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")
    df = pd.read_parquet(merged_path)
    df["date"] = pd.to_datetime(df["date"])
    fed_val = float(fed) if fed != "None" else None
    pv_val = float(pv) if pv != "None" else None
    ps_val = float(ps) if ps != "None" else None
    params = {"buy_floor": bf, "buy_low": bl, "buy_mid": bm, "buy_high": bh,
              "sell_heavy": sh, "sell_extreme": se,
              "fed_gate": fed_val, "pb_veto": pv_val, "pb_sell": ps_val}
    result = run_backtest(df, params, int(w), base_amount=750, idle_cash_rate=0.02, min_trades=10)
    flows = result.get("cash_flows", [])
    if not flows:
        return {"error": "无有效交易数据", "daily": [], "trades": [], "meta": {}}
    pe_pct_col = f"pe_pct_w{w}"
    pb_pct_col = f"pb_pct_w{w}"
    fed_pct_col = f"fed_pct_w{w}"
    daily = []; cf = []
    last_shares = 0.0; last_cum = 0.0; total_cash = 0.0; flow_idx = 0
    price_vals = df["price"].values; date_vals = df["date"].values
    last_xirr = 0.0; last_xirr_month = -1
    for i in range(len(df)):
        cur_date = str(date_vals[i])[:10]
        cur_price = price_vals[i] if not pd.isna(price_vals[i]) else None
        buy_amt = 0.0; sell_amt = 0.0
        parts = cur_date.split("-") if cur_date else []
        cur_month = int(parts[0])*12 + int(parts[1]) if len(parts) >= 2 else -1
        while flow_idx < len(flows):
            fd, fa, famt = flows[flow_idx][0], flows[flow_idx][1], flows[flow_idx][2]
            if fd <= cur_date:
                if fa == "buy":
                    cf.append((fd, -float(famt))); last_cum += float(famt); buy_amt += float(famt)
                elif fa in ("sell", "clear"):
                    cf.append((fd, -float(famt))); sell_amt += abs(float(famt)); total_cash += abs(float(famt))
                last_shares = float(flows[flow_idx][3]); flow_idx += 1
            else:
                break
        if cur_price is None:
            daily.append({"date": cur_date, "cum_invested": 0, "equity": 0, "net_principal": 0,
                          "total_value": 0, "return_pct": 0, "xirr": 0.0,
                          "buy_amount": 0, "sell_amount": 0})
            continue
        eq = last_shares * cur_price if last_shares > 0 else 0
        total_value = eq + total_cash
        net_principal = max(last_cum - total_cash, 0)
        ret = (total_value - last_cum) / last_cum * 100 if last_cum > 0 else 0
        dx = last_xirr
        if cur_month != last_xirr_month and len(cf) >= 3 and eq > 0:
            dx = _xirr(cf, cur_date, eq); last_xirr = dx; last_xirr_month = cur_month
        pe_pct = float(df.iloc[i][pe_pct_col]) if pe_pct_col in df.columns and not pd.isna(df.iloc[i].get(pe_pct_col)) else None
        pb_pct = float(df.iloc[i][pb_pct_col]) if pb_pct_col in df.columns and not pd.isna(df.iloc[i].get(pb_pct_col)) else None
        fed_pct_v = float(df.iloc[i][fed_pct_col]) if fed_pct_col in df.columns and not pd.isna(df.iloc[i].get(fed_pct_col)) else None
        pe_raw = float(df.iloc[i]["pe"]) if "pe" in df.columns and not pd.isna(df.iloc[i].get("pe")) else None
        pb_raw = float(df.iloc[i]["pb"]) if "pb" in df.columns and not pd.isna(df.iloc[i].get("pb")) else None
        daily.append({"date": cur_date, "price": round(cur_price, 2),
            "cum_invested": round(last_cum, 0), "equity": round(eq, 0),
            "net_principal": round(net_principal, 0), "total_value": round(total_value, 0),
            "return_pct": round(ret, 2), "xirr": round(dx, 4),
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
            "pe_raw": round(pe_raw, 2) if pe_raw else None,
            "pb_raw": round(pb_raw, 4) if pb_raw else None,
            "pe_pct": round(pe_pct, 4) if pe_pct else None,
            "pb_pct": round(pb_pct, 4) if pb_pct else None,
            "fed_pct": round(fed_pct_v, 4) if fed_pct_v else None})
    trades_detail = []
    for t in flows:
        d, act, amt, sh, pr, pct, pbv, inv = t[0], t[1], float(t[2]), float(t[3]), float(t[4]), t[5], t[6], t[7]
        pct_str = f"PE%={(pct*100):.1f}%"
        if act == "buy":
            if pct < bf: reason = f"{pct_str}<{bf*100:.0f}%→3x"
            elif pct < bl: reason = f"{pct_str}<{bl*100:.0f}%→2x"
            elif pct < bm: reason = f"{pct_str}<{bm*100:.0f}%→1x"
            elif pct < bh: reason = f"{pct_str}<{bh*100:.0f}%→0.5x"
            else: reason = ""
        elif act == "sell": reason = f"{pct_str}≥{sh*100:.0f}%→卖20%"
        elif act == "clear": reason = f"{pct_str}≥{se*100:.0f}%→极端卖出"
        else: reason = ""
        trades_detail.append({"date": str(d)[:10], "action": act,
            "amount": round(amt, 0), "shares": round(sh, 4), "price": round(pr, 2),
            "pe_pct": round(float(pct), 4),
            "pb_pct": round(float(pbv), 4) if pbv is not None else None,
            "cum_invested": round(float(inv), 0), "reason": reason})
    meta = {"code": code, "window_years": w, "params": params,
        "total_invested": result["total_invested"], "total_cash_in": result["total_cash_in"],
        "net_principal": result["net_principal"], "final_value": result["final_value"],
        "position_value": result.get("position_value", 0), "idle_cash": result.get("idle_cash", 0),
        "interest_earned": result.get("interest_earned", 0), "max_equity": max((d.get("equity", 0) for d in daily), default=0), "xirr": result["xirr"],
        "final_return": result["final_return"], "trades": result["trades"],
        "buys": result["buys"], "sells": result["sells"]}
    return {"meta": meta, "daily": daily, "trades": trades_detail}


# ── akshare 严格买入 ¥1000/笔 ──

@app.get("/aks-strict1000/grid-search", response_class=HTMLResponse)
def aks_strict1000_grid_page():
    with open(BASE / "templates" / "aks_strict1000_grid.html") as f:
        return f.read()

@app.get("/api/aks-strict1000/grid-search")
def api_aks_strict1000_grid():
    import json as _json
    path = BASE / "grid_search_aks" / "output" / "latest_strict1000.json"
    if not path.exists():
        return {"error": "未生成, 请先运行 grid_search_aks/grid_search.py --strict --strict-base 1000"}
    with open(path) as f:
        return _json.load(f)

@app.get("/aks-strict1000/grid-search/{code}", response_class=HTMLResponse)
def aks_strict1000_detail_page(code: str):
    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")
    with open(BASE / "templates" / "aks_strict1000_detail.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)

@app.get("/api/aks-strict1000/grid-search/{code}")
def api_aks_strict1000_detail(code: str, w: int = 10,
                               bf: float = 0.08, bl: float = 0.12, bm: float = 0.22, bh: float = 0.40,
                               sh: float = 0.75, se: float = 0.85,
                               fed: str = "None", pv: str = "None", ps: str = "None"):
    import sys as _sys
    _sys.path.insert(0, str(BASE / "grid_search_aks"))
    from grid_search import run_backtest, calc_xirr as _xirr
    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")
    df = pd.read_parquet(merged_path)
    df["date"] = pd.to_datetime(df["date"])
    fed_val = float(fed) if fed != "None" else None
    pv_val = float(pv) if pv != "None" else None
    ps_val = float(ps) if ps != "None" else None
    params = {"buy_floor": bf, "buy_low": bl, "buy_mid": bm, "buy_high": bh,
              "sell_heavy": sh, "sell_extreme": se,
              "fed_gate": fed_val, "pb_veto": pv_val, "pb_sell": ps_val}
    result = run_backtest(df, params, int(w), base_amount=1000, idle_cash_rate=0.02, min_trades=10)
    flows = result.get("cash_flows", [])
    if not flows:
        return {"error": "无有效交易数据", "daily": [], "trades": [], "meta": {}}
    pe_pct_col = f"pe_pct_w{w}"
    pb_pct_col = f"pb_pct_w{w}"
    fed_pct_col = f"fed_pct_w{w}"
    daily = []; cf = []
    last_shares = 0.0; last_cum = 0.0; total_cash = 0.0; flow_idx = 0
    price_vals = df["price"].values; date_vals = df["date"].values
    last_xirr = 0.0; last_xirr_month = -1
    for i in range(len(df)):
        cur_date = str(date_vals[i])[:10]
        cur_price = price_vals[i] if not pd.isna(price_vals[i]) else None
        buy_amt = 0.0; sell_amt = 0.0
        parts = cur_date.split("-") if cur_date else []
        cur_month = int(parts[0])*12 + int(parts[1]) if len(parts) >= 2 else -1
        while flow_idx < len(flows):
            fd, fa, famt = flows[flow_idx][0], flows[flow_idx][1], flows[flow_idx][2]
            if fd <= cur_date:
                if fa == "buy":
                    cf.append((fd, -float(famt))); last_cum += float(famt); buy_amt += float(famt)
                elif fa in ("sell", "clear"):
                    cf.append((fd, -float(famt))); sell_amt += abs(float(famt)); total_cash += abs(float(famt))
                last_shares = float(flows[flow_idx][3]); flow_idx += 1
            else:
                break
        if cur_price is None:
            daily.append({"date": cur_date, "cum_invested": 0, "equity": 0, "net_principal": 0,
                          "total_value": 0, "return_pct": 0, "xirr": 0.0,
                          "buy_amount": 0, "sell_amount": 0})
            continue
        eq = last_shares * cur_price if last_shares > 0 else 0
        total_value = eq + total_cash
        net_principal = max(last_cum - total_cash, 0)
        ret = (total_value - last_cum) / last_cum * 100 if last_cum > 0 else 0
        dx = last_xirr
        if cur_month != last_xirr_month and len(cf) >= 3 and eq > 0:
            dx = _xirr(cf, cur_date, eq); last_xirr = dx; last_xirr_month = cur_month
        pe_pct = float(df.iloc[i][pe_pct_col]) if pe_pct_col in df.columns and not pd.isna(df.iloc[i].get(pe_pct_col)) else None
        pb_pct = float(df.iloc[i][pb_pct_col]) if pb_pct_col in df.columns and not pd.isna(df.iloc[i].get(pb_pct_col)) else None
        fed_pct_v = float(df.iloc[i][fed_pct_col]) if fed_pct_col in df.columns and not pd.isna(df.iloc[i].get(fed_pct_col)) else None
        pe_raw = float(df.iloc[i]["pe"]) if "pe" in df.columns and not pd.isna(df.iloc[i].get("pe")) else None
        pb_raw = float(df.iloc[i]["pb"]) if "pb" in df.columns and not pd.isna(df.iloc[i].get("pb")) else None
        daily.append({"date": cur_date, "price": round(cur_price, 2),
            "cum_invested": round(last_cum, 0), "equity": round(eq, 0),
            "net_principal": round(net_principal, 0), "total_value": round(total_value, 0),
            "return_pct": round(ret, 2), "xirr": round(dx, 4),
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
            "pe_raw": round(pe_raw, 2) if pe_raw else None,
            "pb_raw": round(pb_raw, 4) if pb_raw else None,
            "pe_pct": round(pe_pct, 4) if pe_pct else None,
            "pb_pct": round(pb_pct, 4) if pb_pct else None,
            "fed_pct": round(fed_pct_v, 4) if fed_pct_v else None})
    trades_detail = []
    for t in flows:
        d, act, amt, sh, pr, pct, pbv, inv = t[0], t[1], float(t[2]), float(t[3]), float(t[4]), t[5], t[6], t[7]
        pct_str = f"PE%={(pct*100):.1f}%"
        if act == "buy":
            if pct < bf: reason = f"{pct_str}<{bf*100:.0f}%→3x"
            elif pct < bl: reason = f"{pct_str}<{bl*100:.0f}%→2x"
            elif pct < bm: reason = f"{pct_str}<{bm*100:.0f}%→1x"
            elif pct < bh: reason = f"{pct_str}<{bh*100:.0f}%→0.5x"
            else: reason = ""
        elif act == "sell": reason = f"{pct_str}≥{sh*100:.0f}%→卖20%"
        elif act == "clear": reason = f"{pct_str}≥{se*100:.0f}%→极端卖出"
        else: reason = ""
        trades_detail.append({"date": str(d)[:10], "action": act,
            "amount": round(amt, 0), "shares": round(sh, 4), "price": round(pr, 2),
            "pe_pct": round(float(pct), 4),
            "pb_pct": round(float(pbv), 4) if pbv is not None else None,
            "cum_invested": round(float(inv), 0), "reason": reason})
    meta = {"code": code, "window_years": w, "params": params,
        "total_invested": result["total_invested"], "total_cash_in": result["total_cash_in"],
        "net_principal": result["net_principal"], "final_value": result["final_value"],
        "position_value": result.get("position_value", 0), "idle_cash": result.get("idle_cash", 0),
        "interest_earned": result.get("interest_earned", 0), "max_equity": max((d.get("equity", 0) for d in daily), default=0), "xirr": result["xirr"],
        "final_return": result["final_return"], "trades": result["trades"],
        "buys": result["buys"], "sells": result["sells"]}
    return {"meta": meta, "daily": daily, "trades": trades_detail}


# ── akshare 严格买入 ¥1500/笔 ──

@app.get("/aks-strict1500/grid-search", response_class=HTMLResponse)
def aks_strict1500_grid_page():
    with open(BASE / "templates" / "aks_strict1500_grid.html") as f:
        return f.read()

@app.get("/api/aks-strict1500/grid-search")
def api_aks_strict1500_grid():
    import json as _json
    path = BASE / "grid_search_aks" / "output" / "latest_strict1500.json"
    if not path.exists():
        return {"error": "未生成, 请先运行 grid_search_aks/grid_search.py --strict --strict-base 1500"}
    with open(path) as f:
        return _json.load(f)

@app.get("/aks-strict1500/grid-search/{code}", response_class=HTMLResponse)
def aks_strict1500_detail_page(code: str):
    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")
    with open(BASE / "templates" / "aks_strict1500_detail.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)

@app.get("/api/aks-strict1500/grid-search/{code}")
def api_aks_strict1500_detail(code: str, w: int = 10,
                               bf: float = 0.08, bl: float = 0.12, bm: float = 0.22, bh: float = 0.40,
                               sh: float = 0.75, se: float = 0.85,
                               fed: str = "None", pv: str = "None", ps: str = "None"):
    import sys as _sys
    _sys.path.insert(0, str(BASE / "grid_search_aks"))
    from grid_search import run_backtest, calc_xirr as _xirr
    merged_path = BASE / "data-store" / "parquet" / "aks_merged" / f"{code}.parquet"
    if not merged_path.exists():
        raise HTTPException(404, "指数不存在")
    df = pd.read_parquet(merged_path)
    df["date"] = pd.to_datetime(df["date"])
    fed_val = float(fed) if fed != "None" else None
    pv_val = float(pv) if pv != "None" else None
    ps_val = float(ps) if ps != "None" else None
    params = {"buy_floor": bf, "buy_low": bl, "buy_mid": bm, "buy_high": bh,
              "sell_heavy": sh, "sell_extreme": se,
              "fed_gate": fed_val, "pb_veto": pv_val, "pb_sell": ps_val}
    result = run_backtest(df, params, int(w), base_amount=1500, idle_cash_rate=0.02, min_trades=10)
    flows = result.get("cash_flows", [])
    if not flows:
        return {"error": "无有效交易数据", "daily": [], "trades": [], "meta": {}}
    pe_pct_col = f"pe_pct_w{w}"
    pb_pct_col = f"pb_pct_w{w}"
    fed_pct_col = f"fed_pct_w{w}"
    daily = []; cf = []
    last_shares = 0.0; last_cum = 0.0; total_cash = 0.0; flow_idx = 0
    price_vals = df["price"].values; date_vals = df["date"].values
    last_xirr = 0.0; last_xirr_month = -1
    for i in range(len(df)):
        cur_date = str(date_vals[i])[:10]
        cur_price = price_vals[i] if not pd.isna(price_vals[i]) else None
        buy_amt = 0.0; sell_amt = 0.0
        parts = cur_date.split("-") if cur_date else []
        cur_month = int(parts[0])*12 + int(parts[1]) if len(parts) >= 2 else -1
        while flow_idx < len(flows):
            fd, fa, famt = flows[flow_idx][0], flows[flow_idx][1], flows[flow_idx][2]
            if fd <= cur_date:
                if fa == "buy":
                    cf.append((fd, -float(famt))); last_cum += float(famt); buy_amt += float(famt)
                elif fa in ("sell", "clear"):
                    cf.append((fd, -float(famt))); sell_amt += abs(float(famt)); total_cash += abs(float(famt))
                last_shares = float(flows[flow_idx][3]); flow_idx += 1
            else:
                break
        if cur_price is None:
            daily.append({"date": cur_date, "cum_invested": 0, "equity": 0, "net_principal": 0,
                          "total_value": 0, "return_pct": 0, "xirr": 0.0,
                          "buy_amount": 0, "sell_amount": 0})
            continue
        eq = last_shares * cur_price if last_shares > 0 else 0
        total_value = eq + total_cash
        net_principal = max(last_cum - total_cash, 0)
        ret = (total_value - last_cum) / last_cum * 100 if last_cum > 0 else 0
        dx = last_xirr
        if cur_month != last_xirr_month and len(cf) >= 3 and eq > 0:
            dx = _xirr(cf, cur_date, eq); last_xirr = dx; last_xirr_month = cur_month
        pe_pct = float(df.iloc[i][pe_pct_col]) if pe_pct_col in df.columns and not pd.isna(df.iloc[i].get(pe_pct_col)) else None
        pb_pct = float(df.iloc[i][pb_pct_col]) if pb_pct_col in df.columns and not pd.isna(df.iloc[i].get(pb_pct_col)) else None
        fed_pct_v = float(df.iloc[i][fed_pct_col]) if fed_pct_col in df.columns and not pd.isna(df.iloc[i].get(fed_pct_col)) else None
        pe_raw = float(df.iloc[i]["pe"]) if "pe" in df.columns and not pd.isna(df.iloc[i].get("pe")) else None
        pb_raw = float(df.iloc[i]["pb"]) if "pb" in df.columns and not pd.isna(df.iloc[i].get("pb")) else None
        daily.append({"date": cur_date, "price": round(cur_price, 2),
            "cum_invested": round(last_cum, 0), "equity": round(eq, 0),
            "net_principal": round(net_principal, 0), "total_value": round(total_value, 0),
            "return_pct": round(ret, 2), "xirr": round(dx, 4),
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
            "pe_raw": round(pe_raw, 2) if pe_raw else None,
            "pb_raw": round(pb_raw, 4) if pb_raw else None,
            "pe_pct": round(pe_pct, 4) if pe_pct else None,
            "pb_pct": round(pb_pct, 4) if pb_pct else None,
            "fed_pct": round(fed_pct_v, 4) if fed_pct_v else None})
    trades_detail = []
    for t in flows:
        d, act, amt, sh, pr, pct, pbv, inv = t[0], t[1], float(t[2]), float(t[3]), float(t[4]), t[5], t[6], t[7]
        pct_str = f"PE%={(pct*100):.1f}%"
        if act == "buy":
            if pct < bf: reason = f"{pct_str}<{bf*100:.0f}%→3x"
            elif pct < bl: reason = f"{pct_str}<{bl*100:.0f}%→2x"
            elif pct < bm: reason = f"{pct_str}<{bm*100:.0f}%→1x"
            elif pct < bh: reason = f"{pct_str}<{bh*100:.0f}%→0.5x"
            else: reason = ""
        elif act == "sell": reason = f"{pct_str}≥{sh*100:.0f}%→卖20%"
        elif act == "clear": reason = f"{pct_str}≥{se*100:.0f}%→极端卖出"
        else: reason = ""
        trades_detail.append({"date": str(d)[:10], "action": act,
            "amount": round(amt, 0), "shares": round(sh, 4), "price": round(pr, 2),
            "pe_pct": round(float(pct), 4),
            "pb_pct": round(float(pbv), 4) if pbv is not None else None,
            "cum_invested": round(float(inv), 0), "reason": reason})
    meta = {"code": code, "window_years": w, "params": params,
        "total_invested": result["total_invested"], "total_cash_in": result["total_cash_in"],
        "net_principal": result["net_principal"], "final_value": result["final_value"],
        "position_value": result.get("position_value", 0), "idle_cash": result.get("idle_cash", 0),
        "interest_earned": result.get("interest_earned", 0), "max_equity": max((d.get("equity", 0) for d in daily), default=0), "xirr": result["xirr"],
        "final_return": result["final_return"], "trades": result["trades"],
        "buys": result["buys"], "sells": result["sells"]}
    return {"meta": meta, "daily": daily, "trades": trades_detail}


# ── 滚动窗口鲁棒性分析 ──

@app.get("/aks-rolling/grid-search", response_class=HTMLResponse)
def aks_rolling_grid_page():
    with open(BASE / "templates" / "aks_rolling_grid.html") as f:
        return f.read()


@app.get("/api/aks-rolling/grid-search")
def api_aks_rolling_grid():
    import json as _json
    path = BASE / "grid_search_aks" / "output" / "latest_rolling.json"
    if not path.exists():
        return {"error": "未生成, 请先运行 grid_search_aks/rolling_window.py"}
    with open(path) as f:
        return _json.load(f)


@app.get("/aks-rolling/grid-search/{code}", response_class=HTMLResponse)
def aks_rolling_detail_page(code: str):
    with open(BASE / "templates" / "aks_rolling_detail.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)


@app.get("/api/aks-rolling/grid-search/{code}")
def api_aks_rolling_detail(code: str):
    import json as _json
    path = BASE / "grid_search_aks" / "output" / "latest_rolling.json"
    if not path.exists():
        return {"error": "未生成"}
    with open(path) as f:
        data = _json.load(f)
    if code not in data:
        raise HTTPException(404, "指数不存在")
    return data[code]


@app.get("/api/grid-search/sell-tune")
def api_sell_tune():
    import json as _json
    json_path = BASE / "grid_search" / "output" / "sell_tune_results.json"
    if not json_path.exists():
        return {"error": "未生成"}
    with open(json_path) as f:
        return _json.load(f)


@app.get("/api/grid-search/sell-tune/{code}")
def api_sell_tune_demo(code: str):
    """运行新最优卖出策略, 返回回测曲线."""
    import numpy as np, sys
    sys.path.insert(0, str(BASE / "backtest"))
    sys.path.insert(0, str(BASE))
    from backtest import vec_rolling_pct, vec_rolling_mean_std, calc_xirr as _xirr
    from grid_search.tune_sell import run_simple_sell

    merged_path = MERGED_DIR / f"{code}.parquet"
    price_path = BASE / "data-store" / "parquet" / "index_price" / f"{code}.parquet"
    if not merged_path.exists() or not price_path.exists():
        raise HTTPException(404, "缺少数据")

    merged = pd.read_parquet(merged_path)
    price = pd.read_parquet(price_path)
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
        return []

    bt_df["price"] = bt_df[price_col].values
    bt_df["fed_val"] = bt_df["fed_dj"].values
    bt_df["pb_val"] = bt_df["pb_dj"].values if "pb_dj" in bt_df.columns else np.nan

    total_days = (bt_df["date"].max() - bt_df["date"].min()).days
    rpy = len(bt_df) / max(total_days / 365.25, 1)
    w = 8; wr = int(w * rpy)
    pb_arr = bt_df["pb_val"].values.astype(float)
    fed_arr = bt_df["fed_val"].values.astype(float)
    bt_df["pb_pct"] = vec_rolling_pct(pb_arr, wr) if not np.isnan(pb_arr).all() else np.full(len(bt_df), np.nan)
    m, s = vec_rolling_mean_std(fed_arr, wr)
    bt_df["fed_mean"] = m; bt_df["fed_std"] = s

    # 新最优卖出参数
    HE, EX, DT, HRATIO, ERATIO = 0.80, 0.85, 0.05, 0.35, 0.50
    result = run_simple_sell(bt_df, HE, EX, DT, HRATIO, ERATIO)
    flows_bt = result.get("cash_flows", [])
    if not flows_bt:
        return []

    # PB 日频数据
    pb_daily = merged[["date", "pb_dj"]].dropna(subset=["pb_dj"]).sort_values("date")
    pb_merged = pd.merge_asof(price_sorted, pb_daily, on="date", direction="backward")
    pb_raw_map = {}
    for _, r in pb_merged.iterrows():
        d = r["date"].strftime("%Y-%m-%d")
        if pd.notna(r["pb_dj"]): pb_raw_map[d] = round(float(r["pb_dj"]), 4)
    total_days_pb = (pb_daily["date"].max() - pb_daily["date"].min()).days
    rpy_pb = len(pb_daily) / max(total_days_pb / 365.25, 1)
    wr_pb = int(w * rpy_pb)
    pb_arr_daily = pb_daily["pb_dj"].values.astype(float)
    pb_pct_daily = vec_rolling_pct(pb_arr_daily, wr_pb)
    pb_daily2 = pb_daily.copy(); pb_daily2["pb_pct_daily"] = pb_pct_daily
    pb_pct_merged = pd.merge_asof(price_sorted, pb_daily2[["date", "pb_pct_daily"]], on="date", direction="backward")
    pb_pct_map = {}
    for _, r in pb_pct_merged.iterrows():
        d = r["date"].strftime("%Y-%m-%d")
        if pd.notna(r["pb_pct_daily"]): pb_pct_map[d] = round(float(r["pb_pct_daily"]), 4)

    # 交易明细
    trades_detail = []
    FLOOR, LOWV, MIDV, HIGHV = 0.10, 0.15, 0.35, 0.70
    for t in flows_bt:
        d, act, amt, sh, pr, pct, inv = str(t[0])[:10], t[1], float(t[2]), float(t[3]), float(t[4]), t[5], t[6]
        pct_str = f"PB%={(pct*100):.1f}%"
        if act == "buy":
            if pct < FLOOR: reason = f"{pct_str} < 10% → 3x, ¥4500"
            elif pct < LOWV: reason = f"{pct_str} < 15% → 2x, ¥3000"
            elif pct < MIDV: reason = f"{pct_str} < 35% → 1x, ¥1500"
            elif pct < HIGHV: reason = f"{pct_str} < 70% → 0.5x, ¥750"
            else: reason = f"{pct_str} ≥ 70% → 警告区"
        elif act == "sell":
            reason = f"{pct_str} ≥ {HE:.0%} → 卖出{HRATIO:.0%}持仓, ¥{abs(amt):.0f}"
        elif act == "clear":
            reason = f"{pct_str} ≥ {EX:.0%} 回撤≥{DT:.0%} → 清仓{ERATIO:.0%}, ¥{abs(amt):.0f}"
        else:
            reason = ""
        trades_detail.append({
            "date": d, "action": act, "amount": round(amt, 0), "shares": round(sh, 4),
            "price": round(pr, 2), "pb_pct": round(float(pct), 4) if pd.notna(pct) else None,
            "cum_invested": round(float(inv), 0), "reason": reason,
        })

    # 每日指标
    daily = []
    cf = []; last_shares = 0.0; last_cum = 0.0; total_cash = 0.0; flow_idx = 0
    last_pb_raw = 0.0; last_pb_pct = None
    for _, pr_row in price_sorted.iterrows():
        cur_date = pr_row["date"].strftime("%Y-%m-%d")
        cur_price = pr_row.get(price_col)
        buy_amt = 0.0; sell_amt = 0.0
        while flow_idx < len(flows_bt):
            fd_, fa, famt = flows_bt[flow_idx][0], flows_bt[flow_idx][1], flows_bt[flow_idx][2]
            if fd_ <= cur_date:
                if fa == "buy":
                    cf.append((fd_, -float(famt))); last_cum += float(famt); buy_amt += float(famt)
                elif fa in ("sell", "clear"):
                    cf.append((fd_, -float(famt))); sell_amt += abs(float(famt)); total_cash += abs(float(famt))
                last_shares = float(flows_bt[flow_idx][3]); flow_idx += 1
            else:
                break
        if cur_date[:10] in pb_raw_map: last_pb_raw = pb_raw_map[cur_date[:10]]
        if cur_date[:10] in pb_pct_map: last_pb_pct = pb_pct_map[cur_date[:10]]
        if cur_price is None or pd.isna(cur_price):
            daily.append({"date": cur_date, "cum_invested": 0, "equity": 0, "net_principal": 0,
                          "total_value": 0, "return_pct": 0, "xirr": 0.0,
                          "buy_amount": 0, "sell_amount": 0, "pb_raw": last_pb_raw, "pb_pct": last_pb_pct})
            continue
        eq = last_shares * cur_price if last_shares > 0 else 0
        total_value = eq + total_cash
        net_principal = max(last_cum - total_cash, 0)
        ret = (total_value - last_cum) / last_cum * 100 if last_cum > 0 else 0
        dx = _xirr(cf, cur_date, eq) if len(cf) >= 3 and eq > 0 else 0.0
        daily.append({
            "date": cur_date, "cum_invested": round(last_cum, 0),
            "equity": round(eq, 0), "net_principal": round(net_principal, 0),
            "total_value": round(total_value, 0), "return_pct": round(ret, 2),
            "xirr": round(dx, 4),
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
            "pb_raw": last_pb_raw, "pb_pct": last_pb_pct,
        })

    meta = {"window_years": w, "feature": "PB_FED 新卖出",
            "params": {"he": HE, "ex": EX, "dt": DT, "heavy_ratio": HRATIO, "extreme_ratio": ERATIO,
                       "buy_floor": FLOOR, "buy_high": HIGHV},
            "total_invested": result["invested"], "final_value": result["final_total"],
            "trades": result["trades"], "buys": result["buys"], "sells": result["sells"]}
    return {"meta": meta, "daily": daily, "trades": trades_detail}


@app.get("/api/grid-search/buy-only")
def api_buy_only():
    """返回 Buy-Only vs 有卖出 对比 JSON."""
    import json as _json
    json_path = BASE / "grid_search" / "output" / "buy_only_vs_sell.json"
    if not json_path.exists():
        return {"error": "数据未生成, 请先运行 grid_search/tune_sell.py"}
    with open(json_path) as f:
        return _json.load(f)


@app.get("/api/grid-search/buy-only/{code}")
def api_buy_only_demo(code: str):
    """运行 Buy-Only 策略, 返回回测曲线 (无卖出)."""
    import numpy as np, sys
    sys.path.insert(0, str(BASE / "backtest"))
    sys.path.insert(0, str(BASE))
    from backtest import vec_rolling_pct, vec_rolling_mean_std, calc_xirr as _xirr
    from grid_search.tune_sell import run_buy_only

    merged_path = MERGED_DIR / f"{code}.parquet"
    price_path = BASE / "data-store" / "parquet" / "index_price" / f"{code}.parquet"
    if not merged_path.exists() or not price_path.exists():
        raise HTTPException(404, "缺少数据")

    merged = pd.read_parquet(merged_path)
    price = pd.read_parquet(price_path)
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
        return []

    bt_df["price"] = bt_df[price_col].values
    bt_df["fed_val"] = bt_df["fed_dj"].values
    bt_df["pb_val"] = bt_df["pb_dj"].values if "pb_dj" in bt_df.columns else np.nan

    total_days = (bt_df["date"].max() - bt_df["date"].min()).days
    rpy = len(bt_df) / max(total_days / 365.25, 1)
    w = 8
    wr = int(w * rpy)
    pb_arr = bt_df["pb_val"].values.astype(float)
    fed_arr = bt_df["fed_val"].values.astype(float)
    bt_df["pb_pct"] = vec_rolling_pct(pb_arr, wr) if not np.isnan(pb_arr).all() else np.full(len(bt_df), np.nan)
    m, s = vec_rolling_mean_std(fed_arr, wr)
    bt_df["fed_mean"] = m; bt_df["fed_std"] = s

    BUY_PARAMS = {"floor": 0.10, "low": 0.15, "mid": 0.35, "high": 0.70, "warn": 0.70, "fed": -0.5}
    result = run_buy_only(bt_df, BUY_PARAMS)
    flows = result.get("cash_flows", [])
    if not flows:
        return []

    # PB 日频数据
    pb_daily = merged[["date", "pb_dj"]].dropna(subset=["pb_dj"]).sort_values("date")
    pb_merged = pd.merge_asof(price_sorted, pb_daily, on="date", direction="backward")
    pb_raw_map = {}
    for _, r in pb_merged.iterrows():
        d = r["date"].strftime("%Y-%m-%d")
        if pd.notna(r["pb_dj"]):
            pb_raw_map[d] = round(float(r["pb_dj"]), 4)
    # PB 百分位日频
    total_days_pb = (pb_daily["date"].max() - pb_daily["date"].min()).days
    rpy_pb = len(pb_daily) / max(total_days_pb / 365.25, 1)
    wr_pb = int(w * rpy_pb)
    pb_arr_daily = pb_daily["pb_dj"].values.astype(float)
    pb_pct_daily = vec_rolling_pct(pb_arr_daily, wr_pb)
    pb_daily2 = pb_daily.copy()
    pb_daily2["pb_pct_daily"] = pb_pct_daily
    pb_pct_merged = pd.merge_asof(price_sorted, pb_daily2[["date", "pb_pct_daily"]],
                                   on="date", direction="backward")
    pb_pct_map = {}
    for _, r in pb_pct_merged.iterrows():
        d = r["date"].strftime("%Y-%m-%d")
        if pd.notna(r["pb_pct_daily"]):
            pb_pct_map[d] = round(float(r["pb_pct_daily"]), 4)

    # 构建每日指标 + 交易明细
    daily = []
    trades_detail = []
    cf = []
    last_shares = 0.0; last_cum = 0.0; flow_idx = 0
    last_pb_raw = 0.0; last_pb_pct = None

    BEST_FL, BEST_LO, BEST_MI, BEST_HI = 0.10, 0.15, 0.35, 0.70
    for t in flows:
        d, act, amt, sh, pr, pct, inv = str(t[0])[:10], t[1], float(t[2]), float(t[3]), float(t[4]), t[5], t[6]
        pct_str = f"PB%={(pct*100):.1f}%"
        if pct < BEST_FL:
            reason = f"{pct_str} < 10% → 3倍定投, ¥1500×3 = ¥{amt:.0f}"
        elif pct < BEST_LO:
            reason = f"{pct_str} < 15% → 2倍定投, ¥1500×2 = ¥{amt:.0f}"
        elif pct < BEST_MI:
            reason = f"{pct_str} < 35% → 1倍定投, ¥1500×1 = ¥{amt:.0f}"
        elif pct < BEST_HI:
            reason = f"{pct_str} < 70% → 0.5倍定投, ¥1500×0.5 = ¥{amt:.0f}"
        else:
            reason = f"{pct_str} ≥ 70% → 警告区(不应买入)"
        trades_detail.append({
            "date": d, "action": act, "amount": round(amt, 0), "shares": round(sh, 4),
            "price": round(pr, 2), "pb_pct": round(float(pct), 4) if pd.notna(pct) else None,
            "cum_invested": round(float(inv), 0), "reason": reason,
        })

    for _, pr_row in price_sorted.iterrows():
        cur_date = pr_row["date"].strftime("%Y-%m-%d")
        cur_price = pr_row.get(price_col)
        buy_amt = 0.0
        while flow_idx < len(flows):
            fd_, fa, famt = flows[flow_idx][0], flows[flow_idx][1], flows[flow_idx][2]
            if fd_ <= cur_date:
                if fa == "buy":
                    cf.append((fd_, -float(famt))); last_cum += float(famt); buy_amt += float(famt)
                last_shares = float(flows[flow_idx][3]); flow_idx += 1
            else:
                break
        if cur_date[:10] in pb_raw_map:
            last_pb_raw = pb_raw_map[cur_date[:10]]
        if cur_date[:10] in pb_pct_map:
            last_pb_pct = pb_pct_map[cur_date[:10]]
        if cur_price is None or pd.isna(cur_price):
            daily.append({"date": cur_date, "cum_invested": 0, "equity": 0, "net_principal": 0,
                          "total_value": 0, "return_pct": 0, "xirr": 0.0,
                          "buy_amount": 0, "sell_amount": 0,
                          "pb_raw": last_pb_raw, "pb_pct": last_pb_pct})
            continue
        eq = last_shares * cur_price if last_shares > 0 else 0
        total_value = eq
        net_principal = last_cum
        ret = (total_value - last_cum) / last_cum * 100 if last_cum > 0 else 0
        dx = _xirr(cf, cur_date, eq) if len(cf) >= 3 and eq > 0 else 0.0
        daily.append({
            "date": cur_date, "cum_invested": round(last_cum, 0),
            "equity": round(eq, 0), "net_principal": round(net_principal, 0),
            "total_value": round(total_value, 0), "return_pct": round(ret, 2),
            "xirr": round(dx, 4),
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": 0,
            "pb_raw": last_pb_raw, "pb_pct": last_pb_pct,
        })

    meta = {"window_years": w, "feature": "Buy-Only",
            "params": {"pb_buy_floor": BEST_FL, "pb_buy_high": BEST_HI, "fed": BUY_PARAMS["fed"]},
            "total_invested": result["total_invested"], "final_value": result["final_value"],
            "trades": result["trades"], "buys": result["buys"]}
    return {"meta": meta, "daily": daily, "trades": trades_detail}


@app.get("/api/grid-search/strategy-demo/{code}")
def api_strategy_demo(code: str):
    """运行最优 PB_FED W=8yr 策略, 返回回测曲线."""
    import numpy as np, sys
    sys.path.insert(0, str(BASE / "backtest"))
    from backtest import vec_rolling_pct, vec_rolling_mean_std, calc_xirr as _xirr

    merged_path = MERGED_DIR / f"{code}.parquet"
    price_path = BASE / "data-store" / "parquet" / "index_price" / f"{code}.parquet"
    if not merged_path.exists() or not price_path.exists():
        raise HTTPException(404, "缺少数据")

    merged = pd.read_parquet(merged_path)
    price = pd.read_parquet(price_path)
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
        return []

    bt_df["price"] = bt_df[price_col].values
    bt_df["fed_val"] = bt_df["fed_dj"].values
    bt_df["pb_val"] = bt_df["pb_dj"].values if "pb_dj" in bt_df.columns else np.nan

    total_days = (bt_df["date"].max() - bt_df["date"].min()).days
    rpy = len(bt_df) / max(total_days / 365.25, 1)
    w = 8
    wr = int(w * rpy)

    pe_arr = bt_df["pe_ttm_dj"].values.astype(float)
    pb_arr = bt_df["pb_val"].values.astype(float)
    fed_arr = bt_df["fed_val"].values.astype(float)
    bt_df["pe_pct"] = vec_rolling_pct(pe_arr, wr)
    bt_df["pb_pct"] = vec_rolling_pct(pb_arr, wr) if not np.isnan(pb_arr).all() else np.full(len(bt_df), np.nan)
    m, s = vec_rolling_mean_std(fed_arr, wr)
    bt_df["fed_mean"] = m
    bt_df["fed_std"] = s

    # ---- PB 百分位 & PB 原始值: 从日频 PB 计算, merge_asof 到 K线 ----
    pb_daily = merged[["date", "pb_dj"]].dropna(subset=["pb_dj"]).copy()
    pb_daily = pb_daily.sort_values("date").reset_index(drop=True)
    # 日频滚动百分位
    total_days_pb = (pb_daily["date"].max() - pb_daily["date"].min()).days
    total_years_pb = max(total_days_pb / 365.25, 1)
    rpy_pb = len(pb_daily) / total_years_pb
    wr_pb = int(w * rpy_pb)
    pb_arr_daily = pb_daily["pb_dj"].values.astype(float)
    pb_pct_daily = vec_rolling_pct(pb_arr_daily, wr_pb)
    pb_daily["pb_pct_daily"] = pb_pct_daily
    # merge_asof 到 K线日期: PB 原始值 + PB 百分位
    pb_merged = pd.merge_asof(price_sorted, pb_daily[["date", "pb_dj", "pb_pct_daily"]],
                              on="date", direction="backward")
    pb_raw_map = {}
    pb_pct_map = {}
    for _, r in pb_merged.iterrows():
        d = r["date"].strftime("%Y-%m-%d")
        if pd.notna(r["pb_dj"]):
            pb_raw_map[d] = round(float(r["pb_dj"]), 4)
        if pd.notna(r["pb_pct_daily"]):
            pb_pct_map[d] = round(float(r["pb_pct_daily"]), 4)

    # Best PB_FED params
    BEST_PARAMS = (0.10, 0.15, 0.35, 0.70, 0.70, 0.75, 0.90, -0.5, 0.0, 0.0, 0.12, 0.04)
    FEAT_CFG = {"primary": "PB", "fed_gate": True, "pb_veto": False}

    from grid_search.grid_search_20260809 import run_one_combo
    result = run_one_combo(bt_df, BEST_PARAMS, FEAT_CFG)
    flows = result.get("cash_flows", [])
    if not flows:
        return []

    # ---- 构建每日指标 + 交易明细 ----
    daily = []
    trades_detail = []
    cf = []
    last_shares = 0.0; last_cum = 0.0; total_cash = 0.0; flow_idx = 0
    last_pb_raw = 0.0
    last_pb_pct = None

    # 先构建每笔交易的理由
    BEST_FL, BEST_LO, BEST_MI, BEST_HI = 0.10, 0.15, 0.35, 0.70
    BEST_HE, BEST_EX = 0.75, 0.90
    for t in flows:
        d, act, amt, sh, pr, pct, pb, inv = t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]
        pct_str = f"PB%={(pct*100):.1f}%"
        if act == "buy":
            if pct < BEST_FL:
                reason = f"{pct_str} < 10% → 3倍定投, ¥1500×3 = ¥{amt:.0f}"
            elif pct < BEST_LO:
                reason = f"{pct_str} < 15% → 2倍定投, ¥1500×2 = ¥{amt:.0f}"
            elif pct < BEST_MI:
                reason = f"{pct_str} < 35% → 1倍定投, ¥1500×1 = ¥{amt:.0f}"
            elif pct < BEST_HI:
                reason = f"{pct_str} < 70% → 0.5倍定投, ¥1500×0.5 = ¥{amt:.0f}"
            else:
                reason = f"{pct_str} ≥ 70% → 警告区(不应买入)"
        elif act == "sell":
            over = pct - BEST_HE
            sell_ratio = min(over * 0.10, 0.25)
            reason = f"{pct_str} ≥ 75% → 分批卖出, 比例=min(({pct:.1%}-75%)×0.1,25%)={sell_ratio:.0%}, ¥{abs(amt):.0f}"
        elif act == "clear":
            reason = f"{pct_str} ≥ 90% 且回撤≥4% → 清仓卖出, ¥{abs(amt):.0f}"
        else:
            reason = ""
        trades_detail.append({
            "date": str(d)[:10],
            "action": act,
            "amount": round(float(amt), 0),
            "shares": round(float(sh), 4),
            "price": round(float(pr), 2),
            "pb_pct": round(float(pct), 4) if pd.notna(pct) else None,
            "cum_invested": round(float(inv), 0),
            "reason": reason,
        })

    for _, pr_row in price_sorted.iterrows():
        cur_date = pr_row["date"].strftime("%Y-%m-%d")
        cur_price = pr_row.get(price_col)
        buy_amt = 0.0; sell_amt = 0.0
        while flow_idx < len(flows):
            fd_, fa, famt = flows[flow_idx][0], flows[flow_idx][1], flows[flow_idx][2]
            if fd_ <= cur_date:
                if fa == "buy":
                    cf.append((fd_, -famt)); last_cum += famt; buy_amt += famt
                elif fa in ("sell", "clear"):
                    cf.append((fd_, -famt)); sell_amt += abs(famt); total_cash += abs(famt)
                last_shares = flows[flow_idx][3]; flow_idx += 1
            else:
                break
        if cur_date[:10] in pb_raw_map:
            last_pb_raw = pb_raw_map[cur_date[:10]]
        if cur_date[:10] in pb_pct_map:
            last_pb_pct = pb_pct_map[cur_date[:10]]
        if cur_price is None or pd.isna(cur_price):
            daily.append({"date": cur_date, "cum_invested": 0, "equity": 0, "net_principal": 0,
                          "total_value": 0, "return_pct": 0, "xirr": 0.0,
                          "buy_amount": 0, "sell_amount": 0,
                          "pb_raw": last_pb_raw, "pb_pct": last_pb_pct})
            continue
        eq = last_shares * cur_price if last_shares > 0 else 0
        total_value = eq + total_cash
        net_principal = max(last_cum - total_cash, 0)
        ret = (total_value - last_cum) / last_cum * 100 if last_cum > 0 else 0
        dx = _xirr(cf, cur_date, eq) if len(cf) >= 3 and eq > 0 else 0.0
        # PB 百分位: 前向填充自日频计算值
        daily.append({
            "date": cur_date, "cum_invested": round(last_cum, 0),
            "equity": round(eq, 0), "net_principal": round(net_principal, 0),
            "total_value": round(total_value, 0), "return_pct": round(ret, 2),
            "xirr": round(dx, 4),
            "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
            "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
            "pb_raw": last_pb_raw,
            "pb_pct": last_pb_pct,
        })

    meta = {
        "window_years": w, "feature": "PB_FED",
        "params": {"pb_buy_floor": BEST_PARAMS[0], "pb_buy_high": BEST_PARAMS[3],
                   "pb_sell_heavy": BEST_PARAMS[5], "pb_sell_extreme": BEST_PARAMS[6],
                   "fed_buy_threshold": BEST_PARAMS[7]},
        "total_invested": result["total_invested"], "final_value": result["final_value"],
        "trades": result["trades"], "buys": result["buys"],
    }

    return {"meta": meta, "daily": daily, "trades": trades_detail}

@app.get("/api/grid-search/data")
def api_grid_search_data():
    """读取网格搜索 CSV 结果, 返回 JSON."""
    import glob as _g, os as _os
    out_dir = BASE / "grid_search" / "output"
    data = {}
    for fpath in sorted(_g.glob(str(out_dir / "top5_*.csv"))):
        fname = _os.path.basename(fpath)
        label = fname.replace("top5_", "").replace(".csv", "")
        try:
            df = pd.read_csv(fpath)
            data[label] = df.where(pd.notna(df), None).to_dict(orient="records")
        except Exception:
            data[label] = []

    # 获取元信息
    total_runs = 0
    generated_at = ""
    try:
        import json
        stat = out_dir.stat() if out_dir.exists() else None
    except Exception:
        stat = None

    for fpath in sorted(_g.glob(str(out_dir / "all_*.csv"))):
        try:
            df = pd.read_csv(fpath, nrows=0)
            total_runs = max(total_runs, len(pd.read_csv(fpath)))
        except Exception:
            pass
    if stat:
        generated_at = str(stat.st_mtime)

    return {"data": data, "total_runs": total_runs or len(data.get("w1_0_w0_0", [])), "generated_at": str(stat) if stat else ""}

@app.get("/api/indices")
def api_indices():
    result = []
    for code, (name, category) in sorted(INDEX_NAMES.items()):
        path = MERGED_DIR / f"{code}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        latest = df.iloc[-1]
        item = {
            "code": code, "name": name, "category": category,
            "rows": len(df),
            "date": str(latest["date"])[:10],
        }
        for col in ["pe_ttm_dj", "pe_ttm_csi", "pb_dj",
                      "bond_yield", "fed_dj", "fed_csi",
                      "pe_pct_dj", "pe_pct_csi",
                      "pb_pct_dj", "fed_pct_dj", "fed_pct_csi"]:
            if col in df.columns:
                v = latest[col]
                item[col] = round(float(v), 4) if pd.notna(v) else None
        if item.get("pe_ttm_csi") is not None:
            item["pe"] = item["pe_ttm_csi"]
            item["fed"] = item.get("fed_csi")
            item["pe_pct"] = item.get("pe_pct_csi")
        elif item.get("pe_ttm_dj") is not None:
            item["pe"] = item["pe_ttm_dj"]
            item["fed"] = item.get("fed_dj")
            item["pe_pct"] = item.get("pe_pct_dj")
        else:
            item["pe"] = None
        result.append(item)
    return result


@app.get("/api/indices/backtest")
def api_indices_backtest():
    """返回所有可回测指数的摘要 (meta only, 不含 daily)"""
    import numpy as np, glob as _glob, sys
    sys.path.insert(0, str(BASE / "backtest"))
    from backtest import vec_rolling_pct, vec_rolling_mean_std, run_one, WINDOW_YEARS_LIST

    results = []
    for code, (name, category) in sorted(INDEX_NAMES.items()):
        merged_path = MERGED_DIR / f"{code}.parquet"
        price_path = BASE / "data-store" / "parquet" / "index_price" / f"{code}.parquet"
        if not merged_path.exists() or not price_path.exists():
            continue
        merged = pd.read_parquet(merged_path)
        price = pd.read_parquet(price_path)
        merged["date"] = pd.to_datetime(merged["date"])
        price["date"] = pd.to_datetime(price["date"])
        dj_col = "pe_ttm_dj"
        if dj_col not in merged.columns or merged[dj_col].notna().sum() < 50:
            continue
        dj_mask = merged[dj_col].notna()
        dj = merged[dj_mask][["date", dj_col, "fed_dj", "pb_dj"]].copy()
        price_col = "index_open" if "index_open" in price.columns else "index_price"
        price_sorted = price[["date", price_col]].dropna().sort_values("date")
        dj_sorted = dj.sort_values("date")
        bt_df = pd.merge_asof(dj_sorted, price_sorted, on="date", direction="backward")
        bt_df = bt_df.dropna(subset=[price_col]).reset_index(drop=True)
        if len(bt_df) < 50:
            continue
        bt_df["price"] = bt_df[price_col].values
        bt_df["fed_val"] = bt_df["fed_dj"].values
        bt_df["pb_val"] = bt_df["pb_dj"].values if "pb_dj" in bt_df.columns else np.nan

        total_days = (bt_df["date"].max() - bt_df["date"].min()).days
        rpy = len(bt_df) / max(total_days / 365.25, 1)
        w = 8
        if total_days / 365.25 < 8:
            w = 5
        if total_days / 365.25 < 5:
            w = 3
        wr = int(w * rpy)

        # 统一使用沪深300最优参数 (大道至简, 一套策略适配所有宽基)
        BEST_PARAMS = (0.15, 0.30, 0.40, 0.70, 0.70, 0.85, 0.95, 1.0, 0.50, 0.70, 0.12, 0.04)
        for base_dir in [str(BASE / "backtest" / "output"), str(BASE / "backtest" / "output_20*")]:
            for csv_path in sorted(_glob.glob(os.path.join(base_dir, "*000300*", "dj_top20.csv")), reverse=True):
                try:
                    dfp = pd.read_csv(csv_path)
                    if 'window_years' not in dfp.columns:
                        continue
                    sub = dfp[dfp['window_years'] == w]
                    if len(sub) > 0:
                        r = sub.iloc[0]
                        BEST_PARAMS = (r['pe_buy_floor'], r['pe_buy_low'],
                            r['pe_buy_mid'], r['pe_buy_high'],
                            r['pe_sell_warn'], r['pe_sell_heavy'],
                            r['pe_sell_extreme'], r['fed_buy_threshold'],
                            r['pb_veto_threshold'], r['pb_confirm_threshold'],
                            r['drawdown_standard'], r['drawdown_tight'])
                        break
                except:
                    pass
            if BEST_PARAMS != (0.15, 0.30, 0.40, 0.70, 0.70, 0.85, 0.95, 1.0, 0.50, 0.70, 0.12, 0.04):
                break

        pe_arr = bt_df[dj_col].values.astype(float)
        pb_arr = bt_df["pb_val"].values.astype(float)
        fed_arr = bt_df["fed_val"].values.astype(float)
        bt_df[f"pe_pct_w{w}"] = vec_rolling_pct(pe_arr, wr)
        bt_df[f"pb_pct_w{w}"] = vec_rolling_pct(pb_arr, wr) if not np.isnan(pb_arr).all() else np.full(len(bt_df), np.nan)
        m, s = vec_rolling_mean_std(fed_arr, wr)
        bt_df[f"fed_mean_w{w}"] = m
        bt_df[f"fed_std_w{w}"] = s

        w_idx = WINDOW_YEARS_LIST.index(w)
        result = run_one(bt_df, BEST_PARAMS, w_idx, 0)
        if not result or result.get("trades", 0) < 2:
            continue

        # 修正回报: include 累积卖出所得现金
        flows_raw = result.get("cash_flows", [])
        cum_cash = sum(abs(t[2]) for t in flows_raw if t[1] in ("sell", "clear"))
        total_val = result.get("final_value", 0) + cum_cash
        inv = result.get("total_invested", 0)
        corrected_return = (total_val - inv) / inv if inv > 0 else 0

        results.append({
            "code": code, "name": name, "category": category,
            "window_years": w,
            "xirr": result.get("xirr", 0),
            "final_return": round(corrected_return, 4),
            "trades": result.get("trades", 0),
            "buys": result.get("buys", 0),
            "pe_buy_floor": BEST_PARAMS[0], "pe_buy_high": BEST_PARAMS[3],
            "pe_sell_heavy": BEST_PARAMS[5], "pe_sell_extreme": BEST_PARAMS[6],
        })
    return results


@app.get("/api/indices/{code}")
def api_index_detail(
    code: str,
    start: str = Query(None),
    end: str = Query(None),
):
    df = _read_index(code)
    rows = df.to_dict(orient="records")
    if start:
        rows = [r for r in rows if r["date"] >= start]
    if end:
        rows = [r for r in rows if r["date"] <= end]
    return rows


@app.get("/api/compare")
def api_compare(codes: str = Query(...)):
    result = {}
    for code in codes.split(","):
        code = code.strip()
        path = MERGED_DIR / f"{code}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "pe_ttm_csi" in df.columns:
            ser = df.set_index("date")["pe_ttm_csi"]
        elif "pe_ttm_dj" in df.columns:
            ser = df.set_index("date")["pe_ttm_dj"]
        else:
            continue
        result[code] = {str(k)[:10]: round(float(v), 4) if pd.notna(v) else None
                        for k, v in ser.dropna().items()}
    return result


@app.get("/api/wind/{code}")
def api_wind_data(code: str):
    """返回 Wind PE 时序数据（若存在）。"""
    path = WIND_DIR / f"{code}.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "date": r["date"],
            "pe_ttm_wind": round(float(r["pe_ttm_wind"]), 4) if pd.notna(r["pe_ttm_wind"]) else None,
        })
    return rows


@app.get("/api/aligned/{code}")
def api_aligned_data(code: str):
    """返回对齐后的 PE 时序数据（Wind 早期 + 现有近期）。"""
    path = ALIGNED_DIR / f"{code}.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    cols = ["date", "pe_aligned", "pe_source", "pe_pct_aligned"]
    for c in ["pe_ttm_wind", "pe_ttm_ours", "pb_dj", "bond_yield"]:
        if c in df.columns:
            cols.append(c)
    str_cols = {"date", "pe_source"}
    df = df[cols].where(pd.notna(df), None)
    rows = []
    for _, r in df.iterrows():
        item = {"date": r["date"]}
        for c in cols:
            if c in str_cols:
                item[c] = r[c]
            else:
                v = r[c]
                item[c] = round(float(v), 4) if v is not None and pd.notna(v) else None
        rows.append(item)
    return rows


@app.get("/analysis/{code}", response_class=HTMLResponse)
def analysis_page(code: str):
    path = MERGED_DIR / f"{code}.parquet"
    if not path.exists():
        raise HTTPException(404, "指数不存在")
    with open(BASE / "templates" / "analysis.html") as f:
        html = f.read()
    return html.replace("{{CODE}}", code)


@app.get("/api/analysis/{code}")
def api_analysis_data(code: str):
    """返回 K线(收盘价) + 蛋卷 PE + 蛋卷 PE 10年滚动百分位，按 K线日期对齐。"""
    import numpy as np

    merged_path = MERGED_DIR / f"{code}.parquet"
    price_path = BASE / "data-store" / "parquet" / "index_price" / f"{code}.parquet"

    if not merged_path.exists() or not price_path.exists():
        raise HTTPException(404, "缺少数据")

    merged = pd.read_parquet(merged_path)
    price = pd.read_parquet(price_path)

    merged["date"] = pd.to_datetime(merged["date"])
    price["date"] = pd.to_datetime(price["date"])

    # K线数据
    price_col = "index_open" if "index_open" in price.columns else "index_price"
    kline = price[["date", price_col]].dropna()
    if kline.empty:
        raise HTTPException(404, "无K线数据")

    # 蛋卷 PE（周频，过滤非空）
    dj_col = "pe_ttm_dj"
    if dj_col not in merged.columns or merged[dj_col].notna().sum() < 50:
        raise HTTPException(404, "无蛋卷PE数据")
    dj = merged[["date", dj_col]].dropna(subset=[dj_col]).copy()

    # 10年滚动百分位（周频对应 520 行窗口）
    dj_sorted = dj.sort_values("date")
    rpy = len(dj_sorted) / 10  # rows per year
    window_rows = int(10 * rpy)
    dj_sorted["pe_pct_10yr"] = _rolling_pct(dj_sorted[dj_col].values.astype(float), window_rows)

    # 以 K线日期为索引，用 merge_asof 反向匹配蛋卷数据
    # （周日发布的 PE 匹配到周一~周三的 K线日期）
    kline_sorted = kline.sort_values("date")
    dj_sorted = dj_sorted.sort_values("date")
    result = pd.merge_asof(kline_sorted, dj_sorted, on="date", direction="backward")
    result = result.where(pd.notna(result), None)

    rows = []
    for _, r in result.iterrows():
        item = {
            "date": r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else None,
            "open": round(float(r[price_col]), 2) if r.get(price_col) is not None and pd.notna(r.get(price_col)) else None,
            "pe_dj": round(float(r[dj_col]), 2) if r.get(dj_col) is not None and pd.notna(r.get(dj_col)) else None,
            "pe_pct_10yr": round(float(r["pe_pct_10yr"]), 2) if r.get("pe_pct_10yr") is not None and pd.notna(r.get("pe_pct_10yr")) else None,
        }
        rows.append(item)
    return rows


@app.get("/api/analysis/{code}/backtest")
def api_analysis_backtest(code: str):
    """运行最优蛋卷参数回测，返回现金流曲线（投入/市值/回报%）。"""
    import numpy as np, os, glob, sys
    sys.path.insert(0, str(BASE / "backtest"))
    from backtest import vec_rolling_pct, vec_rolling_mean_std, run_one, WINDOW_YEARS_LIST

    merged_path = MERGED_DIR / f"{code}.parquet"
    price_path = BASE / "data-store" / "parquet" / "index_price" / f"{code}.parquet"
    if not merged_path.exists() or not price_path.exists():
        raise HTTPException(404, "缺少数据")

    merged = pd.read_parquet(merged_path)
    price = pd.read_parquet(price_path)
    merged["date"] = pd.to_datetime(merged["date"])
    price["date"] = pd.to_datetime(price["date"])

    dj_col = "pe_ttm_dj"
    if dj_col not in merged.columns or merged[dj_col].notna().sum() < 50:
        raise HTTPException(404, "无蛋卷PE数据")

    # 蛋卷数据 (周频)
    dj_mask = merged[dj_col].notna()
    dj = merged[dj_mask][["date", dj_col, "fed_dj", "pb_dj"]].copy()
    # 用 merge_asof 给蛋卷日期匹配最近的价格
    price_col2 = "index_open" if "index_open" in price.columns else "index_price"
    price_sorted = price[["date", price_col2]].dropna().sort_values("date")
    dj_sorted = dj.sort_values("date")
    bt_df = pd.merge_asof(dj_sorted, price_sorted, on="date", direction="backward")
    bt_df = bt_df.dropna(subset=[price_col2]).reset_index(drop=True)

    if len(bt_df) < 50:
        return []

    bt_df["price"] = bt_df[price_col2].values
    bt_df["fed_val"] = bt_df["fed_dj"].values
    bt_df["pb_val"] = bt_df["pb_dj"].values if "pb_dj" in bt_df.columns else np.nan

    # 计算滚动百分位 (8yr 优先)
    total_days = (bt_df["date"].max() - bt_df["date"].min()).days
    rpy = len(bt_df) / max(total_days / 365.25, 1)
    w = 8
    if total_days / 365.25 < 8:
        w = 5
    if total_days / 365.25 < 5:
        w = 3
    wr = int(w * rpy)

    # 统一使用沪深300最优参数
    BEST_PARAMS = (0.15, 0.30, 0.40, 0.70, 0.70, 0.85, 0.95, 1.0, 0.50, 0.70, 0.12, 0.04)
    import glob as _glob
    for base_dir in [str(BASE / "backtest" / "output"), str(BASE / "backtest" / "output_20*")]:
        for csv_path in sorted(_glob.glob(os.path.join(base_dir, "*000300*", "dj_top20.csv")), reverse=True):
            try:
                df = pd.read_csv(csv_path)
                if 'window_years' not in df.columns: continue
                sub = df[df['window_years'] == w]
                if len(sub) > 0:
                    r = sub.iloc[0]
                    BEST_PARAMS = (r['pe_buy_floor'], r['pe_buy_low'],
                        r['pe_buy_mid'], r['pe_buy_high'],
                        r['pe_sell_warn'], r['pe_sell_heavy'],
                        r['pe_sell_extreme'], r['fed_buy_threshold'],
                        r['pb_veto_threshold'], r['pb_confirm_threshold'],
                        r['drawdown_standard'], r['drawdown_tight'])
                    break
            except: pass
        if BEST_PARAMS != (0.15, 0.30, 0.40, 0.70, 0.70, 0.85, 0.95, 1.0, 0.50, 0.70, 0.12, 0.04):
            break

    pe_arr = bt_df[dj_col].values.astype(float)
    pb_arr = bt_df["pb_val"].values.astype(float)
    fed_arr = bt_df["fed_val"].values.astype(float)
    bt_df[f"pe_pct_w{w}"] = vec_rolling_pct(pe_arr, wr)
    bt_df[f"pb_pct_w{w}"] = vec_rolling_pct(pb_arr, wr) if not np.isnan(pb_arr).all() else np.full(len(bt_df), np.nan)
    m, s = vec_rolling_mean_std(fed_arr, wr)
    bt_df[f"fed_mean_w{w}"] = m
    bt_df[f"fed_std_w{w}"] = s

    # (rm this line: BEST_PARAMS already set above)
    w_idx = WINDOW_YEARS_LIST.index(w)

    result = run_one(bt_df, BEST_PARAMS, w_idx, 0)
    flows = result.get("cash_flows", [])
    if not flows or not isinstance(flows, list):
        return []

    # 构建现金流曲线: date, cum_invested, equity, return_pct
    from backtest import calc_xirr as _xirr

    rows = []
    cum_invested = 0
    cum_cash = 0
    cashflows_so_far = []  # 累计现金流用于算运行 XIRR
    for t in flows:
        d, act, amt = t[0], t[1], t[2]
        sh = t[3]
        pr = t[4]
        if act == "buy":
            cum_invested += amt
            cashflows_so_far.append((d, -amt))  # 买入=现金流出
        elif act in ("sell", "clear"):
            cashflows_so_far.append((d, -amt))  # amt为负, -amt=正=现金流入
            cum_cash += abs(amt)

        equity = sh * pr if pr else 0
        total_value = equity + cum_cash
        ret_pct = (total_value - cum_invested) / cum_invested * 100 if cum_invested > 0 else 0

        # 运行 XIRR
        run_xirr = 0.0
        if len(cashflows_so_far) >= 2 and equity > 0:
            run_xirr = _xirr(cashflows_so_far, d, equity)

        rows.append({
            "date": d,
            "action": act,
            "amount": round(float(amt), 0),
            "cum_invested": round(cum_invested, 0),
            "net_principal": round(cum_invested - cum_cash, 0),
            "cum_cash": round(cum_cash, 0),
            "equity": round(equity, 0),
            "total_value": round(total_value, 0),
            "return_pct": round(ret_pct, 2),
            "xirr": round(run_xirr, 4),
        })

    meta = {
        "window_years": w,
        "params": {
            "pe_buy_floor": BEST_PARAMS[0], "pe_buy_low": BEST_PARAMS[1],
            "pe_buy_mid": BEST_PARAMS[2], "pe_buy_high": BEST_PARAMS[3],
            "pe_sell_warn": BEST_PARAMS[4], "pe_sell_heavy": BEST_PARAMS[5],
            "pe_sell_extreme": BEST_PARAMS[6], "fed_buy_threshold": BEST_PARAMS[7],
            "pb_veto_threshold": BEST_PARAMS[8],
        },
        "final_return": round(result.get("final_return", 0), 4),
        "xirr": round(result.get("xirr", 0), 4),
    }

    # 构建每日指标 (展开到K线日期, 所有指标均可在日频对齐)
    daily = []
    if rows:
        cf = []
        last_shares = 0
        last_cum = 0
        total_cash = 0
        flow_idx = 0
        for _, pr_row in price_sorted.iterrows():
            cur_date = pr_row["date"].strftime("%Y-%m-%d")
            cur_price = pr_row.get(price_col2)
            buy_amt = 0
            sell_amt = 0
            while flow_idx < len(flows):
                fd, fa, famt = flows[flow_idx][0], flows[flow_idx][1], flows[flow_idx][2]
                if fd <= cur_date:
                    if fa == "buy":
                        cf.append((fd, -famt))
                        last_cum += famt
                        buy_amt += famt
                    elif fa in ("sell", "clear"):
                        cf.append((fd, -famt))
                        sell_amt += abs(famt)
                        total_cash += abs(famt)
                    last_shares = flows[flow_idx][3]
                    flow_idx += 1
                else:
                    break
            if cur_price is None or pd.isna(cur_price):
                daily.append({"date": cur_date, "cum_invested": 0, "equity": 0, "net_principal": 0, "total_value": 0, "return_pct": 0, "xirr": 0.0})
                continue
            eq = last_shares * cur_price if last_shares > 0 else 0
            total_value = eq + total_cash
            net_principal = max(last_cum - total_cash, 0)
            ret = (total_value - last_cum) / last_cum * 100 if last_cum > 0 else 0
            dx = _xirr(cf, cur_date, eq) if len(cf) >= 3 and eq > 0 else 0.0
            daily.append({
                "date": cur_date,
                "cum_invested": round(last_cum, 0),
                "equity": round(eq, 0),
                "net_principal": round(net_principal, 0),
                "total_value": round(total_value, 0),
                "return_pct": round(ret, 2),
                "xirr": round(dx, 4),
                "buy_amount": round(buy_amt, 0) if buy_amt > 0 else 0,
                "sell_amount": round(sell_amt, 0) if sell_amt > 0 else 0,
            })

    return {"meta": meta, "flows": rows, "daily": daily}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
