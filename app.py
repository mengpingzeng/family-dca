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


# ============================================================================
# API
# ============================================================================

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

    # K线数据（开盘价）
    kline = price[["date", "index_open", "index_price"]].dropna()
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
            "open": round(float(r["index_open"]), 2) if r.get("index_open") is not None and pd.notna(r.get("index_open")) else None,
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
    # 用 merge_asof 给蛋卷日期匹配最近的开盘价
    price_sorted = price[["date", "index_open"]].dropna().sort_values("date")
    dj_sorted = dj.sort_values("date")
    bt_df = pd.merge_asof(dj_sorted, price_sorted, on="date", direction="backward")
    bt_df = bt_df.dropna(subset=["index_open"]).reset_index(drop=True)

    if len(bt_df) < 50:
        return []

    bt_df["price"] = bt_df["index_open"].values
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

    # 从 CSV 读取该窗口的最优参数
    BEST_PARAMS = (0.15, 0.30, 0.40, 0.70, 0.70, 0.85, 0.95, 1.0, 0.50, 0.70, 0.12, 0.04)
    import glob as _glob
    for base_dir in [str(BASE / "backtest" / "output"), str(BASE / "backtest" / "output_20*")]:
        for csv_path in sorted(_glob.glob(os.path.join(base_dir, f"*{code}*", "dj_top20.csv")), reverse=True):
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
            cur_price = pr_row.get("index_open")
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
            net_principal = last_cum - total_cash
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
