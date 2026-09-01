#!/usr/bin/env python3
"""
成交记账服务 — FastAPI (端口 8001)

用户买入/卖出后，通过浏览器表单提交真实成交记录，写入 ledger.json，
确保账本与实际账户资金一致。

页面:
  GET  /trade              记账表单 (可选 ?code=000300&action=buy 预填)
  GET  /trade/confirm      记账成功确认页
  POST /api/trade          提交成交 (表单)
  GET  /api/ledger         返回当前账本 (JSON)

账本更新规则:
  买入: shares += 数量; avg_cost 按加权平均重算; total_invested += 金额
  卖出: shares -= 数量; avg_cost 不变; 卖出回笼金额不扣 total_invested
  频率字段: last_buy_week / last_sell_month 更新为当前周/月

用法:
  python wind_new_search/push/trade_server.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

PUSH_DIR = Path(__file__).resolve().parent
LEDGER_PATH = PUSH_DIR / "ledger.json"
CONFIG_PATH = PUSH_DIR / "config.json"

app = FastAPI(title="宽基理财 · 成交记账")


def load_ledger():
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {}


def save_ledger(ledger):
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _this_week():
    d = datetime.now().isocalendar()
    return f"{d.year}-{d.week}"


def _this_month():
    return datetime.now().strftime("%Y-%m")


def _read_etf_price(etf_code):
    """读取 ETF 最新收盘价, 用于表单预填参考价."""
    if not etf_code:
        return None
    path = PROJECT_DIR / "data-store" / "parquet" / "etf" / f"{etf_code}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if len(df) == 0:
        return None
    return float(df["close"].iloc[-1])


def apply_trade(ledger, code, name, action, shares, price):
    """应用一笔成交到账本, 返回 (更新后的entry, 消息)."""
    entry = ledger.get(code, {"name": name, "shares": 0.0, "avg_cost": 0.0,
                              "total_invested": 0.0, "last_buy_week": None,
                              "last_sell_month": None})
    if action == "buy":
        old_shares = float(entry.get("shares", 0.0))
        old_cost = float(entry.get("avg_cost", 0.0))
        new_shares = old_shares + shares
        # 加权平均成本
        entry["avg_cost"] = (old_cost * old_shares + price * shares) / new_shares if new_shares > 0 else price
        entry["shares"] = new_shares
        entry["total_invested"] = float(entry.get("total_invested", 0.0)) + price * shares
        entry["last_buy_week"] = _this_week()
        msg = f"✅ 已记录买入 {name} {shares:.0f}份 @ ¥{price:.3f} (成本¥{price*shares:,.0f})"
    else:  # sell
        old_shares = float(entry.get("shares", 0.0))
        if shares > old_shares:
            raise ValueError(f"卖出份额({shares:.0f})超过持仓({old_shares:.0f})")
        entry["shares"] = old_shares - shares
        entry["last_sell_month"] = _this_month()
        msg = f"✅ 已记录卖出 {name} {shares:.0f}份 @ ¥{price:.3f} (回笼¥{price*shares:,.0f})"
    ledger[code] = entry
    return entry, msg


# ============================================================================
# 页面
# ============================================================================

@app.get("/trade", response_class=HTMLResponse)
def trade_page(code: str = "", action: str = ""):
    cfg = load_config()
    ledger = load_ledger()
    options = []
    for c, info in cfg["indices"].items():
        held = ledger.get(c, {})
        shares = float(held.get("shares", 0.0))
        etf_price = _read_etf_price(info.get("etf"))
        ref = etf_price if etf_price else ""
        sel = " selected" if c == code else ""
        options.append(f'<option value="{c}"{sel} data-ref="{ref}">{info["name"]}</option>')
    action_buy = " checked" if action != "sell" else ""
    action_sell = " checked" if action == "sell" else ""
    held_info = ""
    if code and code in ledger and float(ledger[code].get("shares", 0)) > 0:
        h = ledger[code]
        held_info = (f'<div class="held">当前持仓: {float(h["shares"]):.0f}份 '
                     f'@ 成本{h.get("avg_cost", 0):.3f}</div>')
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>成交记账</title><style>
body{{font-family:-apple-system,sans-serif;max-width:480px;margin:0 auto;padding:16px;background:#f5f6fa;color:#333}}
h1{{font-size:20px}}
.card{{background:#fff;border-radius:12px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
label{{display:block;margin-top:12px;font-size:13px;color:#666}}
select,input{{width:100%;padding:10px;border:1px solid #ddd;border-radius:8px;font-size:16px;box-sizing:border-box;margin-top:4px}}
.radios{{display:flex;gap:16px;margin-top:4px}}
button{{width:100%;padding:12px;margin-top:16px;border:none;border-radius:8px;font-size:16px;color:#fff;background:#07c160;cursor:pointer}}
.held{{margin-top:8px;font-size:13px;color:#07c160}}
.ref{{font-size:12px;color:#999}}
a{{color:#07c160}}
</style></head><body>
<h1>📝 成交记账</h1>
<div class="card">
<form method="post" action="/api/trade">
<label>指数</label>
<select name="code" id="code" onchange="upd()">{''.join(options)}</select>
<label>方向</label>
<div class="radios">
<label><input type="radio" name="action" value="buy"{action_buy}> 买入</label>
<label><input type="radio" name="action" value="sell"{action_sell}> 卖出</label>
</div>
{held_info}
<label>成交份额</label>
<input type="number" name="shares" id="shares" min="0" step="1" value="1000" required>
<label>成交价(元) <span class="ref" id="ref"></span></label>
<input type="number" name="price" id="price" step="0.001" value="" required>
<button type="submit">提交记账</button>
</form>
</div>
<script>
function upd(){{
  var sel=document.getElementById('code');
  var opt=sel.options[sel.selectedIndex];
  var ref=opt.getAttribute('data-ref');
  document.getElementById('ref').textContent=ref?('参考: ¥'+ref):'';
  if(ref && !document.getElementById('price').value) document.getElementById('price').value=ref;
}}
upd();
</script>
</body></html>"""
    return HTMLResponse(html)


@app.get("/trade/confirm", response_class=HTMLResponse)
def confirm_page(msg: str = ""):
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>记账成功</title><style>
body{{font-family:-apple-system,sans-serif;max-width:480px;margin:0 auto;padding:16px;text-align:center;background:#f5f6fa;color:#333}}
h1{{font-size:22px;color:#07c160}}
.card{{background:#fff;border-radius:12px;padding:24px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
a{{display:inline-block;margin-top:16px;color:#07c160}}
</style></head><body>
<div class="card"><h1>{msg}</h1>
<p><a href="/trade">继续记账</a> · <a href="/trade">查看账本</a></p>
</div></body></html>"""
    return HTMLResponse(html)


# ============================================================================
# API
# ============================================================================

@app.post("/api/trade")
async def submit_trade(code: str = Form(...), action: str = Form(...),
                       shares: float = Form(...), price: float = Form(...)):
    cfg = load_config()
    ledger = load_ledger()
    if code not in cfg["indices"]:
        return JSONResponse({"error": f"未知指数 {code}"}, status_code=400)
    name = cfg["indices"][code]["name"]
    if shares <= 0:
        return JSONResponse({"error": "份额必须>0"}, status_code=400)
    if price <= 0:
        return JSONResponse({"error": "价格必须>0"}, status_code=400)
    try:
        entry, msg = apply_trade(ledger, code, name, action, shares, price)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    save_ledger(ledger)
    return RedirectResponse(f"/trade/confirm?msg={msg}", status_code=303)


@app.get("/api/ledger")
def api_ledger():
    return JSONResponse(load_ledger())


@app.get("/api/config")
def api_config():
    cfg = load_config()
    return JSONResponse({"indices": cfg["indices"]})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
