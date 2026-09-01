#!/usr/bin/env python3
"""生成 mean_20day.html — 主要宽基指数 20日均线上下 K线 高亮图。

高于 20 日均线: K线绘制为蓝色; 低于 20 日均线: K线绘制为红色。
输出为自包含单文件网页 (内嵌数据 + echarts)。

用法: python3 build_mean20_html.py [echarts.min.js 路径]
"""

import json, os, sys
from datetime import datetime
import numpy as np
import pandas as pd
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
MERGED = os.path.join(BASE, "data-store", "parquet", "merged")
OUT = os.path.join(BASE, "mean_20day.html")
ECHARTS_URL = "https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"

# 主要宽基指数
INDICES = [
    ("000300", "沪深300"),
    ("000905", "中证500"),
    ("000852", "中证1000"),
    ("000016", "上证50"),
    ("000688", "科创50"),
    ("000510", "中证A500"),
    ("399006", "创业板指"),
    ("399330", "深证100"),
]

COLOR_ABOVE = {"color": "#3b82f6", "color0": "#60a5fa", "borderColor": "#2563eb", "borderColor0": "#93c5fd"}
COLOR_BELOW = {"color": "#ef4444", "color0": "#f87171", "borderColor": "#b91c1c", "borderColor0": "#fca5a5"}
COLOR_NA = {"color": "#94a3b8", "color0": "#94a3b8", "borderColor": "#64748b", "borderColor0": "#64748b"}


def load_index(code: str):
    df = pd.read_parquet(os.path.join(MERGED, f"{code}.parquet"))
    df = df[["date", "index_price"]].copy()
    df = df.dropna(subset=["index_price"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["close"] = df["index_price"].astype(float)
    # 尽量补充开盘价: index_price -> index_price_aks, 都没有则用收盘价替代
    df["index_open"] = None
    src = os.path.join(BASE, "data-store", "parquet", "index_price", f"{code}.parquet")
    if os.path.exists(src):
        s = pd.read_parquet(src)
        if "index_open" in s.columns:
            s = s[["date", "index_open"]].dropna(subset=["index_open"])
            if len(s):
                s["date"] = pd.to_datetime(s["date"])
                df = df.drop(columns=["index_open"]).merge(s, on="date", how="left")
    if "index_open" not in df.columns or df["index_open"].notna().sum() < 20:
        src2 = os.path.join(BASE, "data-store", "parquet", "index_price_aks", f"{code}.parquet")
        if os.path.exists(src2):
            s2 = pd.read_parquet(src2)[["date", "index_open"]].dropna(subset=["index_open"])
            s2["date"] = pd.to_datetime(s2["date"])
            df = df.drop(columns=["index_open"], errors="ignore").merge(s2, on="date", how="left")
    df["open"] = df["index_open"].astype(float).fillna(df["close"])
    # MA20(第 t 个交易日) = 收盘价在 [t-19, t] 窗口的均值, 只用到当天及之前的数据, 不含未来信息
    df["ma20"] = df["close"].rolling(window=20, min_periods=20).mean()
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return df


def verify_ma20_no_future(code: str, df) -> None:
    """独立复核 MA20 不含未来信息: 逐日累加窗口重算, 并校验日期严格递增。"""
    closes = df["close"].astype(float).values
    ma20 = df["ma20"].astype(float).values
    dates = df["date"].values
    assert (dates[1:] > dates[:-1]).all(), f"{code} 日期未严格递增"
    n = len(closes)
    for i in range(n):
        lo = max(0, i - 19)
        if i - lo + 1 < 20:
            assert pd.isna(ma20[i]), f"{code} 前19根应无MA20, 第{i}根异常"
        else:
            expect = closes[lo:i + 1].mean()
            assert abs(expect - ma20[i]) < 1e-6, f"{code} MA20 与逐日重算不一致 @ {dates[i]}"


def detect_ma20_signal(df):
    """基于 20 日均线的连续上涨/连续下降区间 + 顶底判断。

    方法(均为常见成熟方法):
      1. 趋势区间: 收盘 >= MA20 记为上涨区, < MA20 记为下跌区; 连续相同即为"连续上涨/连续下降"区间。
      2. 区间极值: 每个连续上涨区间的最高收盘点 -> 区间顶; 连续下跌区间的最低收盘点 -> 区间底。
      3. TD Sequential (DeMark 衰竭计数): 连续 N 天收盘高于/低于 4 天前收盘, 达到 9/13 视为趋势衰竭,
         卖出9 提示顶部, 买入9 提示底部。
    """
    closes = df["close"].astype(float).values
    ma = df["ma20"].values
    highs = np.maximum(df["open"].astype(float).values, closes)
    lows = np.minimum(df["open"].astype(float).values, closes)
    dates = df["date"].dt.strftime("%Y-%m-%d")
    n = len(closes)

    regimes = [None] * n
    run_len = [0] * n
    runs = []  # (kind, start, end)

    prev, start = None, 0
    for i in range(n):
        if pd.isna(ma[i]):
            if prev is not None:
                runs.append((prev, start, i - 1))
                prev = None
            regimes[i] = None
            run_len[i] = 0
            continue
        r = "up" if closes[i] >= ma[i] else "down"
        regimes[i] = r
        if r != prev:
            if prev is not None:
                runs.append((prev, start, i - 1))
            prev, start = r, i
        run_len[i] = i - start + 1
    if prev is not None:
        runs.append((prev, start, n - 1))

    tops, bottoms = [], []
    for kind, s, e in runs:
        seg = closes[s:e + 1]
        if kind == "up":
            k = int(np.argmax(seg)) + s
            tops.append([dates[k], round(highs[k], 2)])
        else:
            k = int(np.argmin(seg)) + s
            bottoms.append([dates[k], round(lows[k], 2)])

    td9_top, td9_bot = [], []
    sc = bc = 0
    for i in range(4, n):
        sc = sc + 1 if closes[i] > closes[i - 4] else 0
        bc = bc + 1 if closes[i] < closes[i - 4] else 0
        if sc in (9, 13):
            td9_top.append([dates[i], round(highs[i], 2)])
        if bc in (9, 13):
            td9_bot.append([dates[i], round(lows[i], 2)])

    return {"run": run_len, "regimes": regimes, "tops": tops, "bottoms": bottoms,
            "td9_top": td9_top, "td9_bot": td9_bot}


def build_index(code: str, name: str):
    df = load_index(code)
    dates, kdata, ma20, dev, status = [], [], [], [], []
    for _, r in df.iterrows():
        o, c = r["open"], r["close"]
        m = r["ma20"]
        d = r["date"].strftime("%Y-%m-%d")
        dates.append(d)
        lo, hi = min(o, c), max(o, c)
        if pd.notna(m):
            st = "a" if c >= m else "b"
            d20 = round(float(m), 2)
            dev.append(round((c / m - 1) * 100, 2))
        else:
            st = "n"
            d20 = None
            dev.append(None)
        kdata.append([round(o, 2), round(c, 2), round(lo, 2), round(hi, 2), st])
        ma20.append(d20)
        status.append(st)
    sig = detect_ma20_signal(df)
    last = -1
    run_days = sig["run"][last]
    cur = {
        "date": dates[last], "close": kdata[last][1], "ma20": ma20[last],
        "dev": dev[last], "status": status[last],
        "run_days": run_days,
        "run_start": dates[last - run_days + 1] if run_days > 0 else None,
        "recent_top": sig["tops"][-1] if sig["tops"] else None,
        "recent_bottom": sig["bottoms"][-1] if sig["bottoms"] else None,
    }
    return {"name": name, "dates": dates, "k": kdata, "ma20": ma20, "dev": dev, "run": sig["run"],
            "tops": sig["tops"], "bottoms": sig["bottoms"],
            "td9_top": sig["td9_top"], "td9_bot": sig["td9_bot"], "cur": cur}


def main():
    echarts_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/echarts.min.js"
    if not os.path.exists(echarts_path):
        print(f"下载 echarts -> {echarts_path}")
        os.makedirs(os.path.dirname(echarts_path), exist_ok=True)
        urllib.request.urlretrieve(ECHARTS_URL, echarts_path)
    if not os.path.exists(echarts_path):
        sys.exit(f"echarts.min.js 不存在: {echarts_path}")
    with open(echarts_path) as f:
        echarts_src = f.read()

    indices = [build_index(code, name) for code, name in INDICES]

    # 逐一独立复核: 20日均线不含未来信息
    for code, name in INDICES:
        verify_ma20_no_future(code, load_index(code))
        print(f"  [校验OK] {name:<8} MA20 不含未来信息")

    payload = json.dumps(indices, ensure_ascii=False, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>主要宽基 20日均线上下 K线高亮 — Family-DCA</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0}}
header{{background:#1e293b;padding:14px 24px;border-bottom:1px solid #334155}}
header h1{{font-size:20px;font-weight:600}}
header p{{color:#94a3b8;font-size:12px;margin-top:4px}}
.tabs{{display:flex;gap:8px;flex-wrap:wrap;padding:12px 24px;border-bottom:1px solid #1e293b}}
.tab{{background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:8px 14px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600}}
.tab:hover{{color:#e2e8f0}}
.tab.active{{background:#1d4ed8;border-color:#2563eb;color:#fff}}
.tab .tag{{font-size:11px;font-weight:400;margin-left:6px}}
.above{{color:#60a5fa}}.below{{color:#f87171}}
.summary{{display:flex;gap:20px;flex-wrap:wrap;padding:12px 24px;font-size:13px}}
.summary .item b{{margin-left:4px}}
.legend{{display:flex;gap:16px;padding:0 24px 12px;font-size:12px;color:#94a3b8}}
.legend i{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-2px}}
.rules{{display:flex;gap:14px;flex-wrap:wrap;padding:10px 24px;font-size:12px;color:#cbd5e1;background:#1e293b;border-bottom:1px solid #334155}}
.rules b{{color:#e2e8f0;margin-right:4px}}
.rules span{{color:#94a3b8}}
.container{{max-width:1600px;margin:0 auto;padding:8px 24px 24px}}
#chart{{width:100%;height:680px}}
select,button{{background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:6px 12px;border-radius:4px;font-size:12px}}
</style>
</head>
<body>
<header>
  <h1>主要宽基 · 20日均线上下 K线高亮 · 顶底判断</h1>
  <p>高于 20 日均线 → <span style="color:#60a5fa">蓝色</span>；低于 20 日均线 → <span style="color:#f87171">红色</span>。副图为收盘价相对 20 日均线的偏离幅度(%)。</p>
</header>
<div class="rules" id="rules">
  <b>顶底判断方法</b>（均为成熟方法）
  <span>① 连续区间: 连续高于(蓝)/低于(红)20日线为一段"连续上涨/连续下跌"</span>
  <span>② 区间极值: 每段连续上涨的最高点 = <b style="color:#f59e0b">区间顶</b>; 每段连续下跌的最低点 = <b style="color:#22d3ee">区间底</b></span>
  <span>③ TD Sequential(DeMark): 连续收盘高于/低于4天前达到 9/13 → <b style="color:#f472b6">卖出9顶</b> / <b style="color:#4ade80">买入9底</b> (趋势衰竭)</span>
  <span>④ 辅助: 摆动高低点(分形) 与 20日线斜率拐头 可作确认, 此处用①②③主信号标绘</span>
</div>
<div class="tabs" id="tabs"></div>
<div class="summary" id="summary"></div>
<div class="legend">
  <span><i style="background:#3b82f6"></i>高于20日线</span>
  <span><i style="background:#ef4444"></i>低于20日线</span>
  <span><i style="background:#fbbf24;border-radius:50%"></i>20日均线</span>
  <span><i style="background:#f59e0b;border-radius:50%;transform:rotate(180deg)"></i>区间顶</span>
  <span><i style="background:#22d3ee;border-radius:50%"></i>区间底</span>
  <span><i style="background:#f472b6;border-radius:50%"></i>TD卖出9(顶)</span>
  <span><i style="background:#4ade80;border-radius:50%"></i>TD买入9(底)</span>
</div>
<div class="container"><div id="chart"></div></div>
<script>
const INDICES = {payload};
</script>
<script>
{echarts_src}
</script>
<script>
const chart = echarts.init(document.getElementById('chart'));
const tabsEl = document.getElementById('tabs');
const sumEl = document.getElementById('summary');

const ABOVE_STYLE = {{color:'#3b82f6', color0:'#60a5fa', borderColor:'#2563eb', borderColor0:'#93c5fd'}};
const BELOW_STYLE = {{color:'#ef4444', color0:'#f87171', borderColor:'#b91c1c', borderColor0:'#fca5a5'}};
const NA_STYLE = {{color:'#94a3b8', color0:'#94a3b8', borderColor:'#64748b', borderColor0:'#64748b'}};
const STYLES = {{a: ABOVE_STYLE, b: BELOW_STYLE, n: NA_STYLE}};

function candleData(idx) {{
  return idx.k.map(it => ({{value: it.slice(0, 4), itemStyle: STYLES[it[4]]}}));
}}

function buildOption(idx) {{
  const k = candleData(idx);
  const start = Math.max(0, 100 - Math.min(idx.k.length, 450) / idx.k.length * 100);
  return {{
    backgroundColor: 'transparent',
    animation: false,
    legend: {{data:['K线','MA20','区间顶','区间底','TD卖出9','TD买入9'], top:0, textStyle:{{color:'#94a3b8',fontSize:11}}, itemWidth:14, itemHeight:8}},
    tooltip: {{
      trigger: 'axis', axisPointer: {{type:'cross'}},
      backgroundColor: '#1e293b', borderColor: '#334155', textStyle: {{color:'#e2e8f0', fontSize:12}},
      formatter: function(ps) {{
        if (!ps.length) return '';
        const i = ps[0].dataIndex;
        const it = idx.k[i];
        const m = idx.ma20[i];
        const d = idx.dev[i];
        const rl = idx.run[i];
        let s = '<b>' + idx.name + ' ' + idx.dates[i] + '</b><br>';
        s += '开 ' + it[0] + '　收 ' + it[1] + '　低 ' + it[2] + '　高 ' + it[3] + '<br>';
        s += 'MA20 ' + (m != null ? m : '-') + '<br>';
        s += (it[4] !== 'n' ? '连续' + (it[4] === 'a' ? '上涨' : '下跌') + ' ' + rl + ' 天<br>' : '');
        s += (d != null ? '偏离 ' + d + '%' : '') + (d != null ? ' <span style="color:' + (d >= 0 ? '#60a5fa' : '#f87171') + '">' + (d >= 0 ? '▲' : '▼') + '</span>' : '');
        return s;
      }}
    }},
    axisPointer: {{link: {{xAxisIndex: 'all'}}}},
    grid: [
      {{left: 70, right: 24, top: 32, height: '58%'}},
      {{left: 70, right: 24, top: '72%', height: '20%'}}
    ],
    xAxis: [
      {{type: 'category', data: idx.dates, boundaryGap: true, axisLine: {{lineStyle: {{color: '#334155'}}}}, axisLabel: {{color: '#64748b'}}}},
      {{type: 'category', gridIndex: 1, data: idx.dates, boundaryGap: true, axisLine: {{lineStyle: {{color: '#334155'}}}}, axisLabel: {{show: false}}, axisTick: {{show: false}}}}
    ],
    yAxis: [
      {{scale: true, splitLine: {{lineStyle: {{color: '#1e293b'}}}}, axisLabel: {{color: '#64748b'}}}},
      {{gridIndex: 1, scale: true, splitLine: {{show: false}}, axisLabel: {{show: false}}, axisLine: {{show: false}}, axisTick: {{show: false}}}}
    ],
    dataZoom: [
      {{type: 'inside', xAxisIndex: [0, 1], start: start, end: 100}},
      {{type: 'slider', xAxisIndex: [0, 1], start: start, end: 100, height: 20, bottom: 4, borderColor: '#334155', backgroundColor: '#0f172a', fillerColor: 'rgba(37,99,235,0.25)', textStyle: {{color: '#94a3b8'}}}}
    ],
    series: [
      {{
        name: 'K线', type: 'candlestick', data: k, z: 3,
        itemStyle: {{color: '#3b82f6', color0: '#60a5fa', borderColor: '#2563eb', borderColor0: '#93c5fd'}}
      }},
      {{
        name: 'MA20', type: 'line', data: idx.ma20, symbol: 'none', smooth: false,
        lineStyle: {{color: '#fbbf24', width: 1.5}}, z: 5, connectNulls: false
      }},
      {{
        name: '区间顶', type: 'scatter', data: idx.tops, symbol: 'triangle', symbolRotate: 180,
        symbolSize: 10, symbolOffset: [0, -10], z: 7,
        itemStyle: {{color: '#f59e0b'}},
        tooltip: {{formatter: function(p) {{ return '<b>' + idx.name + ' 区间顶</b><br>' + p.value[0] + '<br>最高 ' + p.value[1]; }}}}
      }},
      {{
        name: '区间底', type: 'scatter', data: idx.bottoms, symbol: 'triangle',
        symbolSize: 10, symbolOffset: [0, 10], z: 7,
        itemStyle: {{color: '#22d3ee'}},
        tooltip: {{formatter: function(p) {{ return '<b>' + idx.name + ' 区间底</b><br>' + p.value[0] + '<br>最低 ' + p.value[1]; }}}}
      }},
      {{
        name: 'TD卖出9', type: 'scatter', data: idx.td9_top, symbol: 'circle',
        symbolSize: 7, symbolOffset: [0, -12], z: 8,
        itemStyle: {{color: '#f472b6'}},
        label: {{show: true, position: 'top', formatter: '9卖', color: '#f472b6', fontSize: 10, fontWeight: 700}},
        tooltip: {{formatter: function(p) {{ return '<b>' + idx.name + ' TD卖出9(顶)</b><br>' + p.value[0] + '<br>最高 ' + p.value[1] + '<br>趋势衰竭, 注意顶部风险'; }}}}
      }},
      {{
        name: 'TD买入9', type: 'scatter', data: idx.td9_bot, symbol: 'circle',
        symbolSize: 7, symbolOffset: [0, 12], z: 8,
        itemStyle: {{color: '#4ade80'}},
        label: {{show: true, position: 'bottom', formatter: '9买', color: '#4ade80', fontSize: 10, fontWeight: 700}},
        tooltip: {{formatter: function(p) {{ return '<b>' + idx.name + ' TD买入9(底)</b><br>' + p.value[0] + '<br>最低 ' + p.value[1] + '<br>趋势衰竭, 留意底部机会'; }}}}
      }},
      {{
        name: '偏离', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: idx.dev,
        itemStyle: {{
          color: function(p) {{ return p.value >= 0 ? 'rgba(59,130,246,0.55)' : 'rgba(239,68,68,0.55)'; }}
        }}
      }}
    ]
  }};
}}

function renderSummary(idx) {{
  const c = idx.cur;
  const cls = c.status === 'a' ? 'above' : (c.status === 'b' ? 'below' : '');
  const txt = c.status === 'a' ? '高于20日线' : (c.status === 'b' ? '低于20日线' : '-');
  const runTxt = c.status === 'a' ? ('连续上涨 <b>' + c.run_days + ' 天</b> (自 ' + c.run_start + ')')
    : (c.status === 'b' ? ('连续下跌 <b>' + c.run_days + ' 天</b> (自 ' + c.run_start + ')') : '-');
  let html = '<span>最新 ' + c.date + '　收盘 <b>' + c.close + '</b></span>';
  html += '<span>MA20 <b>' + (c.ma20 != null ? c.ma20 : '-') + '</b></span>';
  html += '<span class="' + cls + '">状态 <b>' + txt + '</b></span>';
  html += '<span>偏离 <b class="' + cls + '">' + (c.dev != null ? c.dev + '%' : '-') + '</b></span>';
  html += '<span>' + runTxt + '</span>';
  html += '<span>最近区间顶 <b style="color:#f59e0b">' + (c.recent_top ? c.recent_top[0] + ' @ ' + c.recent_top[1] : '-') + '</b></span>';
  html += '<span>最近区间底 <b style="color:#22d3ee">' + (c.recent_bottom ? c.recent_bottom[0] + ' @ ' + c.recent_bottom[1] : '-') + '</b></span>';
  sumEl.innerHTML = html;
}}

function renderTabs() {{
  INDICES.forEach((idx, i) => {{
    const b = document.createElement('button');
    b.className = 'tab' + (i === 0 ? ' active' : '');
    b.innerHTML = idx.name + '<span class="tag ' + (idx.cur.status === 'a' ? 'above' : 'below') + '">' + (idx.cur.status === 'a' ? '▲' : '▼') + '</span>';
    b.onclick = () => {{
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      chart.setOption(buildOption(idx), true);
      renderSummary(idx);
    }};
    tabsEl.appendChild(b);
  }});
}}

renderTabs();
chart.setOption(buildOption(INDICES[0]), true);
renderSummary(INDICES[0]);
window.addEventListener('resize', () => chart.resize());
</script>
</body>
</html>
"""
    with open(OUT, "w") as f:
        f.write(html)
    print(f"OK -> {OUT} ({len(html)/1024:.0f} KB, {len(indices)} 个指数)")
    for idx in indices:
        c = idx["cur"]
        print(f"  {idx['name']:<8} {c['date']} close={c['close']} ma20={c['ma20']} dev={c['dev']}% status={c['status']}")


if __name__ == "__main__":
    main()
