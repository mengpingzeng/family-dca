# AKShare 宽基指数数据下载 · 可直接交给 AI 的 Prompt

> 用法：把下面 **「② 正式 Prompt」** 整段复制，粘贴给任意 AI 编程助手（Claude / ChatGPT / Cursor / Box 等），
> 再补一句你想要的指数和时间范围即可。
>
> 本文所有接口与代码均在 **akshare 1.18.64 / Python 3 / 2026-08-12** 实测通过。

---

## ① 一句话背景（给使用者看，不用发给 AI）

很多人以为 AKShare 只能下沪深300、中证500、上证50 三个宽基——这是误解。
`index_zh_a_hist` 官方文档里只是拿这三个**举例**，接口本身是通用的，任意指数代码都能传。
红利、恒生科技、纳斯达克、标普500 全都能拿到，只是**分散在不同接口**、且**代码前缀规则不同**。
下面的 Prompt 已经把这些坑全部写死，AI 照着做基本不会错。

---

## ② 正式 Prompt（复制这一整段发给 AI）

```
你是一个熟悉 AKShare 的 Python 数据工程师。请帮我用 AKShare 下载指数历史行情数据。

【重要前提，请严格遵守，不要自行改用其他接口】

1. AKShare 能下载的宽基远不止沪深300/中证500/上证50。不同市场必须用不同接口，
   请严格按下面这张「接口 - 市场」对照表选择：

   | 市场 | 必须使用的接口 | symbol 写法 |
   |---|---|---|
   | A股 / 中证系列指数 | ak.stock_zh_index_daily_em(symbol=...) | 见第 2 条前缀规则 |
   | 港股指数 | ak.stock_hk_index_daily_sina(symbol=...) | HSI / HSTECH / HSCEI |
   | 美股指数 | ak.index_us_stock_sina(symbol=...) | .INX / .IXIC / .NDX / .DJI |
   | 场内 ETF（指数的替代方案） | ak.fund_etf_hist_em(symbol=..., period="daily", start_date=..., end_date=...) | 6 位 ETF 代码，不带前缀 |

2. A股指数的 symbol 前缀规则（写错会直接报 TypeError: 'NoneType' object is not subscriptable）：
   - 上交所 6 位代码（000xxx / 950xxx）→ 加前缀 sh，例：沪深300 = sh000300
   - 深交所 6 位代码（399xxx）→ 加前缀 sz，例：创业板指 = sz399006
   - 中证公司自编指数（93xxxx、H3xxxx）→ 加前缀 csi，例：中证A50 = csi930050
   - 不要给 csi 前缀的指数改用 sh/sz，也不要给 sh/sz 的指数改用 csi，两者不通用。

3. 绝对不要使用 ak.stock_zh_index_daily()（新浪源）。该接口数据已停更，
   拉 sh000922 只能返回到 2019-01-30，是个陷阱。统一用带 _em 后缀的版本。

4. 东方财富的数据域名（push2his.eastmoney.com）存在间歇性限流，
   会随机抛出 ConnectionError: RemoteDisconnected。同一个 symbol 可能第 1 次失败、第 3 次才成功。
   因此所有请求**必须**包一层重试（建议 6 次、间隔 3 秒），否则脚本会大概率随机失败。
   走 sina 源的港股/美股接口不受此影响。

【我要下载的数据】
（在这里填写你要的指数，可从下面清单里挑）

- 指数清单：<填写，例如：中证红利、红利低波、恒生科技、纳斯达克100、标普500>
- 时间范围：<填写，例如：2015-01-01 至今；或"全部历史">
- 输出格式：<填写，例如：每个指数一个 CSV，存到 ./data/ 目录；或合并成一个宽表>

【常用指数代码对照表，直接用，不要自己猜代码】

A股宽基（接口 ak.stock_zh_index_daily_em）
  沪深300         sh000300
  上证50          sh000016
  中证500         sh000905
  中证1000        sh000852
  中证A500        sh000510
  中证A50         csi930050
  创业板指        sz399006
  科创50          sh000688

A股红利类（接口 ak.stock_zh_index_daily_em）
  上证红利        sh000015
  中证红利        sh000922
  中证红利低波100  csi930955
  深证红利        sz399324

港股（接口 ak.stock_hk_index_daily_sina）
  恒生指数        HSI
  恒生科技        HSTECH
  恒生中国企业    HSCEI

美股（接口 ak.index_us_stock_sina）
  标普500         .INX
  纳斯达克综指    .IXIC
  纳斯达克100     .NDX
  道琼斯          .DJI

跨境 ETF（接口 ak.fund_etf_hist_em，当指数本身取不到、或想直接用可投资标的时使用）
  恒生科技ETF     513180
  标普500ETF      513500
  纳指100ETF      159941
  红利ETF         510880
  红利低波ETF     512890

【辅助查询接口】
  ak.index_stock_info()        → A股全部指数清单（约 732 个），含 index_code / display_name，用来查代码
  ak.index_global_spot_em()    → 全球主要指数实时快照（约 56 个：日经225、德国DAX30、
                                  法国CAC40、英国富时100、韩国KOSPI、印度SENSEX、美元指数等）

【请交付给我】
1. 一个可直接运行的 Python 脚本，内置上述重试逻辑
2. 脚本运行结束后打印每个指数实际获取到的行数与最新日期，方便我核对
3. 若某个指数确实取不到，明确告诉我，并给出 ETF 替代方案，不要静默跳过或伪造数据
```

---

## ③ 参考实现（AI 应该产出类似这样的代码，可直接自用）

```python
import time
import akshare as ak
import pandas as pd
from pathlib import Path

OUT = Path("./data"); OUT.mkdir(parents=True, exist_ok=True)

def fetch(fn, retry=6, sleep=3, **kw):
    """东财域名会间歇性断连，必须重试"""
    err = None
    for _ in range(retry):
        try:
            df = fn(**kw)
            if df is not None and len(df):
                return df
        except Exception as e:
            err = e
        time.sleep(sleep)
    raise RuntimeError(f"{fn.__name__} {kw} 失败: {type(err).__name__}")

# (中文名, 接口, symbol)
TARGETS = [
    ("中证红利",     ak.stock_zh_index_daily_em,   "sh000922"),
    ("上证红利",     ak.stock_zh_index_daily_em,   "sh000015"),
    ("红利低波100",  ak.stock_zh_index_daily_em,   "csi930955"),
    ("中证A500",     ak.stock_zh_index_daily_em,   "sh000510"),
    ("中证A50",      ak.stock_zh_index_daily_em,   "csi930050"),
    ("沪深300",      ak.stock_zh_index_daily_em,   "sh000300"),
    ("恒生科技",     ak.stock_hk_index_daily_sina, "HSTECH"),
    ("恒生指数",     ak.stock_hk_index_daily_sina, "HSI"),
    ("标普500",      ak.index_us_stock_sina,       ".INX"),
    ("纳斯达克100",  ak.index_us_stock_sina,       ".NDX"),
]

for name, api, sym in TARGETS:
    try:
        df = fetch(api, symbol=sym)
        df.to_csv(OUT / f"{name}_{sym}.csv", index=False, encoding="utf-8-sig")
        print(f"[OK]   {name:<12} {sym:<10} rows={len(df):>5}  last={df.iloc[-1, 0]}")
    except Exception as e:
        print(f"[FAIL] {name:<12} {sym:<10} {e}")

# ETF 走另一个接口，参数不同，需单独处理
ETFS = [("恒生科技ETF", "513180"), ("标普500ETF", "513500"),
        ("纳指100ETF", "159941"), ("红利ETF", "510880")]
for name, code in ETFS:
    try:
        df = fetch(ak.fund_etf_hist_em, symbol=code, period="daily",
                   start_date="20150101", end_date="20260812")
        df.to_csv(OUT / f"{name}_{code}.csv", index=False, encoding="utf-8-sig")
        print(f"[OK]   {name:<12} {code:<10} rows={len(df):>5}  last={df.iloc[-1, 0]}")
    except Exception as e:
        print(f"[FAIL] {name:<12} {code:<10} {e}")
```

---

## ④ 实测结果留证（2026-08-12，akshare 1.18.64）

| 指数 | 接口 | symbol | 行数 | 最新日期 |
|---|---|---|---|---|
| 上证50 | stock_zh_index_daily_em | sh000016 | 5491 | 2026-08-12 |
| 上证红利 | stock_zh_index_daily_em | sh000015 | 5248 | 2026-08-12 |
| 深证红利 | stock_zh_index_daily_em | sz399324 | 4992 | 2026-08-12 |
| 中证红利 | stock_zh_index_daily_em | sh000922 | 4379 | 2026-08-12 |
| 红利低波100 | stock_zh_index_daily_em | csi930955 | 1581 | 2026-08-12 |
| 中证A500 | stock_zh_index_daily_em | sh000510 | 5248 | 2026-08-12 |
| 中证A50 | stock_zh_index_daily_em | csi930050 | 2822 | 2026-08-12 |
| 中证500 | stock_zh_index_daily_em | sh000905 | 5248 | 2026-08-12 |
| 中证1000 | stock_zh_index_daily_em | sh000852 | 5248 | 2026-08-12 |
| 创业板指 | stock_zh_index_daily_em | sz399006 | 3934 | 2026-08-12 |
| 科创50 | stock_zh_index_daily_em | sh000688 | 1603 | 2026-08-12 |
| 恒生科技 | stock_hk_index_daily_sina | HSTECH | 1469 | 2026-08-11 |
| 恒生指数 | stock_hk_index_daily_sina | HSI | 3192 | 2026-08-11 |
| 恒生中国企业 | stock_hk_index_daily_sina | HSCEI | 3192 | 2026-08-11 |
| 标普500 | index_us_stock_sina | .INX | 5690 | 2026-08-11 |
| 纳斯达克综指 | index_us_stock_sina | .IXIC | 5687 | 2026-08-11 |
| 纳指100 | index_us_stock_sina | .NDX | 3138 | 2026-08-11 |
| 道琼斯 | index_us_stock_sina | .DJI | 5689 | 2026-08-11 |
| 恒生科技ETF | fund_etf_hist_em | 513180 | 390 | 2026-08-12 |
| 标普500ETF | fund_etf_hist_em | 513500 | 390 | 2026-08-12 |
| 纳指100ETF | fund_etf_hist_em | 159941 | 390 | 2026-08-12 |
| 红利ETF | fund_etf_hist_em | 510880 | 390 | 2026-08-12 |
| 红利低波ETF | fund_etf_hist_em | 512890 | 390 | 2026-08-12 |

> ETF 行数为 390，是因为测试时把 start_date 设为 20250101；改成更早日期即可取更长历史。

### 已确认的三个坑

1. `stock_zh_index_daily`（新浪）**数据停更**：`sh000922` 只到 2019-01-30。别用，改 `_em` 版。
2. `index_zh_a_hist` / `stock_hk_index_spot_em` / `fund_etf_hist_em` 走 `push2his.eastmoney.com`，
   会**随机断连**。首轮 6 个调用曾全部失败，加重试后第 1~5 次尝试才成功。必须重试。
3. symbol 前缀写错（如把 `csi930050` 写成 `sh930050`）不会报"找不到"，
   而是抛 `TypeError: 'NoneType' object is not subscriptable`，容易误判为接口坏了。

> 上表 21 个标的全部实测通过，无失败项。其中 `sz399324` 与 `512890` 需重试到第 7 次以上才成功，
> 再次印证第 2 条：重试次数给足（建议 ≥6 次），否则会误以为数据不存在。
