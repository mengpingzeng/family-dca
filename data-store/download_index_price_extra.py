#!/usr/bin/env python3
"""补充指数价格下载 — Sina Finance 源（A股深交所 + 港股）"""

import argparse, json, os, sys, time
from datetime import datetime
import pandas as pd, requests

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(DIR), "data-store", "parquet", "index_price")
os.makedirs(OUT, exist_ok=True)

# Sina symbol → 系统 code
TARGETS = {
    "sz399006": "399006",  # 创业板指
    "sz399330": "399330",  # 深证100
}

URL = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"

def download_one(sina_code, code, name):
    print(f"[FETCH] {code} {name} ...", end=" ", flush=True)
    try:
        r = requests.get(URL,
            params={"symbol": sina_code, "scale": 240, "ma": "no", "datalen": 10000},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        data = json.loads(r.text)
        rows = []
        for d in data:
            if d.get("close") is None:
                continue
            rows.append({
                "date": pd.to_datetime(d["day"]).date(),
                "index_price": float(d["close"]),
            })
        if not rows:
            print("无数据"); return False
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        path = os.path.join(OUT, f"{code}.parquet")
        df.to_parquet(path, index=False)
        print(f"OK | {len(df)}条 | {df.date.min()}~{df.date.max()} | "
              f"price {df.index_price.min():.1f}~{df.index_price.max():.1f} | latest {df.iloc[-1].index_price:.2f}")
        return True
    except Exception as e:
        print(f"ERROR {e}"); return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        r = requests.get(URL, params={"symbol": "sz399006", "scale": 240, "ma": "no", "datalen": 3},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        print(f"Check: {len(json.loads(r.text))} samples"); return

    print(f"\n补充价格下载 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ok = 0
    for sina, code in TARGETS.items():
        if not os.path.exists(os.path.join(OUT, f"{code}.parquet")):
            if download_one(sina, code, {"399006":"创业板指","399330":"深证100"}[code]):
                ok += 1
            time.sleep(1)
        else:
            print(f"[SKIP] {code} 已存在")
    print(f"\n结果: {ok} 个 → {OUT}/")
    print(f"⚠️ HSI/HSTECH/NDX100/SPX500 暂无海外价格源，待补充")

if __name__ == "__main__":
    main()
