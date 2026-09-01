#!/usr/bin/env python3
"""
每日自动推送 — 收盘后全流程:
  1. 下载蛋卷 PE/PB (周频数据, 当天可能无新值, 脚本内部处理)
  2. 下载国债收益率 (日频)
  3. 下载指数价格 (日频)
  4. 构建推送数据 (push/data/*.parquet)
  5. 判断信号并推送到企业微信群

用法: python wind_new_search/push/update_and_push.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PY = "/usr/local/python3.11/bin/python3.11"
DATA_DIR = PROJECT_DIR / "data-store"


def run(name, args):
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run([PY] + args, cwd=str(PROJECT_DIR))
    if r.returncode != 0:
        print(f"[FAIL] {name} 退出码 {r.returncode}", flush=True)
        sys.exit(1)
    return r


def main():
    # 1. 数据下载
    run("下载蛋卷PE", [str(DATA_DIR / "download_danjuan_pe.py")])
    run("下载蛋卷PB", [str(DATA_DIR / "download_danjuan_pb.py")])
    run("下载国债", [str(DATA_DIR / "download_bond_yield.py")])
    # 指数价格: 中证源(沪深中证系) + 新浪源(深市/港股美股)
    run("下载指数价格", [str(DATA_DIR / "download_index_price.py")])
    run("下载补充价格", [str(DATA_DIR / "download_index_price_extra.py")])
    run("下载港股美股价格", [str(DATA_DIR / "download_wind_price.py")])

    # 2. 构建推送数据
    run("构建推送数据", [str(PROJECT_DIR / "wind_new_search" / "push" / "build_push_data.py")])

    # 3. 推送
    run("推送信号", [str(PROJECT_DIR / "wind_new_search" / "push" / "push_daily.py")])
    print("\n✅ 全流程完成", flush=True)


if __name__ == "__main__":
    main()
