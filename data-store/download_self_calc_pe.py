#!/usr/bin/env python3
"""
自算 PE/PB 下载器（方式2：丹卷风格计算）

从个股 PE/PB 出发，应用丹卷规则（排除负PE、截尾极端值），
计算等权 PE/PB。输出到 parquet/self_calc/。

算法：
  1. 获取指数成分股列表
  2. 获取每只成分股的 PE(TTM) 和 PB
  3. 剔除 PE ≤ 0（亏损公司）
  4. PE > 200 截断为 200
  5. top/bottom 5% 截尾（去除极端值影响）
  6. 等权 PE = 剩余PE的调和平均数
  7. 等权 PB = 剩余PB的调和平均数

覆盖范围: 上证50、沪深300、中证500、中证1000、科创50、创业板指 (6指数)

数据源:
  - 成分股: 新浪财经 Market_Center API
  - 个股市值: 东方财富 push2 API

用法:
    python download_self_calc_pe.py                  # 下载全部
    python download_self_calc_pe.py --check           # 验证接口
    python download_self_calc_pe.py --list            # 列出指数
    python download_self_calc_pe.py --code 000300     # 下载指定指数
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from statistics import harmonic_mean

import numpy as np
import pandas as pd
import requests

# ============================================================================
# 指数配置
# ============================================================================

SELF_CALC_INDEXES = [
    {"code": "000016", "name": "上证50",   "sina_node": "sh000016", "constituent_ok": False},
    {"code": "000300", "name": "沪深300",  "sina_node": "hs300",    "constituent_ok": True},
    {"code": "000905", "name": "中证500",  "sina_node": "sz000905", "constituent_ok": False},
    {"code": "000852", "name": "中证1000", "sina_node": "sz000852", "constituent_ok": False},
    {"code": "000688", "name": "科创50",   "sina_node": "sh000688", "constituent_ok": False},
    {"code": "399006", "name": "创业板指", "sina_node": "sz399006", "constituent_ok": False},
]

SINA_MARKET_CENTER = (
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
    "/Market_Center.getHQNodeData"
)
PUSH2_CLIST = "https://push2.eastmoney.com/api/qt/clist/get"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# 丹卷风格过滤参数
PE_CAP_MAX = 200        # PE 上限截断
TRIM_PCT = 0.05         # 首尾截尾比例

# ============================================================================
# 成分股获取
# ============================================================================

def _fetch_constituents_via_push2(index_code: str) -> dict:
    """
    通过东方财富 push2 API 获取所有 A 股 PE/PB 数据。

    注意：此函数获取所有 A 股数据（非过滤特定指数），
    成分股过滤需要在获取后由调用方完成。
    目前保留作为备选方案，需配合成分股列表使用。
    """
    return {}  # 独立使用时没有成分股列表过滤，返回空让上层回退


def _fetch_constituents_via_sina(sina_node: str) -> dict:
    """
    通过新浪财经 Market_Center API 获取指数成分股 PE/PB。
    
    可用的 node 值（已确认）:
      hs300 = 沪深300
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    stock_data = {}
    # 已知可用的新浪 node 名称
    known_nodes = {
        "hs300": "沪深300",
    }

    # 尝试多个可能的 node 名称
    nodes_to_try = [sina_node]
    if sina_node == "000016":
        nodes_to_try = ["sh000016", "sz50", "sh50"]
    elif sina_node == "000905":
        nodes_to_try = ["sz000905", "zz500", "zh500"]
    elif sina_node == "000852":
        nodes_to_try = ["sz000852", "zz1000", "zh1000"]
    elif sina_node == "000688":
        nodes_to_try = ["sh000688", "kcb50", "star50"]
    elif sina_node == "399006":
        nodes_to_try = ["sz399006", "cyb50", "chnext"]
    elif sina_node == "hs300":
        nodes_to_try = ["hs300"]

    for node in nodes_to_try:
        all_items = []
        page = 1
        max_page = 20

        while page <= max_page:
            params = {
                "page": str(page),
                "num": "100",
                "sort": "symbol",
                "asc": "1",
                "node": node,
            }
            try:
                r = session.get(SINA_MARKET_CENTER, params=params, timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    break
                data = json.loads(r.text)
            except Exception:
                break

            if not isinstance(data, list) or len(data) == 0:
                break

            all_items.extend(data)
            page += 1
            time.sleep(0.15)

        if len(all_items) < 5:
            continue

        for item in all_items:
            code = item.get("code", "")
            pe = item.get("per")
            pb = item.get("pb")
            name = item.get("name", "")
            mcap = item.get("mktcap")
            pe_val = None
            if pe and pe != "--":
                try:
                    pe_val = float(pe)
                except (ValueError, TypeError):
                    pass
            pb_val = None
            if pb and pb != "--":
                try:
                    pb_val = float(pb)
                except (ValueError, TypeError):
                    pass
            if code and (pe_val is not None or pb_val is not None):
                stock_data[code] = {
                    "name": name,
                    "pe": pe_val,
                    "pb": pb_val,
                    "mcap": mcap,
                }
        break  # 用第一个可用的 node，不继续尝试

    return stock_data


def _fetch_constituents_via_sina_page(index_code: str) -> dict:
    """
    通过新浪 vII_NewestComponent 页面获取指数成分股（备选方案）。

    注意：此页面仅返回最新的 ~40-100 条记录，非全量成分股。
    仅在其他方案不可用时作为近似替代。
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    url = (
        "https://vip.stock.finance.sina.com.cn/corp/go.php/"
        f"vII_NewestComponent/indexid/{index_code}.phtml"
    )
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return {}
    except Exception:
        return {}

    import re
    # 提取页面中的股票代码链接
    codes = re.findall(r'>(\d{6})<', r.text)
    stock_codes = list(set(c for c in codes if c != index_code and c[0] in "036" and len(c) == 6))

    if len(stock_codes) < 5:
        return {}

    # 通过 push2 批量获取这些股票的 PE/PB
    return _batch_fetch_stock_data(stock_codes)


def _batch_fetch_stock_data(stock_codes: list) -> dict:
    """通过 push2 API 批量获取指定股票代码的 PE/PB。"""
    if not stock_codes:
        return {}

    session = requests.Session()
    session.verify = False
    session.headers.update({
        **HEADERS,
        "Referer": "https://quote.eastmoney.com/",
    })

    # 构建 push2 的 secid 列表
    secids = []
    for code in stock_codes:
        if code.startswith("6") or code.startswith("5"):
            secids.append(f"1.{code}")
        else:
            secids.append(f"0.{code}")

    stock_data = {}

    # 一次最多查询 50 只
    for i in range(0, len(secids), 50):
        batch = secids[i:i+50]
        params = {
            "pn": "1",
            "pz": "100",
            "po": "0",
            "np": "1",
            "fltt": "2",
            "invt": "2",
            "fs": ",".join(batch),
            "fields": "f2,f9,f12,f14,f20,f23",
        }
        try:
            r = session.get(PUSH2_CLIST, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception:
            continue

        for item in (data.get("data") or {}).get("diff") or []:
            code = item.get("f12", "")
            pe = item.get("f9")
            pb = item.get("f23")
            name = item.get("f14", "")
            mcap = item.get("f20")

            pe_val = None
            if pe and pe != "-":
                try:
                    pe_val = float(pe)
                except (ValueError, TypeError):
                    pass
            pb_val = None
            if pb and pb != "-":
                try:
                    pb_val = float(pb)
                except (ValueError, TypeError):
                    pass

            if code:
                stock_data[code] = {
                    "name": name,
                    "pe": pe_val,
                    "pb": pb_val,
                    "mcap": mcap,
                }
        time.sleep(0.2)

    return stock_data


def fetch_constituents(index_info: dict) -> dict:
    """
    多级回退获取指数成分股 PE/PB 数据。
    
    返回: {code: {name, pe, pb, mcap}, ...}
    """
    index_code = index_info["code"]
    sina_node = index_info.get("sina_node", "")

    # 方案 1: 新浪 Market_Center API（最可靠，含 PE/PB）
    stock_data = _fetch_constituents_via_sina(sina_node)
    if len(stock_data) >= 10:
        return stock_data

    # 方案 2: 新浪 vII_NewestComponent 页面 + push2 批量查询
    stock_data = _fetch_constituents_via_sina_page(index_code)
    if len(stock_data) >= 10:
        return stock_data

    # 方案 3: push2 全市场扫描（耗时，但最完整）
    stock_data = _fetch_constituents_via_push2(index_code)
    return stock_data


# ============================================================================
# 丹卷风格计算
# ============================================================================

def calc_danjuan_style(stock_data: dict) -> dict:
    """
    应用丹卷规则计算等权 PE/PB。

    规则:
      1. 剔除 PE ≤ 0（亏损公司）
      2. PE > PE_CAP_MAX 截断为 PE_CAP_MAX
      3. top/bottom TRIM_PCT 截尾
      4. 等权 PE = 调和平均数
      5. 等权 PB = 调和平均数
    """
    result = {
        "stock_count": len(stock_data),
        "pe_self_calc": None,
        "pb_self_calc": None,
        "pe_used": 0,
        "pb_used": 0,
        "pe_excluded_neg": 0,
        "pe_excluded_trim": 0,
    }

    # 收集有效 PE 值
    pe_vals = []
    for code, info in stock_data.items():
        pe = info.get("pe")
        if pe is not None and pe > 0:
            if pe > PE_CAP_MAX:
                pe_vals.append(PE_CAP_MAX)
            else:
                pe_vals.append(pe)
        elif pe is not None and pe <= 0:
            result["pe_excluded_neg"] += 1

    # 截尾
    trim_count = 0
    if len(pe_vals) > 20:
        pe_sorted = sorted(pe_vals)
        n = len(pe_sorted)
        lo = int(n * TRIM_PCT)
        hi = int(n * (1 - TRIM_PCT))
        trim_count = lo + (n - hi)
        pe_vals = pe_sorted[lo:hi]

    result["pe_excluded_trim"] = trim_count
    result["pe_used"] = len(pe_vals)

    # 调和平均
    if len(pe_vals) > 0:
        try:
            result["pe_self_calc"] = harmonic_mean(pe_vals)
        except Exception:
            result["pe_self_calc"] = sum(pe_vals) / len(pe_vals)

    # PB：收集有效 PB 值
    pb_vals = []
    for code, info in stock_data.items():
        pb = info.get("pb")
        if pb is not None and pb > 0:
            pb_vals.append(pb)

    result["pb_used"] = len(pb_vals)
    if len(pb_vals) > 0:
        try:
            result["pb_self_calc"] = harmonic_mean(pb_vals)
        except Exception:
            result["pb_self_calc"] = sum(pb_vals) / len(pb_vals)

    return result


# ============================================================================
# 下载流程
# ============================================================================

def _save_result(result_df, calc_result, index_info, output_dir):
    """保存计算结果为 Parquet。"""
    if result_df.empty:
        return

    summary = pd.DataFrame([{
        "code": index_info["code"],
        "name": index_info["name"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "stock_count": calc_result["stock_count"],
        "pe_used": calc_result["pe_used"],
        "pb_used": calc_result["pb_used"],
        "pe_excluded_neg": calc_result["pe_excluded_neg"],
        "pe_excluded_trim": calc_result["pe_excluded_trim"],
    }])

    detail_dir = os.path.join(output_dir, "detail")
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(detail_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    result_df.to_parquet(
        os.path.join(detail_dir, f"{index_info['code']}_detail.parquet"),
        index=False,
    )
    summary.to_parquet(
        os.path.join(summary_dir, f"{index_info['code']}_summary.parquet"),
        index=False,
    )


def download_index(index_info: dict, output_dir: str) -> bool:
    """下载并计算单个指数的丹卷风格 PE/PB。"""
    code = index_info["code"]
    name = index_info["name"]

    if not index_info.get("constituent_ok", False):
        print(f"[SKIP] {code} {name} — 无可用成分股数据源，目前仅沪深300支持自算")
        return False

    print(f"[FETCH] {code} {name} 成分股 ...", end=" ", flush=True)
    stock_data = fetch_constituents(index_info)

    if len(stock_data) < 5:
        print(f"[FAIL] 仅获取到 {len(stock_data)} 只成分股，跳过")
        return False

    print(f"获取 {len(stock_data)} 只", end=" ", flush=True)

    # 统计 PE/PB 覆盖率
    pe_count = sum(1 for v in stock_data.values() if v.get("pe") is not None)
    pb_count = sum(1 for v in stock_data.values() if v.get("pb") is not None)
    print(f"(PE覆盖{pe_count}, PB覆盖{pb_count}) ...", end=" ", flush=True)

    calc_result = calc_danjuan_style(stock_data)

    pe_val = calc_result["pe_self_calc"]
    pb_val = calc_result["pb_self_calc"]
    pe_str = f"{pe_val:.4f}" if pe_val else "N/A"
    pb_str = f"{pb_val:.4f}" if pb_val else "N/A"

    print(f"→ PE_self={pe_str}, PB_self={pb_str} "
          f"| 有效PE={calc_result['pe_used']}, "
          f"排除负PE={calc_result['pe_excluded_neg']}, "
          f"截尾={calc_result['pe_excluded_trim']}")

    # 保存明细
    rows = []
    for code_s, info in stock_data.items():
        rows.append({
            "stock_code": code_s,
            "stock_name": info["name"],
            "pe": info.get("pe"),
            "pb": info.get("pb"),
            "mcap": info.get("mcap"),
        })
    detail_df = pd.DataFrame(rows)
    _save_result(detail_df, calc_result, index_info, output_dir)

    return True


def download_all(output_dir: str):
    """下载全部指数。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n自算丹卷风格 PE/PB 下载 — {now}")
    print(f"输出目录: {output_dir}")
    print(f"丹卷规则: PE≤0淘汰, PE>{PE_CAP_MAX}截断, 首尾{TRIM_PCT*100:.0f}%截尾, 调和平均\n")

    success = 0
    for idx in SELF_CALC_INDEXES:
        if download_index(idx, output_dir):
            success += 1
        print()
        time.sleep(REQUEST_DELAY)

    print(f"{'='*50}")
    print(f"结果: 成功 {success}/{len(SELF_CALC_INDEXES)}")
    print(f"输出目录: {output_dir}")


def check():
    """验证接口连通性。"""
    print("自算丹卷 PE/PB 接口检查\n")
    idx = SELF_CALC_INDEXES[1]  # 沪深300
    print(f"测试指数: {idx['name']} ({idx['code']})")

    stock_data = fetch_constituents(idx)
    if len(stock_data) < 5:
        print("[FAIL] 无法获取成分股数据")
        sys.exit(1)

    print(f"[OK] 获取到 {len(stock_data)} 只成分股")
    pe_count = sum(1 for v in stock_data.values() if v.get("pe"))
    pb_count = sum(1 for v in stock_data.values() if v.get("pb"))
    print(f"  PE覆盖: {pe_count}/{len(stock_data)}")
    print(f"  PB覆盖: {pb_count}/{len(stock_data)}")

    calc_result = calc_danjuan_style(stock_data)
    print(f"  丹卷PE: {calc_result['pe_self_calc']:.4f}" if calc_result["pe_self_calc"] else "  丹卷PE: N/A")
    print(f"  丹卷PB: {calc_result['pb_self_calc']:.4f}" if calc_result["pb_self_calc"] else "  丹卷PB: N/A")
    print(f"  有效PE数: {calc_result['pe_used']}, 排除负PE: {calc_result['pe_excluded_neg']}, 截尾: {calc_result['pe_excluded_trim']}")


def list_indices():
    """列出所有指数。"""
    print(f"{'指数代码':<8} {'指数名称':<10} {'新浪node':<15}")
    print("-" * 35)
    for idx in SELF_CALC_INDEXES:
        print(f"{idx['code']:<8} {idx['name']:<10} {idx.get('sina_node', '-'):<15}")
    print(f"\n共 {len(SELF_CALC_INDEXES)} 个指数")


# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="自算丹卷风格 PE/PB 下载器"
    )
    parser.add_argument("--check", action="store_true", help="验证接口连通性")
    parser.add_argument("--list", action="store_true", help="列出所有指数")
    parser.add_argument("--code", type=str, default=None, help="下载指定指数代码")
    parser.add_argument("--output", type=str, default=None,
                        help="输出根目录（默认 data-store/parquet/self_calc/）")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(
        PROJECT_DIR, "data-store", "parquet", "self_calc"
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.check:
        check()
    elif args.list:
        list_indices()
    elif args.code:
        matched = [i for i in SELF_CALC_INDEXES if i["code"] == args.code]
        if not matched:
            print(f"[ERROR] 未知指数代码: {args.code}", file=sys.stderr)
            print("使用 --list 查看可用指数代码", file=sys.stderr)
            sys.exit(1)
        download_index(matched[0], output_dir)
    else:
        download_all(output_dir)


if __name__ == "__main__":
    main()
