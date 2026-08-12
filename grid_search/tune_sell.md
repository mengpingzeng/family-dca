# Buy-Only vs 有卖出 对比实验

> 2026-08-09

## 目标

验证"去掉卖出逻辑、纯定投买入"是否优于当前最优的 PB_FED 策略。

## 策略

### Buy-Only (本实验)

```
买入 = PB 百分位 (每周一次, FED 门控)
  PB% < 10%  → 3x ¥4,500
  PB% < 15%  → 2x ¥3,000
  PB% < 35%  → 1x ¥1,500
  PB% < 70%  → 0.5x ¥750
  PB% ≥ 70%  → 停止买入
卖出 = 无 (永远不卖)
FED = fed < mean - 0.5×std 时不买
```

### 有卖出 (当前最优 PB_FED)

同上买入 + 以下卖出:
```
PB% ≥ 75% → heavy sell: min((PB%-75%)×0.1, 25%)/月
PB% ≥ 90% 且回撤≥4% → extreme sell
```

## 参数

| 参数 | 值 |
|------|-----|
| 窗口 | 8yr |
| 买入参数 | floor=10%, low=15%, mid=35%, high=70%, warn=70% |
| FED | -0.5 |
| 宽基指数 | 000300, 000905, 000852, 000016, 000688, 399006, 399330 |

## 评分

| 指标 | 计算方式 |
|------|----------|
| final_xirr | 二分法 IRR (最终日) |
| simple_return | (市值-投入)/投入 |
| trades | 总交易次数 |

## 输出

- `grid_search/output/buy_only_vs_sell.csv` — 各指数对比数据
- `grid_search/output/buy_only_vs_sell.json` — JSON 供前端
- `http://47.107.124.45:8000/grid-search/buy-only` — 可视化页面
