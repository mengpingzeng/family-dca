# 宽基统一策略网格搜索

> 生成时间: 2026-08-09

## 目标

在 **7 个宽基指数**（沪深300、中证500、中证1000、上证50、科创50、创业板指、深证100）上，**使用同一套策略参数**进行回测，找出同时满足"高 XIRR + 低 XIRR 波动率"的最优策略。

## 遍历维度

| 维度 | 内容 | 数量 |
|------|------|------|
| 数据源 | 蛋卷基金 PE (`pe_ttm_dj`) | 1 |
| 宽基指数 | 000300, 000905, 000852, 000016, 000688, 399006, 399330 | 7 |
| 窗口年限 | 6yr, 8yr, 10yr | 3 |
| 特征组合 | PE_only, PE_PB, PE_FED, PE_PB_FED, PB_only, PB_FED | 6 |
| 权重 | (1.0, 0.0), (0.8, 0.2), (0.6, 0.4) | 3 |
| 策略采样 | 拒绝采样, 每 (特征组合,窗口) 1000 组 | ~18,000 |

## 特征组合说明

| 组合 | 主信号 | PB否决 | FED门控 | 参数空间 |
|------|--------|--------|---------|----------|
| `PE_only` | PE百分位 | ❌ | ❌ | ~60K |
| `PE_PB` | PE百分位 | ✅ | ❌ | ~242K |
| `PE_FED` | PE百分位 | ❌ | ✅ | ~242K |
| `PE_PB_FED` | PE百分位 | ✅ | ✅ | ~2.9M |
| `PB_only` | PB百分位 | ❌ | ❌ | ~60K |
| `PB_FED` | PB百分位 | ❌ | ✅ | ~242K |

> **FED 门控**: `fed_val < fed_mean + fd * fed_std` 时不买入（fed低=债券便宜=应买债不买股）  
> **PB 否决**: `pb_pct >= vt` 时不买入（PB分位高=估值贵）

## 评分公式

```
per_index_score = avg_xirr_i × w1 - sigma_xirr_i × w2
strategy_score  = Σ_i per_index_score / N      (N = 有效指数数, ≥3)
```

- `avg_xirr_i`: 该指数运行 XIRR 年化序列的均值
- `sigma_xirr_i`: 该指数运行 XIRR 年化序列的标准差（波动率）
- XIRR 计算: **<1年用简单年化收益率** (ret / years), **≥1年用二分法 XIRR**

## 策略参数

```python
# 买入档位 (PE或PB百分位)
buy_floor   ∈ {0.05, 0.10, 0.15, 0.20}     # 3倍投入 (极度低估)
buy_low     ∈ {0.15, 0.20, 0.25, 0.30}     # 2倍投入
buy_mid     ∈ {0.30, 0.35, 0.40, 0.45}     # 1倍投入
buy_high    ∈ {0.50, 0.55, 0.60, 0.65, 0.70} # 0.5倍投入

# 卖出档位
sell_warn   ∈ {0.65, 0.70, 0.75}          # 警告区, 停止买入
sell_heavy  ∈ {0.75, 0.80, 0.85}          # 分批卖出 5%~25%/月
sell_extr   ∈ {0.85, 0.90, 0.95}          # 极端高估, 回撤清仓 25%~50%

# 辅助参数
fed_thresh  ∈ {-0.5, 0.0, 0.5, 1.0}      # FED 买入阈值 (标准差倍数)
pb_veto     ∈ {0.50, 0.55, 0.60, 0.65}   # PB 百分位否决阈值
pb_confirm  ∈ {0.70, 0.75, 0.80}         # (保留, PE_PB_FED 专用)
dd_std      ∈ {0.06, 0.08, 0.10, 0.12}   # 回撤判断参数
dd_tight    ∈ {0.03, 0.04, 0.05}         # 回撤卖出门槛

约束: buy_floor < buy_low < buy_mid < buy_high
      sell_warn < sell_heavy < sell_extreme
```

## 使用方法

```bash
cd /mnt/data/zmp/family-dca

# 默认参数
python grid_search/grid_search_20260809.py

# 自定义采样数和随机种子
python grid_search/grid_search_20260809.py --n 500 --seed 123

# 粗略估算
#   每个 (特征组合,窗口) = 1000 组参数 × 7 个指数
#   18 个 (特征组合,窗口) × 7000 ≈ 126,000 次 run_one
#   预估耗时: 10~30 分钟
```

## 输出文件

```
grid_search/output/
├── top5_w1_0_w0_0.csv      # w1=1.0 各组合 Top5
├── top5_w0_8_w0_2.csv      # w1=0.8 各组合 Top5
├── top5_w0_6_w0_4.csv      # w1=0.6 各组合 Top5
├── all_w1_0_w0_0.csv       # w1=1.0 全量结果
├── all_w0_8_w0_2.csv
├── all_w0_6_w0_4.csv
└── report.html              # 可视化报告
```

## 输出字段说明

| 字段 | 含义 |
|------|------|
| `strategy_score` | 跨7个宽基的平均评分 |
| `pe_buy_floor` ~ `pe_buy_high` | PE/PB 买入档位 (百分位) |
| `pe_sell_warn` ~ `pe_sell_extreme` | PE/PB 卖出档位 (百分位) |
| `fed_buy_threshold` | FED 买入阈值 (标准差, 仅含FED的组合) |
| `pb_veto_threshold` | PB 否决阈值 (仅含PB的组合) |
| `{code}_avg_xirr` | 单个指数的 avg_xirr |
| `{code}_sigma` | 单个指数的 sigma_xirr |

## 数据结构

```
data-store/parquet/
├── merged/           # 已合并的指标数据 (PE/PB/FED + 价格)
│   ├── 000300.parquet
│   └── ...
├── index_price/      # K线日频价格
│   └── ...
└── index_pe_dj/      # 蛋卷 PE (备用)
```

每个回测使用 `merged` 中的 `pe_ttm_dj` (蛋卷 PE)、`fed_dj` (FED 模型)、`pb_dj` (PB)，通过 `merge_asof` 对齐到蛋卷 PE 的周频日期。

## HTML 报告内容

1. **跨指数条形图**: Top1 策略在各指数上的 avg_xirr / sigma_xirr
2. **特征组合得分分布**: 各特征组合的 max/mean score (w1=1.0)
3. **Tab 表格**: 每个 (权重,窗口,特征组合) 的 Top5 策略参数明细

## 关键设计决策

- **所有归一化已内置**: PE%、PB% 默认 0~1 百分位, FED 使用 z-score (均值/标准差归一化)
- **PB only 时不启用PB否决**: PB 既是主信号又是否决条件会矛盾
- **拒绝采样**: 从 2.9M 空间中按约束条件随机抽样, 避免穷举
- **运行 XIRR 而非最终 XIRR**: 使用每个交易点的运行 XIRR 序列, 能反映策略稳定性
