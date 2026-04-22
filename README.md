# 多因子选股系统

这是一个可直接运行的 Python 多因子选股脚手架，支持：

- 行情 + 基础面数据拉取（`yfinance`）
- 多因子计算（动量、低波、估值、质量、规模）
- 因子标准化与加权打分
- 输出 Top N 股票结果到 CSV
- 离线 `--demo` 模式（无网络也可演示）

## 1) 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

## 2) 快速开始（离线演示）

```bash
python3 multifactor_system.py --demo --top-n 5
```

运行后会生成 `selected_stocks.csv`。

## 3) 实盘数据运行（联网）

```bash
python3 multifactor_system.py \
  --tickers "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,BRK-B,JPM,UNH" \
  --period 1y \
  --top-n 5 \
  --output selected_stocks.csv
```

## 4) 自定义因子权重

```bash
python3 multifactor_system.py --demo \
  --w-momentum 0.35 \
  --w-low-volatility 0.15 \
  --w-value 0.20 \
  --w-quality 0.20 \
  --w-size 0.10
```

> 程序会自动把权重归一化（和不一定为 1 也可）。

## 5) 因子说明

- `momentum`: 近 126 个交易日收益率，越高越好
- `low_volatility`: 近 63 个交易日波动率取负，越高越好（即波动越低越好）
- `value`: `1 / PE`，越高越便宜
- `quality`: `ROE`，越高越好
- `size`: `-log(market_cap)`，偏向小市值

## 6) 风险提示

本项目用于研究与教学，不构成投资建议。建议在实盘前补充：

- 交易成本与滑点
- 停牌/涨跌停处理
- 行业/风格中性化
- 回测与样本外验证
- 风控约束（单票上限、换手率上限、最大回撤阈值）
