#!/usr/bin/env python3
"""多因子选股系统（示例版）

功能：
1. 拉取股票历史行情与基础面数据（默认使用 yfinance）
2. 计算多因子（动量、波动率、估值、质量、规模）
3. 对因子做标准化并按权重合成总分
4. 输出 Top N 选股结果与评分明细

注意：
- 该脚本是一个可扩展框架，适合先搭建研究流水线，再逐步替换成更稳定的数据源。
- 生产实盘请增加：交易成本、滑点、停牌/涨跌停、行业中性化、风险暴露约束。
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclasses.dataclass
class FactorWeights:
    momentum: float = 0.30
    low_volatility: float = 0.20
    value: float = 0.20
    quality: float = 0.20
    size: float = 0.10

    def normalized(self) -> "FactorWeights":
        total = sum(dataclasses.asdict(self).values())
        if total <= 0:
            raise ValueError("因子权重总和必须大于 0")
        scale = 1.0 / total
        return FactorWeights(**{k: v * scale for k, v in dataclasses.asdict(self).items()})


class MultiFactorStockSelector:
    def __init__(self, tickers: Iterable[str], weights: Optional[FactorWeights] = None):
        self.tickers = [t.strip().upper() for t in tickers if t.strip()]
        if not self.tickers:
            raise ValueError("股票池不能为空")
        self.weights = (weights or FactorWeights()).normalized()

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """截尾 + 标准化，增强横截面鲁棒性。"""
        clipped = series.clip(lower=series.quantile(0.05), upper=series.quantile(0.95))
        std = clipped.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (clipped - clipped.mean()) / std

    @staticmethod
    def _safe_inv(value: float) -> float:
        if value is None or np.isnan(value) or value == 0:
            return np.nan
        return 1.0 / value

    def load_market_data(self, period: str = "1y") -> pd.DataFrame:
        """下载价格数据，返回 MultiIndex columns 的 DataFrame。"""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("缺少依赖 yfinance，请先 pip install -r requirements.txt") from exc

        data = yf.download(
            tickers=self.tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if data.empty:
            raise RuntimeError("未获取到行情数据，请检查代码/网络")
        return data

    def load_fundamental_snapshot(self) -> pd.DataFrame:
        """拉取截面基础面数据（PE、ROE、市值）。"""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("缺少依赖 yfinance，请先 pip install -r requirements.txt") from exc

        rows = []
        for ticker in self.tickers:
            info = yf.Ticker(ticker).info
            rows.append(
                {
                    "ticker": ticker,
                    "pe": info.get("trailingPE"),
                    "roe": info.get("returnOnEquity"),
                    "market_cap": info.get("marketCap"),
                }
            )
        df = pd.DataFrame(rows).set_index("ticker")
        return df

    def build_factor_table(self, prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """构建因子表并计算总分。"""
        # 处理 yfinance 输出结构：Close 可能是单层或多层列
        if isinstance(prices.columns, pd.MultiIndex):
            close = prices["Close"]
        else:
            close = prices[["Close"]]
            close.columns = [self.tickers[0]]

        momentum = close.iloc[-1] / close.iloc[-126] - 1.0
        vol = close.pct_change().tail(63).std(ddof=0)

        factor_df = pd.DataFrame(index=close.columns)
        factor_df["momentum_raw"] = momentum
        factor_df["low_volatility_raw"] = -vol

        factor_df = factor_df.join(fundamentals, how="left")
        factor_df["value_raw"] = factor_df["pe"].apply(self._safe_inv)
        factor_df["quality_raw"] = factor_df["roe"]
        factor_df["size_raw"] = -np.log(factor_df["market_cap"].replace({0: np.nan}))

        # 标准化
        factor_df["momentum_z"] = self._zscore(factor_df["momentum_raw"])
        factor_df["low_volatility_z"] = self._zscore(factor_df["low_volatility_raw"])
        factor_df["value_z"] = self._zscore(factor_df["value_raw"])
        factor_df["quality_z"] = self._zscore(factor_df["quality_raw"])
        factor_df["size_z"] = self._zscore(factor_df["size_raw"])

        w = self.weights
        factor_df["score"] = (
            w.momentum * factor_df["momentum_z"]
            + w.low_volatility * factor_df["low_volatility_z"]
            + w.value * factor_df["value_z"]
            + w.quality * factor_df["quality_z"]
            + w.size * factor_df["size_z"]
        )

        factor_df = factor_df.sort_values("score", ascending=False)
        return factor_df

    def select_top_n(self, factor_table: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        return factor_table.head(top_n).copy()


def run_demo(output_path: Path, top_n: int) -> pd.DataFrame:
    """离线演示：构造伪数据，保证无网络环境下可运行。"""
    np.random.seed(7)
    tickers = [f"DEMO{i:02d}" for i in range(1, 21)]
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=260, freq="B")

    price_panel = {
        t: 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.02, len(dates))))
        for t in tickers
    }
    close = pd.DataFrame(price_panel, index=dates)
    prices = pd.concat({"Close": close}, axis=1)

    fundamentals = pd.DataFrame(
        {
            "pe": np.random.uniform(8, 35, len(tickers)),
            "roe": np.random.uniform(0.03, 0.35, len(tickers)),
            "market_cap": np.random.uniform(5e9, 3e12, len(tickers)),
        },
        index=tickers,
    )

    selector = MultiFactorStockSelector(tickers)
    table = selector.build_factor_table(prices, fundamentals)
    top = selector.select_top_n(table, top_n=top_n)
    top.to_csv(output_path, encoding="utf-8-sig")
    return top


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多因子选股系统")
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,BRK-B,JPM,UNH",
        help="股票池，逗号分隔。示例：AAPL,MSFT,NVDA",
    )
    parser.add_argument("--period", type=str, default="1y", help="历史窗口，如 6mo/1y/2y")
    parser.add_argument("--top-n", type=int, default=5, help="输出前 N 只股票")
    parser.add_argument("--output", type=str, default="selected_stocks.csv", help="结果输出 CSV 路径")
    parser.add_argument("--demo", action="store_true", help="使用离线演示数据，不联网")

    # 自定义权重
    parser.add_argument("--w-momentum", type=float, default=0.30)
    parser.add_argument("--w-low-volatility", type=float, default=0.20)
    parser.add_argument("--w-value", type=float, default=0.20)
    parser.add_argument("--w-quality", type=float, default=0.20)
    parser.add_argument("--w-size", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    if args.demo:
        top = run_demo(output_path=output_path, top_n=args.top_n)
        print("[DEMO] 已生成离线选股结果：")
        print(top[["score"]].round(4))
        print(f"\nCSV 已输出到: {output_path.resolve()}")
        return

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    weights = FactorWeights(
        momentum=args.w_momentum,
        low_volatility=args.w_low_volatility,
        value=args.w_value,
        quality=args.w_quality,
        size=args.w_size,
    )

    selector = MultiFactorStockSelector(tickers=tickers, weights=weights)
    prices = selector.load_market_data(period=args.period)
    fundamentals = selector.load_fundamental_snapshot()

    factor_table = selector.build_factor_table(prices=prices, fundamentals=fundamentals)
    selected = selector.select_top_n(factor_table=factor_table, top_n=args.top_n)
    selected.to_csv(output_path, encoding="utf-8-sig")

    print("已完成多因子打分，Top 结果如下：")
    print(selected[["score", "momentum_raw", "value_raw", "quality_raw"]].round(4))
    print(f"\nCSV 已输出到: {output_path.resolve()}")


if __name__ == "__main__":
    main()
