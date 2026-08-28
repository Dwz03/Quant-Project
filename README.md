# Quant Trading Project

A learning project to build a simple quantitative trading system while developing
Python, software engineering, data structures, algorithms, and backtesting skills.

## Current Features

- Market data loading and local caching
- Basic trading strategies
- Portfolio and position management
- Backtesting engine
- Transaction costs and slippage
- Performance metrics:
  - Total return
  - Annualized volatility
  - Sharpe ratio
  - Maximum drawdown
- Unit testing with pytest
- Basic event queue

## Project Structure

```text
quant_project/
├── main.py
├── src/
│   ├── backtest.py
│   ├── data_loader.py
│   ├── eventqueue.py
│   ├── metrics.py
│   ├── portfolio.py
│   ├── position.py
│   ├── strategy.py
│   └── tradingbot.py
├── tests/
├── practice/
└── data/