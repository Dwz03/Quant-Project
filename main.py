# import
import matplotlib.pyplot as plt
from src.position import Position
from src.portfolio import Portfolio
from src.strategy import MovingAverageStrategy
from src.tradingbot import TradingBot
from src.data_loader import (load_market_data, add_returns, get_data_path,
                             clean_market_data, save_market_data, load_local_market_data)
from src.backtest import Backtester
from src.metrics import performance_summary

# Main script
def main():

    data = load_market_data(
    "SPY",
    "2020-01-01",
    "2025-01-01")

    strategy = MovingAverageStrategy(short_window=20,long_window=50)

    data = add_returns(data)

    data = strategy.generate_signal(data)

    backtester = Backtester(data)

    result = backtester.run()

    print(result[["Close", "Signal", "Position", "Return",
                  "Strategy_Return", "equity", "Buy_Hold_Equity"]].tail())

    summary = performance_summary(result["Strategy_Return"])

    print(f"Total Return: {summary['Total Return']:.2%}")
    print(f"Annualized Volatility: {summary['Annualized Volatility']:.2%}")
    print(f"Sharpe Ratio: {summary['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {summary['Max Drawdown']:.2%}")


# Test
if __name__ == "__main__":
    main()
    print(__name__)


