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

import pandas as pd

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

    result = backtester.run(cost_rate = 0.0005, slippage_rate = 0.0001)

    summary_strategy = performance_summary(result["Net_Strategy_Return"])
    summary_buy_hold = performance_summary(result["Net_Buy_Hold_Return"])

    comparison = pd.DataFrame({"Strategy" : summary_strategy, "Buy & Hold" : summary_buy_hold})

    print(comparison)

    plt.figure(figsize = (10, 6))

    plt.plot(result.index, result["Net_Equity"], label = "Strategy")
    plt.plot(result.index, result["Net_Buy_Hold_Equity"], label = "Buy & Hold")

    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Strategy vs Buy & Hold")
    plt.legend()

    plt.show()

# Test
if __name__ == "__main__":
    main()
    print(__name__)


