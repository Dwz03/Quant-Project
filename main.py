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

    results = []

    for cost, slip in zip([0, 0.0005, 0.001], [0, 0.0002, 0.0005]):

        backtester = Backtester(data)

        result = backtester.run(cost_rate=cost, slippage_rate=slip)
        summary_net = performance_summary(result["Net_Strategy_Return"])

        results.append({"Cost Rate": cost, "Slippage Rate": slip, **summary_net})

    sensitivity_df = pd.DataFrame(results)

    sensitivity_df["Cost (bps)"] = sensitivity_df["Cost Rate"] * 10000
    sensitivity_df["Slippage (bps)"] = sensitivity_df["Slippage Rate"] * 10000

    sensitivity_df = sensitivity_df.drop(columns=["Cost Rate", "Slippage Rate"])

    print(sensitivity_df)

# Test
if __name__ == "__main__":
    main()
    print(__name__)


