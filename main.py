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
from src.fill import Fill
from src.order import Order
from src.execution import ExecutionHandler

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

    portfolio = Portfolio(10000)

    order = Order("AAPL", 10, "BUY")

    execution = ExecutionHandler(
        slippage_rate=0.001,
        commission_rate=0.0005
    )

    fill = execution.execute_order(order, 100)

    portfolio.process_fill(fill)

    print(order.status)
    print(fill.price)
    print(portfolio.cash)
    print(portfolio.positions["AAPL"].quantity)

    sell_order = Order("AAPL", 4, "SELL")

    sell_fill = execution.execute_order(sell_order, 110)

    portfolio.process_fill(sell_fill)

    print(sell_order.status)
    print(sell_fill.price)
    print(portfolio.cash)
    print(portfolio.positions["AAPL"].quantity)

# Test
if __name__ == "__main__":
    main()


