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
from src.risk_manager import RiskManager

import pandas as pd

# Main script
def main():

    portfolio = Portfolio(10000)

    prices = {
        "AAPL": 100,
        "MSFT": 200,
        "GOOG": 150
    }

    orders = [
        Order("AAPL", 15, "BUY"),
        Order("MSFT", 7, "BUY"),
        Order("GOOG", 10, "BUY"),
        Order("AAPL", 5, "SELL")
    ]

    risk_manager = RiskManager(0.2, 1.0)

    execution = ExecutionHandler(0.001, 0.005)

    for order in orders:
        if risk_manager.check_order(order, portfolio, prices):
            fill = execution.execute_order(order, prices[order.symbol])
            portfolio.process_fill(fill)

        else:
            print("Order rejected by risk manager")

    for symbol, position in portfolio.positions.items():
        print(f"{symbol}: {position.quantity}")

    print(f"portfolio remaining cash: {portfolio.cash}")


# Test
if __name__ == "__main__":
    main()


