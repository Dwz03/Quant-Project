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
from src.rebalancer import Rebalancer

import pandas as pd

# Main script
def main():

    portfolio = Portfolio(10000)

    prices = {
        "AAPL": 100,
        "MSFT": 200,
        "GOOG": 150
    }

    target_weights = {
        "AAPL": 0.4,
        "MSFT": 0.59
    }

    rebalancer = Rebalancer()

    orders = rebalancer.generate_orders(target_weights, portfolio, prices)

    turnover = rebalancer.calculate_turnover(orders, portfolio, prices)

    print(f"Turnover: {turnover:.2%}")

    risk_manager = RiskManager(1.0, 1.0)

    execution = ExecutionHandler(0.001, 0.005)

    for order in orders:

        if risk_manager.check_order(order, portfolio, prices):

            fill = execution.execute_order(
                order,
                prices[order.symbol]
            )

            portfolio.process_fill(fill)

        else:
            print(
                f"{order.symbol} {order.side} rejected"
            )

    for symbol, position in portfolio.positions.items():
        print(symbol, position.quantity)

    print("Cash:", portfolio.cash)
    print("Total value:", portfolio.total_value(prices))


# Test
if __name__ == "__main__":
    main()


