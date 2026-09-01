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
from src.trading_engine import TradingEngine

import pandas as pd

# Main script

config = {
    "initial_cash" : 10000,
    "risk" : {
        "max_position_weight" : 1.0,
        "max_leverage" : 1.0
    },
    "execution" : {
        "commission_rate" : 0.005,
        "slippage_rate" : 0.005
    }
}


def main():

    portfolio = Portfolio(config["initial_cash"])

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

    risk_manager = RiskManager(config["risk"]["max_position_weight"], config["risk"]["max_leverage"])

    execution = ExecutionHandler(config["execution"]["slippage_rate"], config["execution"]["commission_rate"])

    engine = TradingEngine(portfolio, risk_manager, execution, rebalancer)

    result = engine.rebalance(target_weights, prices)

    for symbol, position in portfolio.positions.items():
        print(symbol, position.quantity)

    print(f"Turnover: {result['turnover']:.2%}")

    print("Cash:", portfolio.cash)
    print("Total value:", portfolio.total_value(prices))


# Test
if __name__ == "__main__":
    main()


