# import
import matplotlib.pyplot as plt
from src.position import Position
from src.portfolio import Portfolio
from src.strategy import MovingAverageStrategy, MomentumStrategy
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
from src.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from src.event_backtester import EventDrivenBacktester

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

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

    portfolio = Portfolio(10000)

    risk_manager = RiskManager(1.0, 1.0)

    execution = ExecutionHandler(
        slippage_rate=0.001,
        commission_rate=0.005
    )

    rebalancer = Rebalancer()

    strategy = MomentumStrategy(2)

    engine = TradingEngine(
        portfolio,
        risk_manager,
        execution,
        rebalancer,
        strategy
    )

    backtester = EventDrivenBacktester(
        engine,
        "AAPL"
    )

    prices = [
        100,
        105,
        102,
        108,
        110
    ]

    results = backtester.run(prices)

    print(results)



# Test
if __name__ == "__main__":
    main()


