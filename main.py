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
from src.events import MarketEvent, SignalEvent, OrderEvent, FillEvent

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

    market_event = MarketEvent("AAPL", 105)

    print(market_event.symbol)
    print(market_event.price)
    print(market_event.type)

    signal_event = SignalEvent("AAPL", "BUY")

    print(signal_event.symbol)
    print(signal_event.signal)
    print(signal_event.type)

    order = Order("AAPL", 40, "BUY")
    order_event = OrderEvent(order)

    print(order_event.type)
    print(order_event.order.symbol)
    print(order_event.order.quantity)

    fill = Fill("AAPL", 40, "BUY", 105, 0.005)
    event = FillEvent(fill)

    print(event.type)
    print(event.fill.symbol)
    print(event.fill.quantity)
    print(event.fill.side)
    print(event.fill.price)
    





# Test
if __name__ == "__main__":
    main()


