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
from src.research import (split_data, estimate_beta, calculate_spread, check_spread_stationarity,
                          check_cointegration, estimate_hedge_ratio)

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

    ko = load_market_data("KO", "2020-01-01", "2025-12-31")
    pep = load_market_data("PEP", "2020-01-01", "2025-12-31")

    pair_data = pd.DataFrame({
        "symbol_1": ko["Close"],
        "symbol_2": pep["Close"]
    }).dropna()

    train, validation, test = split_data(
        pair_data,
        train_ratio=0.6,
        validation_ratio=0.2
    )

    hedge = estimate_hedge_ratio(train)

    alpha = hedge["alpha"]
    beta = hedge["beta"]

    train_spread = calculate_spread(
        train,
        beta=beta,
        alpha=alpha
    )

    stationarity = check_spread_stationarity(
        train_spread["spread"]
    )

    cointegration = check_cointegration(train)

    print("alpha:", alpha)
    print("beta:", beta)
    print("ADF:", stationarity)
    print("Cointegration:", cointegration)






# Test
if __name__ == "__main__":
    main()


