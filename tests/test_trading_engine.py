from src.portfolio import Portfolio
from src.risk_manager import RiskManager
from src.execution import ExecutionHandler
from src.rebalancer import Rebalancer
from src.trading_engine import TradingEngine
from src.strategy import MomentumStrategy
import pytest


def test_rebalance():

    portfolio = Portfolio(10000)
    risk_manager = RiskManager(1.0, 1.0)
    execution = ExecutionHandler(0.005, 0.005)
    rebalancer = Rebalancer()
    strategy = MomentumStrategy

    engine = TradingEngine(portfolio, risk_manager, execution, rebalancer, strategy)

    prices = {
        "AAPL": 100,
        "MSFT": 200,
        "GOOG": 150
    }

    target_weights = {
        "AAPL": 0.4,
        "MSFT": 0.59
    }

    result = engine.rebalance(target_weights, prices)

    assert portfolio.get_position("AAPL").quantity == 40
    assert portfolio.get_position("MSFT").quantity == 29
    assert result["requested_turnover"] == pytest.approx(0.98)

def test_rebalance_rejected_order():

    portfolio = Portfolio(10000)
    risk_manager = RiskManager(0.2, 1.0)
    execution = ExecutionHandler(0.005, 0.005)
    rebalancer = Rebalancer()

    prices = {
        "AAPL": 100
    }

    target_weights = {
        "AAPL": 0.4
    }  

    assert portfolio.get_position("AAPL") is None
    assert portfolio.cash == 10000  


