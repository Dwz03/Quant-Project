from src.risk_manager import RiskManager
from src.portfolio import Portfolio
from src.order import Order
from src.position import Position
import pytest

def test_risk_manager():

    portfolio = Portfolio(100000)

    prices = {"AAPL": 100}

    order = Order(symbol="AAPL", quantity=100, side="BUY")

    risk_manager = RiskManager(max_position_pct=0.2, max_leverage=1.0)

    assert risk_manager.check_order(order, portfolio, prices) is True

def test_reject_order_when_max_position_exceeded():

    portfolio = Portfolio(100000)

    prices = {"AAPL": 100}

    risk_manager = RiskManager(max_position_pct=0.2, max_leverage=1.0)

    order = Order(symbol="AAPL", quantity=300, side="BUY")

    assert risk_manager.check_order(order, portfolio, prices) is False

def test_reject_order_when_no_enough_cash():

    portfolio = Portfolio(10000)

    prices = {"AAPL": 100}

    risk_manager = RiskManager(max_position_pct=1.0,  max_leverage=2.0)

    order = Order(symbol="AAPL", quantity=150, side="BUY")

    assert risk_manager.check_order(order, portfolio, prices) is False

def test_reject_order_when_leverage_exceeded():

    portfolio = Portfolio(10000)

    prices = {"AAPL": 100}

    risk_manager = RiskManager(max_position_pct=1.0, max_leverage=0.5)

    order = Order(symbol="AAPL", quantity=60, side="BUY")

    assert risk_manager.check_order(order, portfolio, prices) is False