import pandas as pd
import pytest

from src.execution import ExecutionHandler
from src.order import Order
from src.portfolio import Portfolio
from src.rebalancer import MIN_REBALANCE_NOTIONAL, Rebalancer
from src.risk_manager import RiskManager
from src.trading_engine import TradingEngine


def test_order_requires_exactly_one_of_quantity_or_notional():
    quantity_order = Order("AAPL", quantity=10, side="BUY")
    notional_order = Order("AAPL", side="BUY", notional=1000.0)

    assert quantity_order.quantity == 10
    assert quantity_order.notional is None
    assert notional_order.quantity is None
    assert notional_order.notional == pytest.approx(1000.0)
    with pytest.raises(ValueError, match="exactly one"):
        Order("AAPL", side="BUY")
    with pytest.raises(ValueError, match="exactly one"):
        Order("AAPL", quantity=10, side="BUY", notional=1000.0)


def test_notional_rebalancer_uses_exact_deltas_and_never_oversells():
    rebalancer = Rebalancer()
    current_values = {
        "UNDER": 5000.0,
        "OVER": 15000.0,
        "DROPPED": 2000.0,
    }

    orders = rebalancer.generate_notional_orders(
        target_weights={"UNDER": 0.10, "OVER": 0.10, "DROPPED": 0.0},
        account_equity=100000.0,
        current_market_values=current_values,
    )
    by_symbol = {order.symbol: order for order in orders}

    assert by_symbol["UNDER"].side == "BUY"
    assert by_symbol["UNDER"].notional == pytest.approx(5000.0)
    assert by_symbol["OVER"].side == "SELL"
    assert by_symbol["OVER"].notional == pytest.approx(5000.0)
    assert by_symbol["DROPPED"].side == "SELL"
    assert by_symbol["DROPPED"].notional == pytest.approx(2000.0)
    assert all(
        order.notional <= current_values[order.symbol]
        for order in orders
        if order.side == "SELL"
    )


def test_notional_rebalancer_ignores_only_operational_micro_deltas():
    orders = Rebalancer().generate_notional_orders(
        target_weights={"AAPL": 0.10},
        account_equity=100000.0,
        current_market_values={"AAPL": 9999.25},
    )

    assert MIN_REBALANCE_NOTIONAL == pytest.approx(1.0)
    assert orders == []


def test_notional_risk_enforces_position_cash_and_leverage_limits():
    risk = RiskManager(max_position_pct=0.06, max_leverage=1.0)
    frozen_target = Order("AAPL", side="BUY", notional=100000 / 17)
    oversized_target = Order("AAPL", side="BUY", notional=6001.0)

    assert risk.check_notional_order(
        frozen_target,
        account_equity=100000.0,
        current_position_market_values={},
        available_buying_power=100000.0,
    )
    assert not risk.check_notional_order(
        oversized_target,
        account_equity=100000.0,
        current_position_market_values={},
        available_buying_power=100000.0,
    )
    assert not risk.check_notional_order(
        frozen_target,
        account_equity=100000.0,
        current_position_market_values={},
        available_buying_power=5000.0,
    )

    diversified_values = {
        f"OLD{number}": 5900.0
        for number in range(16)
    }
    assert not risk.check_notional_order(
        frozen_target,
        account_equity=100000.0,
        current_position_market_values=diversified_values,
        available_buying_power=100000.0,
    )
    assert not risk.check_notional_order(
        Order("AAPL", side="SELL", notional=5001.0),
        account_equity=100000.0,
        current_position_market_values={"AAPL": 5000.0},
        available_buying_power=0.0,
    )


class _TwoAssetStrategy:
    name = "Test Notional Strategy"

    def generate_target_weights(self, history):
        return {"AAA": 0.05, "BBB": 0.05}


class _EligibleBroker:
    def __init__(self, unsupported=None):
        self.unsupported = unsupported
        self.submitted = []

    def supports_notional_order(self, symbol):
        return symbol != self.unsupported

    def submit_order(self, order):
        self.submitted.append(order)
        return f"order-{order.symbol}"


def _notional_engine(broker):
    return TradingEngine(
        Portfolio(100000),
        RiskManager(max_position_pct=0.06, max_leverage=1.0),
        ExecutionHandler(commission_rate=0.0, slippage_rate=0.0),
        Rebalancer(),
        _TwoAssetStrategy(),
        broker=broker,
    )


def test_notional_batch_consumes_buying_power_cumulatively():
    plan = _notional_engine(_EligibleBroker()).preview_notional_broker_cycle(
        history=pd.DataFrame({"AAA": [1.0], "BBB": [1.0]}),
        account_equity=100000.0,
        current_position_market_values={},
        available_buying_power=6000.0,
    )

    assert [order.symbol for order in plan["accepted_orders"]] == ["AAA"]
    assert [
        rejection["order"].symbol
        for rejection in plan["rejected_orders"]
    ] == ["BBB"]


def test_notional_asset_ineligibility_fails_the_whole_plan():
    broker = _EligibleBroker(unsupported="BBB")

    with pytest.raises(RuntimeError, match="BBB"):
        _notional_engine(broker).preview_notional_broker_cycle(
            history=pd.DataFrame({"AAA": [1.0], "BBB": [1.0]}),
            account_equity=100000.0,
            current_position_market_values={},
            available_buying_power=100000.0,
        )

    assert broker.submitted == []
