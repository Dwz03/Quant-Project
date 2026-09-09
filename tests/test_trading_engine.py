from src.portfolio import Portfolio
from src.risk_manager import RiskManager
from src.execution import ExecutionHandler
from src.rebalancer import Rebalancer
from src.trading_engine import TradingEngine
from src.strategy import MomentumStrategy, MeanReversionTradingStrategy, MomentumTradingStrategy
from src.position import Position
from src.order import Order
import pytest
import pandas as pd


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

def test_run_mean_reversion_strategy_cycle():

    history = pd.DataFrame({
        "AAPL": [100, 102, 101, 95]
    })

    portfolio = Portfolio(10000)

    risk_manager = RiskManager(
        max_position_pct=1.0,
        max_leverage=1.0
    )

    execution = ExecutionHandler(
        commission_rate=0.0,
        slippage_rate=0.0
    )

    rebalancer = Rebalancer()

    strategy = MeanReversionTradingStrategy(
        window=3,
        target_weight=0.5
    )

    engine = TradingEngine(
        portfolio,
        risk_manager,
        execution,
        rebalancer,
        strategy
    )

    result = engine.run_strategy_cycle(history)

    assert result["target_weights"]["AAPL"] == 0.5

    assert portfolio.get_position("AAPL").quantity > 0

class FakeBroker:

    def __init__(self):
        self.submitted_orders = []

    def submit_order(self, order):

        self.submitted_orders.append(order)

        return {
            "id": "test-order-1",
            "status": "submitted",
            "symbol": order.symbol
        }

def test_run_broker_cycle_submits_strategy_order():

    history = pd.DataFrame({
        "AAPL": [100, 102, 101, 95]
    })

    portfolio = Portfolio(10000)

    risk_manager = RiskManager(
        max_position_pct=1.0,
        max_leverage=1.0
    )

    execution = ExecutionHandler(
        commission_rate=0.0,
        slippage_rate=0.0
    )

    rebalancer = Rebalancer()

    strategy = MeanReversionTradingStrategy(
        window=3,
        target_weight=0.5
    )

    broker = FakeBroker()

    engine = TradingEngine(
        portfolio,
        risk_manager,
        execution,
        rebalancer,
        strategy,
        broker=broker
    )

    result = engine.run_broker_cycle(history)

    assert len(broker.submitted_orders) == 1

    assert broker.submitted_orders[0].symbol == "AAPL"

    assert result["target_weights"]["AAPL"] == 0.5

def test_reconcile_broker_orders():

    class FakeStatus:

        value = "filled"


    class FakeBrokerOrder:

        id = "order-1"
        symbol = "AAPL"
        status = FakeStatus()
        filled_qty = "10"
        filled_avg_price = "100.5"


    class FakeAccount:

        cash = "8995.0"
        equity = "10000.0"


    class FakeBrokerPosition:

        symbol = "AAPL"
        qty = "10"
        avg_entry_price = "100.5"


    class FakeBroker:

        def get_order(self, order_id):

            assert order_id == "order-1"

            return FakeBrokerOrder()

        def get_account(self):

            return FakeAccount()

        def get_positions(self):

            return [
                FakeBrokerPosition()
            ]


    portfolio = Portfolio(10000)

    engine = TradingEngine(
        portfolio=portfolio,
        risk_manager=None,
        execution=None,
        rebalancer=None,
        strategy=None,
        broker=FakeBroker()
    )

    result = engine.reconcile_broker_orders(
        ["order-1"]
    )

    assert len(result) == 1

    assert result[0]["status"] == "filled"

    assert result[0]["filled_qty"] == 10

    assert (
        result[0]["filled_avg_price"]
        == 100.5
    )

    assert portfolio.cash == 8995.0

    assert (
        portfolio.get_position("AAPL").quantity
        == 10
    )

    assert (
        portfolio.get_position("AAPL").average_cost
        == 100.5
    )

def test_wait_for_orders_until_filled(monkeypatch):

    class FakeStatus:

        def __init__(self, value):
            self.value = value


    class FakeBrokerOrder:

        def __init__(self, status):
            self.symbol = "AAPL"
            self.status = FakeStatus(status)
            self.filled_qty = (
                "10" if status == "filled" else "0"
            )
            self.filled_avg_price = (
                "100.5"
                if status == "filled"
                else None
            )


    class FakeAccount:
        cash = "8995"
        equity = "10000"


    class FakePosition:
        symbol = "AAPL"
        qty = "10"
        avg_entry_price = "100.5"


    class FakeBroker:

        def __init__(self):
            self.calls = 0

        def get_order(self, order_id):

            self.calls += 1

            if self.calls == 1:
                return FakeBrokerOrder(
                    "accepted"
                )

            return FakeBrokerOrder(
                "filled"
            )

        def get_account(self):
            return FakeAccount()

        def get_positions(self):
            return [FakePosition()]


    # Prevent real waiting inside test
    monkeypatch.setattr(
        "src.trading_engine.time.sleep",
        lambda seconds: None
    )

    portfolio = Portfolio(10000)

    broker = FakeBroker()

    engine = TradingEngine(
        portfolio=portfolio,
        risk_manager=None,
        execution=None,
        rebalancer=None,
        strategy=None,
        broker=broker
    )

    result = engine.wait_for_orders(
        ["order-1"],
        timeout=5,
        poll_interval=0
    )

    assert result[0]["status"] == "filled"

    assert (
        portfolio.get_position("AAPL").quantity
        == 10
    )

    assert portfolio.cash == 8995

def test_client_order_id_is_deterministic():

    portfolio = Portfolio(10000)

    strategy = MomentumTradingStrategy(
        lookback=2,
        target_weight=0.10
    )

    risk_manager = RiskManager(
        max_position_pct=0.20,
        max_leverage=1.0
    )

    execution = ExecutionHandler(
        0.0,
        0.0
    )

    rebalancer = Rebalancer()


    class FakeBroker:

        def __init__(self):
            self.client_ids = []

        def submit_order(
            self,
            order
        ):

            self.client_ids.append(
                order.client_order_id
            )

            return "fake-order"


    broker = FakeBroker()

    engine = TradingEngine(
        portfolio,
        risk_manager,
        execution,
        rebalancer,
        strategy,
        broker=broker
    )

    history = pd.DataFrame({
        "AAPL": [
            100,
            101,
            105
        ]
    })


    engine.run_broker_cycle(
        history,
        cycle_key="2026-09-09"
    )

    first_id = (
        broker.client_ids[0]
    )


    engine.run_broker_cycle(
        history,
        cycle_key="2026-09-09"
    )

    second_id = (
        broker.client_ids[1]
    )


    assert first_id == second_id

    assert first_id.startswith(
        "qt-20260909-"
    )

def test_unshortable_asset_is_not_submitted():

    class FakeBroker:

        def __init__(self):
            self.submit_count = 0

        def can_short(
            self,
            symbol
        ):
            return False

        def submit_order(
            self,
            order
        ):
            self.submit_count += 1
            return "fake-order"


    portfolio = Portfolio(
        10000
    )

    strategy = MomentumTradingStrategy(
        lookback=2,
        target_weight=0.10,
        allow_short=True
    )

    risk_manager = RiskManager(
        max_position_pct=0.20,
        max_leverage=1.0
    )

    execution = ExecutionHandler(
        0.0,
        0.0
    )

    rebalancer = Rebalancer()

    broker = FakeBroker()

    engine = TradingEngine(
        portfolio,
        risk_manager,
        execution,
        rebalancer,
        strategy,
        broker=broker
    )


    # Falling price -> Momentum wants short
    history = pd.DataFrame({
        "AAPL": [
            105,
            102,
            100
        ]
    })


    result = engine.run_broker_cycle(
        history,
        cycle_key="2026-09-09"
    )


    assert len(
        result["orders"]
    ) == 1

    assert (
        result["orders"][0].side
        == "SELL"
    )

    assert (
        result["submitted_orders"]
        == []
    )

    assert (
        broker.submit_count
        == 0
    )

def test_closing_long_does_not_require_shortability():

    class FakeBroker:

        def __init__(self):
            self.submit_count = 0

        def can_short(
            self,
            symbol
        ):
            return False

        def submit_order(
            self,
            order
        ):
            self.submit_count += 1
            return "fake-order"


    portfolio = Portfolio(
        10000
    )

    # Existing long position
    portfolio.positions["AAPL"] = Position(
        symbol="AAPL",
        quantity=10,
        average_cost=100
    )

    broker = FakeBroker()

    strategy = MomentumTradingStrategy(
        lookback=2,
        target_weight=0.0,
        allow_short=False
    )

    risk_manager = RiskManager(
        max_position_pct=1.0,
        max_leverage=1.0
    )

    execution = ExecutionHandler(
        0.0,
        0.0
    )

    rebalancer = Rebalancer()

    engine = TradingEngine(
        portfolio,
        risk_manager,
        execution,
        rebalancer,
        strategy,
        broker=broker
    )


    history = pd.DataFrame({
        "AAPL": [
            100,
            101,
            102
        ]
    })


    result = engine.run_broker_cycle(
        history,
        cycle_key="2026-09-09"
    )


    assert len(
        result["orders"]
    ) == 1

    order = result["orders"][0]

    assert (
        order.side
        == "SELL"
    )

    assert (
        order.quantity
        == 10
    )

    assert (
        broker.submit_count
        == 1
    )

def test_sell_beyond_long_position_requires_shortability():

    class FakeBroker:

        def __init__(self):
            self.submit_count = 0

        def can_short(
            self,
            symbol
        ):
            return False

        def submit_order(
            self,
            order
        ):
            self.submit_count += 1
            return "fake-order"


    portfolio = Portfolio(
        10000
    )

    portfolio.positions["AAPL"] = Position(
        symbol="AAPL",
        quantity=5,
        average_cost=100
    )


    broker = FakeBroker()


    order = Order(
        symbol="AAPL",
        quantity=10,
        side="SELL"
    )


    engine = TradingEngine(
        portfolio=portfolio,
        risk_manager=None,
        execution=None,
        rebalancer=None,
        strategy=None,
        broker=broker
    )


    assert (
        engine._requires_shorting(
            order
        )
        is True
    )

    assert (
        engine._broker_allows_order(
            order
        )
        is False
    )