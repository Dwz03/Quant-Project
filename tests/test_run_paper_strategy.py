from src.portfolio import Portfolio
from src.strategy import (
    MeanReversionTradingStrategy,
    MovingAverageTradingStrategy,
    PairsTradingStrategy,
    PCAResidualTradingStrategy,
)
from src.rebalancer import Rebalancer
from src.risk_manager import RiskManager
from src.execution import ExecutionHandler
from src.trading_engine import TradingEngine
from src.strategy_factory import build_strategy
from scripts.run_paper_strategy import (
    _execution_feed_for_strategy,
    _require_paper_broker,
)
from alpaca.data.enums import DataFeed
import pandas as pd
import numpy as np
import pytest


def test_paper_runner_accepts_explicit_paper_broker():

    class FakePaperBroker:
        is_paper = True

    _require_paper_broker(
        FakePaperBroker()
    )


@pytest.mark.parametrize(
    "broker",
    [object(), type("LiveBroker", (), {
        "is_paper": False
    })()]
)
def test_paper_runner_rejects_broker_without_paper_mode(
    broker
):

    with pytest.raises(
        RuntimeError,
        match="explicitly paper-mode broker"
    ):
        _require_paper_broker(broker)


def test_volatility_paper_profile_explicitly_selects_sip_execution_data():
    assert _execution_feed_for_strategy("volatility_20") == DataFeed.SIP
    assert _execution_feed_for_strategy("moving_average") == DataFeed.IEX


def test_synced_portfolio_does_not_duplicate_order():

    class FakeAccount:
        cash = "9000"
        equity = "10000"

    class FakeBrokerPosition:

        symbol = "AAPL"
        qty = "10"
        avg_entry_price = "95"

    portfolio = Portfolio(10000)

    portfolio.sync_from_broker(
        FakeAccount(),
        [FakeBrokerPosition()]
    )

    history = pd.DataFrame({
        "AAPL": [
            105,
            104,
            103,
            100
        ]
    })

    strategy = MeanReversionTradingStrategy(
        window=3,
        target_weight=0.10,
        max_gross_exposure=0.10,
        allow_short=False
    )

    target_weights = (
        strategy.generate_target_weights(history)
    )

    prices = {
        "AAPL": 100
    }

    rebalancer = Rebalancer()

    orders = rebalancer.generate_orders(
        target_weights,
        portfolio,
        prices
    )

    assert target_weights == {
        "AAPL": 0.10
    }

    assert len(orders) == 0

def test_pairs_strategy_end_to_end_broker_cycle():

    history = pd.DataFrame({
        "AAPL": [
            100, 101, 102, 103,
            104, 105, 106, 90
        ],
        "MSFT": [
            200, 202, 204, 206,
            208, 210, 212, 214
        ]
    })

    portfolio = Portfolio(10000)

    strategy = PairsTradingStrategy(
        symbol_1="AAPL",
        symbol_2="MSFT",
        window=3,
        threshold=1.0,
        target_gross_exposure=0.10
    )

    risk_manager = RiskManager(
        max_position_pct=0.10,
        max_leverage=1.0
    )

    rebalancer = Rebalancer()

    execution = ExecutionHandler(
        commission_rate=0.0,
        slippage_rate=0.0
    )


    class FakeBroker:

        def __init__(self):
            self.submitted = []

        def can_short(
            self,
            symbol
        ):
            return True

        def submit_order(self, order):

            self.submitted.append(order)

            return (
                f"fake-{order.symbol}-"
                f"{order.side}"
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

    result = engine.run_broker_cycle(
        history
    )


    # -------------------------
    # 1. Strategy layer
    # -------------------------

    weights = result["target_weights"]

    assert weights["AAPL"] > 0
    assert weights["MSFT"] < 0

    gross = (
        abs(weights["AAPL"])
        + abs(weights["MSFT"])
    )

    assert gross == pytest.approx(
        0.10
    )


    # -------------------------
    # 2. Rebalancer layer
    # -------------------------

    orders = result["orders"]

    assert len(orders) == 2

    order_by_symbol = {
        order.symbol: order
        for order in orders
    }

    assert (
        order_by_symbol["AAPL"].side
        == "BUY"
    )

    assert (
        order_by_symbol["MSFT"].side
        == "SELL"
    )


    # -------------------------
    # 3. Broker layer
    # -------------------------

    assert len(
        result["submitted_orders"]
    ) == 2

    assert len(
        broker.submitted
    ) == 2

def test_pca_strategy_end_to_end_broker_cycle():

    factor = np.linspace(
        -0.01,
        0.01,
        15
    )

    returns = pd.DataFrame({
        "AAPL": factor,

        "MSFT":
            0.8 * factor
            + np.array([
                0.001,
                -0.001,
                0.0005,
                -0.0005,
                0.0,
                0.001,
                -0.001,
                0.0005,
                -0.0005,
                0.0,
                0.001,
                -0.001,
                0.0005,
                -0.0005,
                0.05
            ]),

        "GOOG":
            1.2 * factor
            + np.array([
                -0.0005,
                0.0005,
                -0.001,
                0.001,
                0.0,
                -0.0005,
                0.0005,
                -0.001,
                0.001,
                0.0,
                -0.0005,
                0.0005,
                -0.001,
                0.001,
                -0.04
            ])
    })

    prices = (
        100
        * (1 + returns).cumprod()
    )

    portfolio = Portfolio(10000)

    strategy = PCAResidualTradingStrategy(
        n_components=1,
        window=5,
        threshold=1.0,
        target_gross_exposure=0.10
    )

    risk_manager = RiskManager(
        max_position_pct=0.10,
        max_leverage=1.0
    )

    execution = ExecutionHandler(
        commission_rate=0.0,
        slippage_rate=0.0
    )

    rebalancer = Rebalancer()


    class FakeBroker:

        def __init__(self):
            self.submitted = []

        def can_short(
            self,
            symbol
        ):
            return True

        def submit_order(self, order):

            self.submitted.append(order)

            return (
                f"fake-{order.symbol}-"
                f"{order.side}"
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

    result = engine.run_broker_cycle(
        prices
    )


    # -------------------------
    # 1. Strategy
    # -------------------------

    weights = result["target_weights"]

    assert set(weights.keys()) == {
        "AAPL",
        "MSFT",
        "GOOG"
    }

    gross = sum(
        abs(weight)
        for weight in weights.values()
    )

    assert gross == pytest.approx(
        0.10
    )

    assert weights["MSFT"] < 0
    assert weights["GOOG"] > 0


    # -------------------------
    # 2. Orders
    # -------------------------

    orders = result["orders"]

    assert len(orders) >= 2

    order_by_symbol = {
        order.symbol: order
        for order in orders
    }

    assert (
        order_by_symbol["MSFT"].side
        == "SELL"
    )

    assert (
        order_by_symbol["GOOG"].side
        == "BUY"
    )


    # -------------------------
    # 3. Broker
    # -------------------------

    assert len(
        result["submitted_orders"]
    ) == len(
        broker.submitted
    )

    assert len(
        broker.submitted
    ) >= 2


def test_build_momentum_strategy():

    strategy = build_strategy(
        "momentum"
    )

    assert (
        strategy.name
        == "Momentum"
    )


@pytest.mark.parametrize(
    "alias",
    ["moving_average", "ma"]
)
def test_build_moving_average_strategy_aliases(alias):

    strategy = build_strategy(alias)

    assert isinstance(
        strategy,
        MovingAverageTradingStrategy
    )


def test_build_moving_average_strategy_propagates_config():

    strategy = build_strategy(
        "moving_average",
        config={
            "short_window": 5,
            "long_window": 20,
            "target_weight": 0.25,
            "allow_short": True
        }
    )

    assert strategy.short_window == 5
    assert strategy.long_window == 20
    assert strategy.target_weight == 0.25
    assert strategy.allow_short is True


def test_build_moving_average_strategy_defaults_remain_compatible():

    strategy = build_strategy("moving_average")

    assert strategy.short_window == 10
    assert strategy.long_window == 30
    assert strategy.target_weight == 0.01
    assert strategy.allow_short is False

def test_unknown_strategy():

    with pytest.raises(
        ValueError
    ):

        build_strategy(
            "random_strategy"
        )

def test_build_pairs_strategy():

    strategy = build_strategy(
        "pairs",
        symbols=[
            "AAPL",
            "MSFT"
        ]
    )

    assert (
        strategy.name
        == "Pairs Trading"
    )

    assert (
        strategy.symbol_1
        == "AAPL"
    )

    assert (
        strategy.symbol_2
        == "MSFT"
    )

def test_build_pca_strategy():

    strategy = build_strategy(
        "pca",
        symbols=[
            "AAPL",
            "MSFT",
            "GOOG"
        ]
    )

    assert (
        strategy.name
        == "PCA Residual Stat Arb"
    )

    assert (
        strategy.n_components
        == 1
    )

    assert (
        strategy.target_gross_exposure
        == 0.01
    )

def test_pairs_requires_two_symbols():

    with pytest.raises(
        ValueError
    ):

        build_strategy(
            "pairs",
            symbols=["AAPL"]
        )
