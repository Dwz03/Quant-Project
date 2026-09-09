import pytest
from src.portfolio import Portfolio
from src.paper_trading_engine import PaperTradingEngine
from src.strategy import MomentumTradingStrategy
from src.trading_engine import TradingEngine
from src.execution import ExecutionHandler
from src.risk_manager import RiskManager
from src.rebalancer import Rebalancer
import pandas as pd

def test_paper_trading_engine_market_closed():

    class FakeClock:

        is_open = False


    class FakeBroker:

        def get_account(self):

            class Account:
                cash = "10000"
                equity = "10000"

            return Account()

        def can_short(
            self,
            symbol
        ):
            return True

        def get_positions(self):
            return []

        def get_clock(self):
            return FakeClock()


    class FakeMarketData:
        pass


    class FakeTradingEngine:

        def __init__(self):

            self.portfolio = Portfolio(
                10000
            )


    paper_engine = PaperTradingEngine(
        broker=FakeBroker(),
        market_data=FakeMarketData(),
        trading_engine=FakeTradingEngine(),
        symbols=["AAPL", "MSFT"]
    )

    result = paper_engine.run_cycle()

    assert (
        result["status"]
        == "MARKET_CLOSED"
    )

    assert result["orders"] == []

def test_paper_trading_engine_open_market_end_to_end():

    # ==================================
    # Fake Market Data
    # ==================================

    class FakeMarketData:

        def get_history(
            self,
            symbols,
            start,
            end
        ):

            return pd.DataFrame(
                {
                    "AAPL": [
                        100,
                        101,
                        102,
                        103
                    ],

                    "MSFT": [
                        200,
                        199,
                        198,
                        197
                    ]
                },
                index=pd.date_range(
                    "2026-01-01",
                    periods=4
                )
            )


        def get_latest_price(
            self,
            symbol
        ):

            prices = {
                "AAPL": 104,
                "MSFT": 196
            }

            return prices[symbol]


    # ==================================
    # Fake Broker Objects
    # ==================================

    class FakeStatus:

        def __init__(self, value):
            self.value = value


    class FakeClock:

        is_open = True


    class FakeBrokerOrder:

        def __init__(
            self,
            order_id,
            symbol,
            quantity,
            side,
            price
        ):

            self.id = order_id

            self.symbol = symbol

            self.qty = quantity

            self.side = side

            self.status = FakeStatus(
                "filled"
            )

            self.filled_qty = quantity

            self.filled_avg_price = price


    class FakePosition:

        def __init__(
            self,
            symbol,
            qty,
            avg_entry_price
        ):

            self.symbol = symbol
            self.qty = str(qty)

            self.avg_entry_price = str(
                avg_entry_price
            )


    class FakeAccount:

        def __init__(
            self,
            cash,
            equity
        ):

            self.cash = str(cash)
            self.equity = str(equity)


    # ==================================
    # Fake Broker
    # ==================================

    class FakeBroker:

        def __init__(self):

            self.orders = {}

            self.prices = {
                "AAPL": 104,
                "MSFT": 196
            }

        def can_short(
            self,
            symbol
        ):
            return True


        def get_clock(self):

            return FakeClock()


        def submit_order(
            self,
            order
        ):

            order_id = (
                f"order-{len(self.orders) + 1}"
            )

            broker_order = FakeBrokerOrder(
                order_id=order_id,
                symbol=order.symbol,
                quantity=order.quantity,
                side=order.side,
                price=self.prices[
                    order.symbol
                ]
            )

            self.orders[
                order_id
            ] = broker_order

            return order_id


        def get_order(
            self,
            order_id
        ):

            return self.orders[
                order_id
            ]


        def get_positions(self):

            positions = []

            quantities = {}

            for order in self.orders.values():

                sign = (
                    1
                    if order.side == "BUY"
                    else -1
                )

                quantities[
                    order.symbol
                ] = (
                    quantities.get(
                        order.symbol,
                        0
                    )
                    +
                    sign
                    * order.qty
                )

            for symbol, qty in (
                quantities.items()
            ):

                if qty != 0:

                    positions.append(
                        FakePosition(
                            symbol,
                            qty,
                            self.prices[
                                symbol
                            ]
                        )
                    )

            return positions


        def get_account(self):

            cash = 10000

            for order in (
                self.orders.values()
            ):

                value = (
                    order.qty
                    * self.prices[
                        order.symbol
                    ]
                )

                if order.side == "BUY":

                    cash -= value

                else:

                    cash += value

            return FakeAccount(
                cash=cash,
                equity=10000
            )


    # ==================================
    # Real Trading Components
    # ==================================

    broker = FakeBroker()

    market_data = FakeMarketData()

    portfolio = Portfolio(
        10000
    )

    strategy = MomentumTradingStrategy(
        lookback=3,
        target_weight=0.10,
        allow_short=True
    )

    risk_manager = RiskManager(
        max_position_pct=0.20,
        max_leverage=1.0
    )

    execution = ExecutionHandler(
        commission_rate=0.0,
        slippage_rate=0.0
    )

    rebalancer = Rebalancer()

    trading_engine = TradingEngine(
        portfolio,
        risk_manager,
        execution,
        rebalancer,
        strategy,
        broker=broker
    )

    paper_engine = PaperTradingEngine(
        broker=broker,
        market_data=market_data,
        trading_engine=trading_engine,
        symbols=[
            "AAPL",
            "MSFT"
        ],
        lookback_days=30
    )

    # ==================================
    # Run ONE autonomous cycle
    # ==================================

    result = (
        paper_engine.run_cycle()
    )

    # ==================================
    # Assertions
    # ==================================

    assert (
        result["status"]
        == "COMPLETED"
    )

    weights = result[
        "target_weights"
    ]

    assert weights["AAPL"] > 0
    assert weights["MSFT"] < 0

    assert (
        len(result["orders"])
        == 2
    )

    assert (
        len(result["order_updates"])
        == 2
    )

    assert all(
        update["status"]
        == "filled"
        for update
        in result["order_updates"]
    )

    # Broker reconciliation should
    # have updated local portfolio

    aapl_position = (
        portfolio.get_position(
            "AAPL"
        )
    )

    msft_position = (
        portfolio.get_position(
            "MSFT"
        )
    )

    assert aapl_position.quantity > 0

    assert msft_position.quantity < 0

def test_paper_engine_skips_duplicate_cycle():

    class FakeClock:

        is_open = True

        timestamp = pd.Timestamp(
            "2026-09-09 14:30:00",
            tz="UTC"
        )


    class FakeAccount:

        cash = "10000"
        equity = "10000"


    class FakeBroker:

        def __init__(self):
            self.submission_count = 0

        def can_short(
            self,
            symbol
        ):
            return True

        def get_clock(self):
            return FakeClock()

        def get_account(self):
            return FakeAccount()

        def get_positions(self):
            return []

        def submit_order(
            self,
            order
        ):

            self.submission_count += 1

            return (
                f"order-"
                f"{self.submission_count}"
            )


    class FakeMarketData:

        def get_history(
            self,
            symbols,
            start,
            end
        ):

            return pd.DataFrame(
                {
                    "AAPL": [
                        100,
                        101,
                        102,
                        103
                    ]
                },
                index=pd.date_range(
                    "2026-09-01",
                    periods=4
                )
            )

        def get_latest_price(
            self,
            symbol
        ):

            return 105

    class FakePortfolio:

        def sync_from_broker(
            self,
            account,
            positions
        ):
            pass


    class FakeTradingEngine:

        def __init__(self):

            self.portfolio = (
                FakePortfolio()
            )

            self.run_count = 0


        def run_broker_cycle(
            self,
            history,
            cycle_key=None
        ):

            self.run_count += 1

            return {
                "target_weights": {
                    "AAPL": 0.10
                },
                "orders": [],
                "submitted_orders": []
            }

    broker = FakeBroker()

    trading_engine = (
        FakeTradingEngine()
    )

    paper_engine = PaperTradingEngine(
        broker=broker,
        market_data=FakeMarketData(),
        trading_engine=trading_engine,
        symbols=["AAPL"]
    )


    first = paper_engine.run_cycle()

    second = paper_engine.run_cycle()


    assert first["status"] == "COMPLETED"

    assert (
        second["status"]
        == "DUPLICATE_SKIPPED"
    )

    assert (
        trading_engine.run_count
        == 1
    )

def test_paper_engine_kill_switch_after_failure():

    class FakeClock:

        is_open = True

        timestamp = pd.Timestamp(
            "2026-09-09 14:30:00",
            tz="UTC"
        )


    class FakeAccount:

        cash = "10000"
        equity = "10000"


    class FakeBroker:

        def get_account(self):
            return FakeAccount()

        def can_short(
            self,
            symbol
        ):
            return True

        def get_positions(self):
            return []

        def get_clock(self):
            return FakeClock()


    class FakeMarketData:

        def get_history(
            self,
            symbols,
            start,
            end
        ):

            return pd.DataFrame(
                {
                    "AAPL": [
                        100,
                        101,
                        102
                    ]
                },
                index=pd.date_range(
                    "2026-09-01",
                    periods=3
                )
            )


        def get_latest_price(
            self,
            symbol
        ):

            return 103


    class FakePortfolio:

        def sync_from_broker(
            self,
            account,
            positions
        ):
            pass


    class FakeTradingEngine:

        def __init__(self):

            self.portfolio = FakePortfolio()

            self.run_count = 0


        def run_broker_cycle(
            self,
            history,
            cycle_key=None
        ):

            self.run_count += 1

            raise RuntimeError(
                "simulated broker failure"
            )


    trading_engine = (
        FakeTradingEngine()
    )

    paper_engine = PaperTradingEngine(
        broker=FakeBroker(),
        market_data=FakeMarketData(),
        trading_engine=trading_engine,
        symbols=["AAPL"]
    )


    # First run fails
    with pytest.raises(
        RuntimeError
    ):

        paper_engine.run_cycle()


    assert (
        paper_engine.kill_switch_active
        is True
    )

    assert (
        paper_engine.kill_switch_reason
        == "simulated broker failure"
    )


    # Second run must NOT enter
    # trading pipeline again
    result = (
        paper_engine.run_cycle()
    )


    assert (
        result["status"]
        == "KILL_SWITCH_ACTIVE"
    )

    assert (
        trading_engine.run_count
        == 1
    )

def test_should_run_before_market_close():

    class FakeClock:

        is_open = True

        timestamp = pd.Timestamp(
            "2026-09-09 15:50:00",
            tz="America/New_York"
        )

        next_close = pd.Timestamp(
            "2026-09-09 16:00:00",
            tz="America/New_York"
        )


    class FakeBroker:
        pass


    class FakeMarketData:
        pass


    class FakeTradingEngine:
        pass


    engine = PaperTradingEngine(
        broker=FakeBroker(),
        market_data=FakeMarketData(),
        trading_engine=(
            FakeTradingEngine()
        ),
        symbols=["AAPL"],
        minutes_before_close=15
    )


    assert (
        engine._should_run_now(
            FakeClock()
        )
        is True
    )

def test_should_not_run_too_early():

    class FakeClock:

        is_open = True

        timestamp = pd.Timestamp(
            "2026-09-09 15:30:00",
            tz="America/New_York"
        )

        next_close = pd.Timestamp(
            "2026-09-09 16:00:00",
            tz="America/New_York"
        )


    engine = PaperTradingEngine(
        broker=object(),
        market_data=object(),
        trading_engine=object(),
        symbols=["AAPL"],
        minutes_before_close=15
    )


    assert (
        engine._should_run_now(
            FakeClock()
        )
        is False
    )