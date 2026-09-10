import pytest
from pathlib import Path
from src.portfolio import Portfolio
from src.paper_trading_engine import PaperTradingEngine
from src.strategy import MomentumTradingStrategy
from src.trading_engine import TradingEngine
from src.execution import ExecutionHandler
from src.risk_manager import RiskManager
from src.rebalancer import Rebalancer
import pandas as pd


def test_default_state_file_is_independent_of_working_directory(
    monkeypatch,
    tmp_path
):

    first = PaperTradingEngine(
        broker=object(),
        market_data=object(),
        trading_engine=object(),
        symbols=["AAPL"]
    )

    monkeypatch.chdir(tmp_path)

    second = PaperTradingEngine(
        broker=object(),
        market_data=object(),
        trading_engine=object(),
        symbols=["AAPL"]
    )

    expected_project_root = Path(
        __file__
    ).resolve().parent.parent

    assert first.state_file == second.state_file
    assert first.state_file.is_absolute()
    assert first.state_file == (
        expected_project_root
        / ".state"
        / "paper_cycle.json"
    )

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

        def get_open_orders(self):
            return []

        def get_order_history(self, start, end):
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
        symbols=["AAPL", "MSFT"],
        state_file=None
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


        def get_open_orders(self):

            return []


        def get_order_history(self, start, end):

            return list(self.orders.values())


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
        lookback_days=30,
        state_file=None
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

        def get_open_orders(self):
            return []

        def get_order_history(self, start, end):
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
        symbols=["AAPL"],
        state_file=None
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

        def get_open_orders(self):
            return []

        def get_order_history(self, start, end):
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
        symbols=["AAPL"],
        state_file=None
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


class PendingOrderStatus:

    def __init__(self, value):
        self.value = value


class PendingBrokerOrder:

    def __init__(
        self,
        status="new",
        filled_qty="0",
        client_order_id="cycle-order-1",
        order_id="existing-order-1",
        symbol="AAPL",
        quantity="10"
    ):

        self.id = order_id
        self.client_order_id = client_order_id
        self.symbol = symbol
        self.side = PendingOrderStatus("buy")
        self.qty = quantity
        self.filled_qty = filled_qty
        self.status = PendingOrderStatus(status)


class ReconciliationClock:

    is_open = True
    timestamp = pd.Timestamp(
        "2026-09-09 15:50:00",
        tz="America/New_York"
    )
    next_close = pd.Timestamp(
        "2026-09-09 16:00:00",
        tz="America/New_York"
    )


class ReconciliationBroker:

    def __init__(
        self,
        open_orders=None,
        order_history=None
    ):
        self.open_orders = list(
            open_orders or []
        )
        self.order_history = list(
            order_history or []
        )

    def get_account(self):

        class Account:
            cash = "10000"
            equity = "10000"

        return Account()

    def get_positions(self):
        return []

    def get_open_orders(self):
        return list(self.open_orders)

    def get_order_history(self, start, end):
        return list(self.order_history)

    def get_clock(self):
        return ReconciliationClock()


class ReconciliationMarketData:

    def get_history(self, symbols, start, end):
        return pd.DataFrame(
            {"AAPL": [100, 101, 102]},
            index=pd.date_range(
                "2026-09-01",
                periods=3
            )
        )

    def get_latest_price(self, symbol):
        return 103


class ReconciliationPortfolio:

    def sync_from_broker(self, account, positions):
        pass


class ReconciliationTradingEngine:

    def __init__(self, timeout=False):
        self.portfolio = ReconciliationPortfolio()
        self.timeout = timeout
        self.run_count = 0
        self.wait_count = 0

    def run_broker_cycle(self, history, cycle_key=None):
        self.run_count += 1

        return {
            "target_weights": {"AAPL": 0.1},
            "orders": [],
            "submitted_orders": (
                ["new-order-1"]
                if self.timeout
                else []
            )
        }

    def wait_for_orders(
        self,
        order_ids,
        timeout,
        poll_interval
    ):
        self.wait_count += 1
        raise TimeoutError("orders still pending")


def build_reconciliation_engine(
    broker,
    trading_engine=None,
    state_file=None
):

    return PaperTradingEngine(
        broker=broker,
        market_data=ReconciliationMarketData(),
        trading_engine=(
            trading_engine
            or ReconciliationTradingEngine()
        ),
        symbols=["AAPL"],
        state_file=state_file
    )


def test_existing_open_order_blocks_new_cycle():

    order = PendingBrokerOrder()
    trading_engine = ReconciliationTradingEngine()
    paper_engine = build_reconciliation_engine(
        ReconciliationBroker([order]),
        trading_engine
    )

    result = paper_engine.run_cycle()

    assert result["status"] == "OPEN_ORDERS_PENDING"
    assert result["outstanding_orders"] == [{
        "order_id": "existing-order-1",
        "client_order_id": "cycle-order-1",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": "10",
        "filled_quantity": "0",
        "status": "new"
    }]
    assert trading_engine.run_count == 0
    assert paper_engine.last_cycle_key is None


def test_partially_filled_order_blocks_new_cycle():

    order = PendingBrokerOrder(
        status="partially_filled",
        filled_qty="4"
    )
    trading_engine = ReconciliationTradingEngine()
    paper_engine = build_reconciliation_engine(
        ReconciliationBroker([order]),
        trading_engine
    )

    result = paper_engine.run_cycle()

    assert result["status"] == "OPEN_ORDERS_PENDING"
    assert (
        result["outstanding_orders"][0]
        ["filled_quantity"]
        == "4"
    )
    assert trading_engine.run_count == 0


def test_disappeared_open_order_allows_normal_cycle():

    broker = ReconciliationBroker([
        PendingBrokerOrder()
    ])
    trading_engine = ReconciliationTradingEngine()
    paper_engine = build_reconciliation_engine(
        broker,
        trading_engine
    )

    first = paper_engine.run_cycle()
    broker.open_orders = []
    second = paper_engine.run_cycle()

    assert first["status"] == "OPEN_ORDERS_PENDING"
    assert second["status"] == "COMPLETED"
    assert trading_engine.run_count == 1


def test_order_timeout_stays_alive_and_prevents_duplicate_batch():

    pending_order = PendingBrokerOrder(
        client_order_id="qt-20260909-timeout"
    )
    broker = ReconciliationBroker()
    trading_engine = ReconciliationTradingEngine(
        timeout=True
    )
    original_run_broker_cycle = (
        trading_engine.run_broker_cycle
    )

    def run_broker_cycle(history, cycle_key=None):
        broker.order_history = [pending_order]
        return original_run_broker_cycle(
            history,
            cycle_key=cycle_key
        )

    trading_engine.run_broker_cycle = (
        run_broker_cycle
    )
    paper_engine = build_reconciliation_engine(
        broker,
        trading_engine
    )

    broker.open_orders = [pending_order]
    original_get_open_orders = broker.get_open_orders
    first_query = True

    def get_open_orders_after_submission():
        nonlocal first_query

        if first_query:
            first_query = False
            return []

        return original_get_open_orders()

    broker.get_open_orders = (
        get_open_orders_after_submission
    )

    timed_out = paper_engine.run_cycle()
    still_pending = paper_engine.run_scheduled_step()

    broker.open_orders = []
    pending_order.status = PendingOrderStatus(
        "filled"
    )
    pending_order.filled_qty = pending_order.qty
    reconciled = paper_engine.run_scheduled_step()
    duplicate = paper_engine.run_scheduled_step()

    assert timed_out["status"] == "ORDERS_PENDING"
    assert still_pending["status"] == "OPEN_ORDERS_PENDING"
    assert reconciled["status"] == "FILLED_COMPLETED"
    assert duplicate["status"] == "FILLED_COMPLETED"
    assert trading_engine.run_count == 1
    assert trading_engine.wait_count == 1
    assert paper_engine.kill_switch_active is False


def test_restart_while_bot_order_is_open_blocks_cycle():

    order = PendingBrokerOrder(
        client_order_id="qt-20260909-open"
    )
    broker = ReconciliationBroker(
        open_orders=[order],
        order_history=[order]
    )
    restarted_trading_engine = (
        ReconciliationTradingEngine()
    )
    restarted_engine = build_reconciliation_engine(
        broker,
        restarted_trading_engine
    )

    result = restarted_engine.run_cycle()

    assert result["cycle_state"] == "PENDING"
    assert restarted_trading_engine.run_count == 0


def test_restart_after_fully_filled_order_consumes_cycle():

    order = PendingBrokerOrder(
        status="filled",
        filled_qty="10",
        client_order_id="qt-20260909-filled"
    )
    trading_engine = ReconciliationTradingEngine()
    restarted_engine = build_reconciliation_engine(
        ReconciliationBroker(
            order_history=[order]
        ),
        trading_engine
    )

    result = restarted_engine.run_cycle()

    assert result["status"] == "FILLED_COMPLETED"
    assert trading_engine.run_count == 0


@pytest.mark.parametrize(
    "terminal_status",
    ["canceled", "rejected", "expired"]
)
def test_restart_after_terminal_unfilled_order_consumes_cycle(
    terminal_status
):

    order = PendingBrokerOrder(
        status=terminal_status,
        client_order_id=(
            f"qt-20260909-{terminal_status}"
        )
    )
    trading_engine = ReconciliationTradingEngine()
    restarted_engine = build_reconciliation_engine(
        ReconciliationBroker(
            order_history=[order]
        ),
        trading_engine
    )

    result = restarted_engine.run_cycle()

    assert result["status"] == "TERMINAL_INCOMPLETE"
    assert trading_engine.run_count == 0


def test_restart_after_partial_fill_then_cancel_consumes_cycle():

    order = PendingBrokerOrder(
        status="canceled",
        filled_qty="4",
        client_order_id="qt-20260909-partial"
    )
    trading_engine = ReconciliationTradingEngine()
    restarted_engine = build_reconciliation_engine(
        ReconciliationBroker(
            order_history=[order]
        ),
        trading_engine
    )

    result = restarted_engine.run_cycle()

    assert result["status"] == "TERMINAL_INCOMPLETE"
    assert (
        result["cycle_orders"][0]
        ["filled_quantity"]
        == "4"
    )
    assert trading_engine.run_count == 0


def test_restart_with_mixed_batch_uses_safest_state():

    filled_order = PendingBrokerOrder(
        status="filled",
        filled_qty="10",
        client_order_id="qt-20260909-filled",
        order_id="filled-order"
    )
    second_order = PendingBrokerOrder(
        status="partially_filled",
        filled_qty="3",
        client_order_id="qt-20260909-second",
        order_id="second-order",
        symbol="MSFT"
    )
    broker = ReconciliationBroker(
        open_orders=[second_order],
        order_history=[
            filled_order,
            second_order
        ]
    )

    pending_engine = build_reconciliation_engine(
        broker,
        ReconciliationTradingEngine()
    )
    pending = pending_engine.run_cycle()

    broker.open_orders = []
    second_order.status = PendingOrderStatus(
        "rejected"
    )
    terminal_trading_engine = (
        ReconciliationTradingEngine()
    )
    terminal_engine = build_reconciliation_engine(
        broker,
        terminal_trading_engine
    )
    terminal = terminal_engine.run_cycle()

    assert pending["cycle_state"] == "PENDING"
    assert terminal["status"] == "TERMINAL_INCOMPLETE"
    assert terminal_trading_engine.run_count == 0


def test_restart_after_first_submitted_batch_order_consumes_cycle():

    first_order = PendingBrokerOrder(
        status="filled",
        filled_qty="10",
        client_order_id="qt-20260909-first"
    )
    trading_engine = ReconciliationTradingEngine()
    restarted_engine = build_reconciliation_engine(
        ReconciliationBroker(
            order_history=[first_order]
        ),
        trading_engine
    )

    result = restarted_engine.run_cycle()

    assert result["status"] == "FILLED_COMPLETED"
    assert trading_engine.run_count == 0


@pytest.mark.parametrize(
    "client_order_id",
    [
        "qt-20260908-yesterday",
        "manual-order-20260909"
    ]
)
def test_nonmatching_history_does_not_consume_today(
    client_order_id
):

    old_or_unrelated_order = PendingBrokerOrder(
        status="filled",
        filled_qty="10",
        client_order_id=client_order_id
    )
    trading_engine = ReconciliationTradingEngine()
    engine = build_reconciliation_engine(
        ReconciliationBroker(
            order_history=[old_or_unrelated_order]
        ),
        trading_engine
    )

    result = engine.run_cycle()

    assert result["status"] == "COMPLETED"
    assert trading_engine.run_count == 1


def test_zero_order_cycle_is_consumed_after_restart(
    tmp_path
):

    state_file = tmp_path / "paper_cycle.json"
    broker = ReconciliationBroker()
    first_trading_engine = (
        ReconciliationTradingEngine()
    )
    first_engine = build_reconciliation_engine(
        broker,
        first_trading_engine,
        state_file=state_file
    )

    first = first_engine.run_cycle()

    restarted_trading_engine = (
        ReconciliationTradingEngine()
    )
    restarted_engine = build_reconciliation_engine(
        broker,
        restarted_trading_engine,
        state_file=state_file
    )
    after_restart = restarted_engine.run_cycle()

    assert first["status"] == "COMPLETED"
    assert after_restart["status"] == "DUPLICATE_SKIPPED"
    assert first_trading_engine.run_count == 1
    assert restarted_trading_engine.run_count == 0
