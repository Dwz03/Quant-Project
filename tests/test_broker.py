import pytest

from src.broker import PaperBroker, Broker
from src.order import Order
from src.portfolio import Portfolio
from src.alpaca_broker import AlpacaPaperBroker


def test_broker_cannot_be_instantiated():

    with pytest.raises(TypeError):
        Broker()


def test_submit_order():

    broker = PaperBroker()

    order = Order("AAPL", 10, "BUY")

    order_id = broker.submit_order(order)

    assert order_id in broker.orders
    assert broker.get_order_status(order_id) == "SUBMITTED"


def test_cancel_order():

    broker = PaperBroker()

    order = Order("AAPL", 10, "BUY")

    order_id = broker.submit_order(order)

    result = broker.cancel_order(order_id)

    assert result is True
    assert broker.get_order_status(order_id) == "CANCELLED"


def test_get_invalid_order():

    broker = PaperBroker()

    with pytest.raises(ValueError):
        broker.get_order_status("invalid-id")

def test_order_lifecycle():

    broker = PaperBroker()

    order = Order("AAPL", 100, "BUY")

    order_id = broker.submit_order(order)

    assert broker.get_order_status(order_id) == "SUBMITTED"
    assert broker.get_filled_quantity(order_id) == 0

    broker.accept_order(order_id)

    assert broker.get_order_status(order_id) == "ACCEPTED"

    broker.fill_order(
        order_id,
        quantity=40,
        price=100
    )

    assert broker.get_order_status(order_id) == "PARTIALLY_FILLED"
    assert broker.get_filled_quantity(order_id) == 40

    broker.fill_order(
        order_id,
        quantity=60,
        price=101
    )

    assert broker.get_order_status(order_id) == "FILLED"
    assert broker.get_filled_quantity(order_id) == 100

def test_order_cannot_overfill():

    broker = PaperBroker()

    order = Order("AAPL", 100, "BUY")

    order_id = broker.submit_order(order)

    broker.accept_order(order_id)

    with pytest.raises(ValueError):

        broker.fill_order(
            order_id,
            quantity=101,
            price=100
        )

def test_broker_fill_updates_portfolio():

    broker = PaperBroker(commission_rate=0.001)

    portfolio = Portfolio(10000)

    order = Order("AAPL", 100, "BUY")

    order_id = broker.submit_order(order)

    broker.accept_order(order_id)

    fill = broker.fill_order(
        order_id,
        quantity=40,
        price=100
    )

    portfolio.process_fill(fill)

    assert broker.get_order_status(order_id) == "PARTIALLY_FILLED"

    assert broker.get_filled_quantity(order_id) == 40

    assert portfolio.get_position("AAPL").quantity == 40

    assert portfolio.cash == pytest.approx(5996)

    assert portfolio.total_commission == pytest.approx(4)

def test_alpaca_broker_get_positions_and_clock():

    class FakeClient:

        def get_all_positions(self):
            return ["test-position"]

        def get_clock(self):
            return "test-clock"

    broker = AlpacaPaperBroker.__new__(
        AlpacaPaperBroker
    )

    broker.client = FakeClient()

    assert broker.get_positions() == [
        "test-position"
    ]

    assert broker.get_clock() == "test-clock"