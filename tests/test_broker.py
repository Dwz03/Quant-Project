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


def test_alpaca_broker_explicitly_declares_paper_mode():

    assert AlpacaPaperBroker.is_paper is True


def test_alpaca_broker_get_open_orders():

    from alpaca.trading.enums import (
        QueryOrderStatus
    )

    class FakeClient:

        def __init__(self):
            self.order_filter = None

        def get_orders(self, filter):
            self.order_filter = filter
            return ["open-order"]

    broker = AlpacaPaperBroker.__new__(
        AlpacaPaperBroker
    )
    broker.client = FakeClient()

    result = broker.get_open_orders()

    assert result == ["open-order"]
    assert (
        broker.client.order_filter.status
        == QueryOrderStatus.OPEN
    )


def test_alpaca_broker_get_order_history():

    from datetime import datetime, timezone
    from alpaca.trading.enums import (
        QueryOrderStatus
    )

    class FakeClient:

        def __init__(self):
            self.order_filter = None

        def get_orders(self, filter):
            self.order_filter = filter
            return ["historical-order"]

    start = datetime(
        2026,
        9,
        9,
        tzinfo=timezone.utc
    )
    end = datetime(
        2026,
        9,
        10,
        tzinfo=timezone.utc
    )
    broker = AlpacaPaperBroker.__new__(
        AlpacaPaperBroker
    )
    broker.client = FakeClient()

    result = broker.get_order_history(
        start,
        end
    )

    assert result == ["historical-order"]
    assert (
        broker.client.order_filter.status
        == QueryOrderStatus.ALL
    )
    assert broker.client.order_filter.after == start
    assert broker.client.order_filter.until == end
    assert broker.client.order_filter.limit == 500
    assert broker.client.order_filter.limit == 500

def test_alpaca_broker_passes_client_order_id():

    from src.order import Order


    class FakeBrokerOrder:

        id = "broker-order-123"


    class FakeClient:

        def __init__(self):

            self.request = None


        def get_order_by_client_id(
            self,
            client_order_id
        ):

            from requests import (
                HTTPError,
                Response
            )

            from alpaca.common.exceptions import (
                APIError
            )

            response = Response()

            response.status_code = 404

            http_error = HTTPError(
                response=response
            )

            raise APIError(
                (
                    '{"code":40410000,'
                    '"message":"order not found"}'
                ),
                http_error
            )


        def submit_order(
            self,
            order_data
        ):

            self.request = order_data

            return FakeBrokerOrder()


    broker = (
        AlpacaPaperBroker
        .__new__(
            AlpacaPaperBroker
        )
    )

    broker.client = FakeClient()


    order = Order(
        symbol="AAPL",
        quantity=10,
        side="BUY",
        client_order_id=(
            "qt-20260909-test123"
        )
    )


    result = broker.submit_order(
        order
    )


    assert (
        result
        == "broker-order-123"
    )

    assert (
        broker.client
        .request
        .client_order_id
        == "qt-20260909-test123"
    )

def test_can_short():

    class FakeAccount:

        shorting_enabled = True


    class FakeAsset:

        tradable = True
        shortable = True


    class FakeClient:

        def get_account(self):

            return FakeAccount()


        def get_asset(
            self,
            symbol
        ):

            return FakeAsset()


    broker = (
        AlpacaPaperBroker
        .__new__(
            AlpacaPaperBroker
        )
    )

    broker.client = FakeClient()


    assert (
        broker.can_short(
            "AAPL"
        )
    is True
    )


def test_alpaca_paper_broker_submits_notional_without_quantity():
    class FakeBrokerOrder:
        id = "broker-notional-123"

    class FakeClient:
        def submit_order(self, order_data):
            self.request = order_data
            return FakeBrokerOrder()

    broker = AlpacaPaperBroker.__new__(AlpacaPaperBroker)
    broker.client = FakeClient()
    order = Order(symbol="AAPL", side="BUY", notional=5882.35)

    assert broker.submit_order(order) == "broker-notional-123"
    assert broker.client.request.notional == pytest.approx(5882.35)
    assert broker.client.request.qty is None
    assert broker.client.request.type.value == "market"
    assert broker.client.request.time_in_force.value == "day"


@pytest.mark.parametrize(
    ("tradable", "fractionable", "expected"),
    [
        (True, True, True),
        (False, True, False),
        (True, False, False),
    ],
)
def test_notional_order_eligibility_requires_tradable_and_fractionable(
    tradable,
    fractionable,
    expected,
):
    class FakeClient:
        def get_asset(self, symbol):
            return type(
                "FakeAsset",
                (),
                {
                    "tradable": tradable,
                    "fractionable": fractionable,
                },
            )()

    broker = AlpacaPaperBroker.__new__(AlpacaPaperBroker)
    broker.client = FakeClient()

    assert broker.supports_notional_order("AAPL") is expected

def test_cannot_short_unshortable_asset():

    class FakeAccount:

        shorting_enabled = True


    class FakeAsset:

        tradable = True
        shortable = False


    class FakeClient:

        def get_account(self):

            return FakeAccount()


        def get_asset(
            self,
            symbol
        ):

            return FakeAsset()


    broker = (
        AlpacaPaperBroker
        .__new__(
            AlpacaPaperBroker
        )
    )

    broker.client = FakeClient()


    assert (
        broker.can_short(
            "AAPL"
        )
        is False
    )

def test_existing_client_order_id_is_not_resubmitted():

    from src.order import Order


    class FakeExistingOrder:

        id = "existing-order-123"


    class FakeClient:

        def __init__(self):

            self.submit_count = 0


        def get_order_by_client_id(
            self,
            client_order_id
        ):

            assert (
                client_order_id
                == "qt-20260909-test123"
            )

            return FakeExistingOrder()


        def submit_order(
            self,
            order_data
        ):

            self.submit_count += 1

            raise AssertionError(
                "submit_order should not "
                "be called"
            )


    broker = (
        AlpacaPaperBroker
        .__new__(
            AlpacaPaperBroker
        )
    )

    broker.client = FakeClient()


    order = Order(
        symbol="AAPL",
        quantity=10,
        side="BUY",
        client_order_id=(
            "qt-20260909-test123"
        )
    )


    result = broker.submit_order(
        order
    )


    assert (
        result
        == "existing-order-123"
    )

    assert (
        broker.client.submit_count
        == 0
    )

def test_missing_client_order_id_submits_new_order():

    from requests import (
        HTTPError,
        Response
    )

    from alpaca.common.exceptions import (
        APIError
    )

    from src.order import Order


    class FakeSubmittedOrder:

        id = "new-order-456"


    class FakeClient:

        def __init__(self):

            self.submit_count = 0


        def get_order_by_client_id(
            self,
            client_order_id
        ):

            response = Response()

            response.status_code = 404

            http_error = HTTPError(
                response=response
            )

            raise APIError(
                (
                    '{"code":40410000,'
                    '"message":"order not found"}'
                ),
                http_error
            )


        def submit_order(
            self,
            order_data
        ):

            self.submit_count += 1

            return FakeSubmittedOrder()


    broker = (
        AlpacaPaperBroker
        .__new__(
            AlpacaPaperBroker
        )
    )

    broker.client = FakeClient()


    order = Order(
        "AAPL",
        10,
        "BUY",
        client_order_id=(
            "qt-20260909-new123"
        )
    )


    result = broker.submit_order(
        order
    )


    assert (
        result
        == "new-order-456"
    )

    assert (
        broker.client.submit_count
        == 1
    )

def test_non_404_api_error_is_not_ignored():

    from requests import (
        HTTPError,
        Response
    )

    from alpaca.common.exceptions import (
        APIError
    )

    from src.order import Order


    class FakeClient:

        def __init__(self):

            self.submit_count = 0


        def get_order_by_client_id(
            self,
            client_order_id
        ):

            response = Response()

            response.status_code = 500

            http_error = HTTPError(
                response=response
            )

            raise APIError(
                (
                    '{"code":50010000,'
                    '"message":"server error"}'
                ),
                http_error
            )


        def submit_order(
            self,
            order_data
        ):

            self.submit_count += 1

            return None


    broker = (
        AlpacaPaperBroker
        .__new__(
            AlpacaPaperBroker
        )
    )

    broker.client = FakeClient()


    order = Order(
        "AAPL",
        10,
        "BUY",
        client_order_id=(
            "qt-20260909-test"
        )
    )


    with pytest.raises(
        APIError
    ):

        broker.submit_order(
            order
        )


    assert (
        broker.client.submit_count
        == 0
    )
