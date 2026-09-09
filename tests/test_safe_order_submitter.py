from types import SimpleNamespace

from src.safe_order_submitter import SafeOrderSubmitter


class FakeClient:

    def __init__(self):
        self.submit_count = 0

    def submit_order(self, order_request):

        self.submit_count += 1

        return SimpleNamespace(
            id="alpaca_1",
            client_order_id=order_request.client_order_id
        )


def test_normal_submission():

    client = FakeClient()

    submitter = SafeOrderSubmitter(client)

    def create_request(client_order_id):

        return SimpleNamespace(
            symbol="AAPL",
            quantity=10,
            client_order_id=client_order_id
        )

    result = submitter.submit(
        create_request,
        "order_123"
    )

    assert result.client_order_id == "order_123"
    assert client.submit_count == 1

class TimeoutButSubmittedClient:

    def __init__(self):

        self.submit_count = 0
        self.saved_order = None


    def submit_order(self, order_request):

        self.submit_count += 1

        self.saved_order = SimpleNamespace(
            id="alpaca_1",
            client_order_id=order_request.client_order_id
        )

        raise TimeoutError()


    def get_order_by_client_id(self, client_order_id):

        if (
            self.saved_order is not None
            and
            self.saved_order.client_order_id
            == client_order_id
        ):
            return self.saved_order

        raise KeyError("order not found")

def test_timeout_does_not_duplicate_order():

    client = TimeoutButSubmittedClient()

    submitter = SafeOrderSubmitter(
        client,
        max_retries=3
    )

    def create_request(client_order_id):

        return SimpleNamespace(
            symbol="AAPL",
            quantity=10,
            client_order_id=client_order_id
        )

    result = submitter.submit(
        create_request,
        "order_123"
    )

    assert result.id == "alpaca_1"

    assert client.submit_count == 1

class FailOnceClient:

    def __init__(self):

        self.submit_count = 0


    def submit_order(self, order_request):

        self.submit_count += 1

        if self.submit_count == 1:
            raise TimeoutError()

        return SimpleNamespace(
            id="alpaca_1",
            client_order_id=order_request.client_order_id
        )


    def get_order_by_client_id(self, client_order_id):

        raise KeyError("order not found")

def test_retry_if_order_really_not_submitted():

    client = FailOnceClient()

    submitter = SafeOrderSubmitter(
        client,
        max_retries=3
    )

    def create_request(client_order_id):

        return SimpleNamespace(
            symbol="AAPL",
            quantity=10,
            client_order_id=client_order_id
        )

    result = submitter.submit(
        create_request,
        "order_123"
    )

    assert result.id == "alpaca_1"

    assert client.submit_count == 2