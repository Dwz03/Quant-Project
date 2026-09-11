from src.alpaca_broker import AlpacaPaperBroker


def test_alpaca_cancel_order_delegates_to_offline_client():
    class FakeTradingClient:
        def __init__(self):
            self.cancelled_order_ids = []

        def cancel_order_by_id(self, order_id):
            self.cancelled_order_ids.append(order_id)

    broker = AlpacaPaperBroker.__new__(AlpacaPaperBroker)
    broker.client = FakeTradingClient()

    result = broker.cancel_order("offline-test-order")

    assert result is True
    assert broker.client.cancelled_order_ids == ["offline-test-order"]
