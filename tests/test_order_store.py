from src.order_store import OrderStore


def test_order_persistence(tmp_path):

    filepath = tmp_path / "orders.json"

    store = OrderStore(filepath)

    store.record_order(
        client_order_id="order_123",
        broker_order_id="alpaca_456",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        filled_quantity=0,
        state="ACCEPTED"
    )

    # simulate bot restart
    new_store = OrderStore(filepath)

    order = new_store.get_order(
        "order_123"
    )

    assert order["symbol"] == "AAPL"

    assert order["quantity"] == 10

    assert order["state"] == "ACCEPTED"

    assert (
        order["broker_order_id"]
        == "alpaca_456"
    )

def test_update_order_state(tmp_path):

    filepath = tmp_path / "orders.json"

    store = OrderStore(filepath)

    store.record_order(
        client_order_id="order_123",
        broker_order_id="alpaca_456",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        state="ACCEPTED"
    )

    store.update_state(
        "order_123",
        "PARTIALLY_FILLED",
        filled_quantity=5
    )

    order = store.get_order(
        "order_123"
    )

    assert (
        order["state"]
        == "PARTIALLY_FILLED"
    )

    assert (
        order["filled_quantity"]
        == 5
    )

def test_state_survives_restart(tmp_path):

    filepath = tmp_path / "orders.json"

    store = OrderStore(filepath)

    store.record_order(
        client_order_id="order_123",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        state="SUBMITTED"
    )

    del store

    # new process / new bot
    restarted_store = OrderStore(
        filepath
    )

    assert (
        restarted_store
        .get_order("order_123")
        is not None
    )
