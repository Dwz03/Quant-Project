import pytest

from src.order_state import (
    OrderState,
    OrderStateManager,
    InvalidOrderTransition
)


def test_order_normal_lifecycle():

    manager = OrderStateManager()

    manager.submit("order_1")

    assert manager.get_state("order_1") == OrderState.SUBMITTED

    manager.update(
        "order_1",
        OrderState.ACCEPTED
    )

    assert manager.get_state("order_1") == OrderState.ACCEPTED

    manager.update(
        "order_1",
        OrderState.PARTIALLY_FILLED
    )

    assert manager.get_state("order_1") == OrderState.PARTIALLY_FILLED

    manager.update(
        "order_1",
        OrderState.FILLED
    )

    assert manager.get_state("order_1") == OrderState.FILLED

def test_order_cancelled():

    manager = OrderStateManager()

    manager.submit("order_1")

    manager.update(
        "order_1",
        OrderState.ACCEPTED
    )

    manager.update(
        "order_1",
        OrderState.CANCELLED
    )

    assert manager.get_state("order_1") == OrderState.CANCELLED

def test_filled_order_cannot_change():

    manager = OrderStateManager()

    manager.submit("order_1")

    manager.update(
        "order_1",
        OrderState.ACCEPTED
    )

    manager.update(
        "order_1",
        OrderState.FILLED
    )

    with pytest.raises(InvalidOrderTransition):

        manager.update(
            "order_1",
            OrderState.ACCEPTED
        )

def test_order_can_skip_intermediate_state():

    manager = OrderStateManager()

    manager.submit("order_1")

    manager.update(
        "order_1",
        OrderState.FILLED
    )

    assert (
        manager.get_state("order_1")
        == OrderState.FILLED
    )