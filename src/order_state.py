from enum import Enum
from alpaca.trading.enums import OrderStatus


class OrderState(Enum):

    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


class InvalidOrderTransition(Exception):
    pass


class OrderStateManager:

    def __init__(self):

        self.orders = {}


    def submit(self, order_id):

        if order_id in self.orders:
            raise ValueError("order already exists")

        self.orders[order_id] = OrderState.SUBMITTED


    def get_state(self, order_id):

        if order_id not in self.orders:
            raise KeyError("order does not exist")

        return self.orders[order_id]


    def update(self, order_id, new_state):

        current_state = self.get_state(order_id)

        allowed_transitions = {

            OrderState.SUBMITTED: {
                OrderState.ACCEPTED,
                OrderState.PARTIALLY_FILLED,
                OrderState.FILLED,
                OrderState.CANCELLED
                },

            OrderState.ACCEPTED: {
                OrderState.PARTIALLY_FILLED,
                OrderState.FILLED,
                OrderState.CANCELLED
            },

            OrderState.PARTIALLY_FILLED: {
                OrderState.PARTIALLY_FILLED,
                OrderState.FILLED,
                OrderState.CANCELLED
            },

            OrderState.FILLED: set(),

            OrderState.CANCELLED: set()
        }

        if new_state not in allowed_transitions[current_state]:
            raise InvalidOrderTransition(
                f"cannot transition from "
                f"{current_state.value} to {new_state.value}"
            )

        self.orders[order_id] = new_state

def alpaca_status_to_order_state(status):

    if status in {
        OrderStatus.NEW,
        OrderStatus.ACCEPTED,
        OrderStatus.PENDING_NEW,
        OrderStatus.ACCEPTED_FOR_BIDDING
    }:
        return OrderState.ACCEPTED

    elif status == OrderStatus.PARTIALLY_FILLED:
        return OrderState.PARTIALLY_FILLED

    elif status == OrderStatus.FILLED:
        return OrderState.FILLED

    elif status in {
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED
    }:
        return OrderState.CANCELLED

    else:
        return None