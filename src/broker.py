from abc import ABC, abstractmethod
import uuid
from .fill import Fill

class Broker(ABC):

    @abstractmethod
    def submit_order(self, order):
        """
        Submit an order to the broker.

        Returns:
            order_id: broker-generated order identifier
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id):
        """
        Cancel an existing order.

        Returns:
            bool: whether the cancellation request succeeded
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id):
        """
        Get the current status of an order.

        Returns:
            str: order status
        """
        pass

class PaperBroker(Broker):

    def __init__(self, commission_rate=0.0):

        self.orders = {}
        self.commission_rate = commission_rate


    def submit_order(self, order):

        order_id = str(uuid.uuid4())

        self.orders[order_id] = {
            "order": order,
            "status": "SUBMITTED",
            "filled_quantity": 0
        }

        return order_id


    def accept_order(self, order_id):

        if order_id not in self.orders:
            raise ValueError("order does not exist")

        if self.orders[order_id]["status"] != "SUBMITTED":
            raise ValueError("order cannot be accepted")

        self.orders[order_id]["status"] = "ACCEPTED"


    def fill_order(self, order_id, quantity, price):

        if order_id not in self.orders:
            raise ValueError("order does not exist")

        order_record = self.orders[order_id]

        if order_record["status"] not in ("ACCEPTED", "PARTIALLY_FILLED"):
            raise ValueError("order cannot be filled")

        if quantity <= 0:
            raise ValueError("fill quantity must be positive")

        order = order_record["order"]

        remaining_quantity = (
            order.quantity - order_record["filled_quantity"]
        )

        if quantity > remaining_quantity:
            raise ValueError("fill quantity exceeds remaining quantity")

        fill = Fill(
            symbol=order.symbol,
            quantity=quantity,
            side=order.side,
            price=price,
            commission=self.commission_rate
        )

        order_record["filled_quantity"] += quantity

        if order_record["filled_quantity"] == order.quantity:
            order_record["status"] = "FILLED"
        else:
            order_record["status"] = "PARTIALLY_FILLED"

        return fill


    def cancel_order(self, order_id):

        if order_id not in self.orders:
            return False

        status = self.orders[order_id]["status"]

        if status in ("FILLED", "CANCELLED"):
            return False

        self.orders[order_id]["status"] = "CANCELLED"

        return True


    def get_order_status(self, order_id):

        if order_id not in self.orders:
            raise ValueError("order does not exist")

        return self.orders[order_id]["status"]


    def get_filled_quantity(self, order_id):

        if order_id not in self.orders:
            raise ValueError("order does not exist")

        return self.orders[order_id]["filled_quantity"]