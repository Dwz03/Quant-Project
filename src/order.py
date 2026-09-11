import math


class Order:

    def __init__(
        self,
        symbol,
        quantity=None,
        side=None,
        client_order_id=None,
        notional=None,
    ):

        self.symbol = symbol
        self.quantity = quantity
        self.notional = notional
        self.side = side

        self.client_order_id = (
            client_order_id
        )

        self.status = "PENDING"
        self.filled_quantity = 0

        self._validate_orders()


    def _validate_orders(self):

        if self.symbol == "":
            raise ValueError(
                "symbol cannot be empty"
            )

        if (self.quantity is None) == (self.notional is None):
            raise ValueError(
                "exactly one of quantity or notional is required"
            )

        value = (
            self.quantity
            if self.quantity is not None
            else self.notional
        )
        field_name = (
            "quantity"
            if self.quantity is not None
            else "notional"
        )

        if not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be a number")

        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field_name} must be positive")

        if self.side not in (
            "BUY",
            "SELL"
        ):
            raise ValueError(
                "side must be BUY or SELL"
            )


    def remaining_quantity(self):

        if self.quantity is None:
            raise ValueError("notional orders do not have a fixed quantity")

        return (
            self.quantity
            - self.filled_quantity
        )


    def add_fill(
        self,
        quantity
    ):

        if self.quantity is None:
            raise ValueError("notional orders do not have a fixed quantity")

        new_filled_quantity = (
            self.filled_quantity
            + quantity
        )

        if (
            new_filled_quantity
            > self.quantity
        ):

            raise ValueError(
                "filled quantity exceeds "
                "order quantity"
            )

        self.filled_quantity = (
            new_filled_quantity
        )

        if (
            self.filled_quantity
            < self.quantity
        ):

            self.status = (
                "PARTIALLY_FILLED"
            )

        elif (
            self.filled_quantity
            == self.quantity
        ):

            self.status = "FILLED"

        else:

            raise ValueError(
                "filled quantity exceeds "
                "order quantity"
            )
