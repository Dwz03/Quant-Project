class Order:

    def __init__(self, symbol, quantity, side):

        self.symbol = symbol
        self.quantity = quantity
        self.side = side
        self.status = "PENDING"
        self.filled_quantity = 0

        self._validate_orders()

    def _validate_orders(self):

        if self.symbol == "":
            raise ValueError("symbol cannot be empty")

        if not isinstance(self.quantity, (int, float)):
            raise TypeError("quantity must be a number")

        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

        if self.side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")

    def remaining_quantity(self):
        return self.quantity - self.filled_quantity

    def add_fill(self, quantity):

        new_filled_quantity = self.filled_quantity + quantity

        if new_filled_quantity > self.quantity:
            raise ValueError("filled quantity exceeds order quantity")

        self.filled_quantity = new_filled_quantity

        if self.filled_quantity < self.quantity:
            self.status = "PARTIALLY_FILLED"

        elif self.filled_quantity == self.quantity:
            self.status = "FILLED"

        else:
            raise ValueError("filled quantity exceeds order quantity")
    