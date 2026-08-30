
class Fill:

    def __init__(self, symbol, quantity, side, price, commission):

        self.symbol = symbol
        self.quantity = quantity
        self.side = side
        self.price = price
        self.commission_rate = commission
        
        self._validate_fill()

    def _validate_fill(self):

        if self.symbol == "":
            raise ValueError("symbol cannot be empty")
        
        if not isinstance(self.quantity, (int, float)):
            raise TypeError("quantity must be a number")
        
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        
        if self.side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")

        if not isinstance(self.price, (int, float)):
            raise TypeError("price must be a number")

        if self.price <= 0:
            raise ValueError("price must be positive")

        if not isinstance(self.commission_rate, (int, float)):
            raise TypeError("commission rate must be a number")

        if self.commission_rate < 0:
            raise ValueError("commission rate cannot be negative")

    def market_value(self):

        return self.price * self.quantity



    