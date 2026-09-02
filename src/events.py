
class MarketEvent:

    def __init__(self, symbol, price):

        if symbol == "":
            raise ValueError("the symbol must not be empty")

        if not isinstance(price, (float, int)):
            raise TypeError("price must be a number")

        if price <= 0:
            raise ValueError("price must be positive")

        self.symbol = symbol
        self.price = price
        self.type = "MARKET"

class SignalEvent:

    def __init__(self, symbol, signal):

        if symbol == "":
            raise ValueError("symbol could not be empty")

        self.symbol = symbol

        if signal not in ("BUY", "SELL"):
            raise ValueError("you could only buy or sell")

        self.signal = signal

        self.type = "SIGNAL"


class OrderEvent:

    def __init__(self, order):

        self.order = order
        self.type = "ORDER"

class FillEvent:

    def __init__(self, fill):

        self.fill = fill
        self.type = "FILL"





