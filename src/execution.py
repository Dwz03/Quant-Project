from .fill import Fill
from .order import Order

class ExecutionHandler:

    def __init__(self, slippage_rate, commission_rate):

        self.slippage_rate = slippage_rate
        self.commission_rate = commission_rate

    def execute_order(self, order, market_price):

        remaining = order.remaining_quantity()

        if remaining <= 0:
            raise ValueError("order is already filled")

        if order.side == "BUY":
            fill_price = market_price * (1 + self.slippage_rate)

        else:
            fill_price = market_price * (1 - self.slippage_rate)

        fill = Fill(order.symbol, remaining, order.side, fill_price, self.commission_rate)

        order.add_fill(fill.quantity)

        return fill

