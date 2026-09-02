from .fill import Fill
from .order import Order
from .events import FillEvent

class ExecutionHandler:

    def __init__(self, slippage_rate, commission_rate, fill_ratio = 1.0):

        self.slippage_rate = slippage_rate
        self.commission_rate = commission_rate

        if not 0 <= fill_ratio <= 1:
            raise ValueError("fill ratio must be between 0 and 1")

        self.fill_ratio = fill_ratio

    def execute_order(self, order, market_price):

        remaining = order.remaining_quantity()


        fill_quantity = int(remaining * self.fill_ratio)

        if fill_quantity == 0:
            return None

        if fill_quantity == 0:
            fill_quantity = 1

        if remaining <= 0:
            raise ValueError("order is already filled")

        if order.side == "BUY":
            fill_price = market_price * (1 + self.slippage_rate)

        else:
            fill_price = market_price * (1 - self.slippage_rate)

        fill = Fill(order.symbol, fill_quantity, order.side, fill_price, self.commission_rate)

        order.add_fill(fill.quantity)

        return fill

    def on_order_event(self, event, prices):

        order = event.order

        fill = self.execute_order(order, prices[order.symbol])

        if fill is None:
            return None

        return FillEvent(fill)

