from .portfolio import Portfolio
from .order import Order
from .events import OrderEvent
import math


MIN_REBALANCE_NOTIONAL = 1.0

class Rebalancer:

    def generate_notional_orders(
        self,
        target_weights,
        account_equity,
        current_market_values,
        min_rebalance_notional=MIN_REBALANCE_NOTIONAL,
    ):
        account_equity = float(account_equity)
        min_rebalance_notional = float(min_rebalance_notional)
        if not math.isfinite(account_equity) or account_equity <= 0:
            raise ValueError("account equity must be positive")
        if (
            not math.isfinite(min_rebalance_notional)
            or min_rebalance_notional < 0
        ):
            raise ValueError("minimum rebalance notional cannot be negative")

        sell_orders = []
        buy_orders = []
        all_symbols = sorted(
            set(target_weights) | set(current_market_values)
        )

        for symbol in all_symbols:
            target_weight = float(target_weights.get(symbol, 0.0))
            current_value = float(current_market_values.get(symbol, 0.0))
            if not math.isfinite(target_weight) or target_weight < 0:
                raise ValueError("notional rebalance requires long-only weights")
            if not math.isfinite(current_value) or current_value < 0:
                raise ValueError("notional rebalance requires long-only positions")

            target_notional = account_equity * target_weight
            delta_notional = target_notional - current_value
            if abs(delta_notional) <= min_rebalance_notional:
                continue

            if delta_notional > 0:
                buy_orders.append(
                    Order(
                        symbol=symbol,
                        side="BUY",
                        notional=delta_notional,
                    )
                )
            else:
                sell_notional = min(abs(delta_notional), current_value)
                if sell_notional > min_rebalance_notional:
                    sell_orders.append(
                        Order(
                            symbol=symbol,
                            side="SELL",
                            notional=sell_notional,
                        )
                    )

        return sell_orders + buy_orders

    def generate_orders(self, target_weights, portfolio, prices):

        sell_orders = []
        buy_orders = []

        portfolio_value = portfolio.total_value(prices)

        all_symbols = set(target_weights.keys()) | set(portfolio.positions.keys())

        for symbol in all_symbols :

            target_weight = target_weights.get(symbol, 0)

            target_value = target_weight * portfolio_value

            position = portfolio.get_position(symbol)

            if position is None:

                current_quantity = 0
            else:

                current_quantity = position.quantity

            current_value = prices[symbol] * current_quantity

            diff = target_value - current_value

            quantity = int(abs(diff) / prices[symbol])

            if quantity == 0:
                continue

            else:

                if diff > 0:
                    order = Order(symbol, quantity, "BUY")
                    buy_orders.append(order)

                elif diff < 0 :
                    order = Order(symbol, quantity, "SELL")
                    sell_orders.append(order)

        orders = sell_orders + buy_orders
        return orders

    def calculate_turnover(self, orders, portfolio, prices):

        portfolio_value = portfolio.total_value(prices)

        if portfolio_value <= 0:
            raise ValueError("portfolio value must be positive")

        turnover_value = 0

        for order in orders:

            symbol = order.symbol
            quantity = order.quantity
            amount = prices[symbol] * quantity
            turnover_value = turnover_value + amount

        total_turnover = turnover_value / portfolio_value

        return total_turnover

    def current_weights(self, portfolio, prices):

        portfolio_value = portfolio.total_value(prices)

        weights = {}

        for symbol, position in portfolio.positions.items():

            value = position.quantity * prices[symbol]

            weights[symbol] = value / portfolio_value

        return weights

    def on_signal_event(self, event, portfolio, prices, target_weight=0.5):

        target_weights = self.current_weights(portfolio, prices)

        if event.signal == "BUY":
            target_weights[event.symbol] = target_weight

        elif event.signal == "SELL":
            target_weights[event.symbol] = 0

        orders = self.generate_orders(target_weights, portfolio, prices)

        order_events = []

        for order in orders:
            order_events.append(OrderEvent(order))

        return order_events
