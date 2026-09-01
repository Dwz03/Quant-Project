from .order import Order
from .portfolio import Portfolio

class RiskManager:

    def __init__(self, max_position_pct, max_leverage):

        if not isinstance(max_position_pct, (int, float)):
            raise TypeError("max position percentage must be a number")

        if max_position_pct <= 0:
            raise ValueError("max position percentage must be positive")

        if max_position_pct > 1:
            raise ValueError("max position percentage could not be larger than 1")

        if not isinstance(max_leverage, (int, float)):
            raise TypeError("max leverage must be a number")

        if max_leverage <= 0:
            raise ValueError("max leverage could not be smaller or equal to 0")

        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage

    def check_max_position(self, order, portfolio, price):

        position = portfolio.get_position(order.symbol)

        if position is None:
            current_quantity = 0
        else:
            current_quantity = position.quantity

        order_quantity = order.quantity

        portfolio_equity = portfolio.total_value(price)

        max_allowed_value = self.max_position_pct * portfolio_equity

        if order.side == "BUY":

            projected_quantity = order_quantity + current_quantity
        
        elif order.side == "SELL":

            projected_quantity = current_quantity - order_quantity 

        else:
            raise ValueError("order side must be buy or sell")

        projected_position_value = projected_quantity * price[order.symbol]

        return abs(projected_position_value) <= max_allowed_value 

    def check_cash(self, order, portfolio, price):

        if order.side == "BUY":

            required_cash = order.quantity * price[order.symbol]

            return required_cash <= portfolio.cash

        elif order.side == "SELL":

            position = portfolio.get_position(order.symbol)

            if position is None:
                return False

            return order.quantity <= position.quantity

        else:
            raise ValueError("order side must be buy or sell")

    def check_leverage(self, order, portfolio, prices):

        current_gross = portfolio.gross_exposure(prices)

        position = portfolio.get_position(order.symbol)

        if position is None:
            current_quantity = 0
        else:
            current_quantity = position.quantity

        current_symbol_exposure = abs(prices[order.symbol] * current_quantity)

        if order.side == "BUY":
            projected_quantity = current_quantity + order.quantity

        elif order.side == "SELL":
            projected_quantity = current_quantity - order.quantity

        else:
            raise ValueError("order must be either buy or sell")

        projected_symbol_exposure = abs(projected_quantity * prices[order.symbol])

        projected_gross = current_gross - current_symbol_exposure + projected_symbol_exposure

        projected_leverage = projected_gross / portfolio.total_value(prices)

        return projected_leverage <= self.max_leverage

    def check_order(self, order, portfolio, prices):

        max_position_ok = self.check_max_position(order, portfolio, prices)
        cash_ok = self.check_cash(order, portfolio, prices)
        leverage_ok = self.check_leverage(order, portfolio, prices)

        return max_position_ok and cash_ok and leverage_ok





    
        











