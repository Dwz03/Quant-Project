from .order import Order
from .portfolio import Portfolio
import math

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

    def check_cash(
        self,
        order,
        portfolio,
        prices
    ):

        position = portfolio.get_position(
            order.symbol
        )

        if position is None:
            current_quantity = 0
        else:
            current_quantity = position.quantity


        if order.side == "SELL":

            # Selling a long position or opening /
            # increasing a short position does not
            # require cash in our simplified model.
            #
            # Short risk is controlled by:
            # max_position + leverage.
            return True


        elif order.side == "BUY":

            # --------------------------------
            # Existing short position
            # --------------------------------

            if current_quantity < 0:

                short_quantity = abs(
                    current_quantity
                )

                # Pure short cover:
                # risk-reducing order
                if order.quantity <= short_quantity:
                    return True

                # Cover short + flip into long
                opening_long_quantity = (
                    order.quantity
                    - short_quantity
                )

                required_cash = (
                    opening_long_quantity
                    * prices[order.symbol]
                )

            else:

                required_cash = (
                    order.quantity
                    * prices[order.symbol]
                )

            return (
                required_cash
                <= portfolio.cash
            )


        else:

            raise ValueError(
                "order side must be buy or sell"
            )

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

    def check_notional_order(
        self,
        order,
        account_equity,
        current_position_market_values,
        available_buying_power,
    ):
        """Risk-check one long-only notional order in dollar space."""
        if order.notional is None or order.quantity is not None:
            raise ValueError("a notional order is required")
        account_equity = float(account_equity)
        available_buying_power = float(available_buying_power)
        if not math.isfinite(account_equity) or account_equity <= 0:
            raise ValueError("account equity must be positive")
        if (
            not math.isfinite(available_buying_power)
            or available_buying_power < 0
        ):
            raise ValueError("available buying power cannot be negative")

        current_values = {
            symbol: float(value)
            for symbol, value in current_position_market_values.items()
        }
        if any(
            not math.isfinite(value) or value < 0
            for value in current_values.values()
        ):
            raise ValueError("notional risk checks require long-only positions")

        comparison_tolerance = account_equity * 1e-12
        current_value = current_values.get(order.symbol, 0.0)
        if order.side == "BUY":
            projected_value = current_value + order.notional
            cash_ok = order.notional <= (
                available_buying_power + comparison_tolerance
            )
        elif order.side == "SELL":
            if order.notional > current_value:
                return False
            projected_value = max(0.0, current_value - order.notional)
            cash_ok = True
        else:
            raise ValueError("order side must be buy or sell")

        projected_values = dict(current_values)
        projected_values[order.symbol] = projected_value
        max_position_ok = projected_value <= (
            self.max_position_pct * account_equity
            + comparison_tolerance
        )
        projected_gross = sum(abs(value) for value in projected_values.values())
        leverage_ok = projected_gross <= (
            self.max_leverage * account_equity
            + comparison_tolerance
        )
        return max_position_ok and cash_ok and leverage_ok

    def check_net_exposure(self, portfolio, prices, max_net_exposure):

        net_ratio = portfolio.net_exposure_ratio(prices)

        return abs(net_ratio) <= max_net_exposure

    def check_portfolio_exposures(self, portfolio, prices, max_gross_exposure, max_net_exposure):

        gross_exposure = portfolio.gross_exposure_ratio(prices)
        net_exposure = portfolio.net_exposure_ratio(prices)

        gross_ok = gross_exposure <= max_gross_exposure
        net_ok = abs(net_exposure) <= max_net_exposure

        return {
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "gross_ok": gross_ok,
            "net_ok": net_ok,
            "portfolio_ok": gross_ok and net_ok
        }





    
        








