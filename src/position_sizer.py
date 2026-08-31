from .order import Order


class PositionSizer:

    def fixed_weight_size(self, portfolio_value, price, target_weight):

        if not isinstance(portfolio_value, (int, float)):
            raise TypeError("portfolio value must be a number")

        if not isinstance(price, (int, float)):
            raise TypeError("price must be a number")

        if not isinstance(target_weight, (int, float)):
            raise TypeError("target weight must be a number")

        if portfolio_value <= 0:
            raise ValueError("portfolio value must be positive")

        if price <= 0:
            raise ValueError("price must be positive")

        if target_weight <= 0:
            raise ValueError("target weights must be larger than 0")

        if target_weight > 1:
            raise ValueError("target weights must be smaller than 1")

        quantity = (portfolio_value * target_weight) / price

        return quantity

    def volatility_size(self, portfolio_value, price, asset_volatility,target_volatility):

        if not isinstance(portfolio_value, (int, float)):
            raise TypeError("portfolio value must be a number")

        if portfolio_value <= 0 :
            raise ValueError("portfolio value must be positive")

        if not isinstance(price, (int, float)):
            raise TypeError("price must be a number")

        if price <= 0:
            raise ValueError("price must be positive")

        if not isinstance(asset_volatility, (int, float)):
            raise TypeError("asset volatility must be a number")

        if asset_volatility <= 0:
            raise ValueError("asset volatility must be positive")

        if not isinstance(target_volatility, (int, float)):
            raise TypeError("target volatility must be a number")

        if target_volatility <= 0:
            raise ValueError("target volatility must be positive")

        weight = target_volatility / asset_volatility

        weight = min(weight, 1)

        quantity = (portfolio_value * weight) / price

        return quantity

    def risk_per_trade_size(self, portfolio_value, entry_price, stop_price, risk_fraction):

        if not isinstance(portfolio_value, (int, float)):
            raise TypeError("portfolio value must be a number")
        
        if portfolio_value <= 0 :
            raise ValueError("portfolio value must be positive")

        if not isinstance(entry_price, (int, float)):
            raise TypeError("price must be a number")
        
        if entry_price <= 0:
            raise ValueError("price must be positive")

        if not isinstance(stop_price, (int, float)):
            raise TypeError("price must be a number")
                
        if stop_price <= 0:
            raise ValueError("price must be positive")

        if not isinstance(risk_fraction, (int, float)):
            raise TypeError("risk fraction must be a number")

        if risk_fraction <= 0:
            raise ValueError("target weights must be larger than 0")
        
        if risk_fraction > 1:
            raise ValueError("target weights must not be larger than 1")

        if entry_price == stop_price:
            raise ValueError("entry price must not be equal to stop price")

        risk_budget = portfolio_value * risk_fraction

        risk_per_share = abs(entry_price - stop_price)

        risk_quantity= risk_budget / risk_per_share

        max_quantity = portfolio_value / entry_price

        quantity = min(risk_quantity, max_quantity)

        return quantity

    def generate_rebalance_order(self, symbol, target_quantity, current_quantity):

        if symbol == "":
            raise ValueError("we could not have an empty symbol name")

        if not isinstance(target_quantity, (int, float)):
            raise TypeError("target quantity must be a number")

        if target_quantity < 0:
            raise ValueError("target quantity must be positive")

        if not isinstance(current_quantity, (int, float)):
            raise TypeError("current quantity must be a number")

        if current_quantity < 0:
            raise ValueError("current quantity must be positive")

        diff = target_quantity - current_quantity

        if diff > 0:
            return Order(symbol, diff, "BUY")
        elif diff < 0:
            return Order(symbol, abs(diff), "SELL")
        else:
            return None






