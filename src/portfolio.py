# list查找是O(n)复杂度， dict是O(1)复杂度

# Import
from .position import Position
from .fill import Fill

# Main script
class Portfolio:
    def __init__(self, cash):
        self.initial_cash = cash
        self.cash = cash
        self.positions = {}
        self.realised_pnl = 0
        self.total_commission = 0

    def _validate_trade(self, price):

        if price <= 0:
            raise ValueError(f"the price of shares must be positive!") # This could be change later
            
    def buy(self, symbol, quantity, price):

        self._validate_trade(price)

        position = self.positions.get(symbol)

        cost = quantity * price

        if cost > self.cash:
            raise ValueError(f"we do not have enough cash to proceed the transaction") # forbid lending

        self.cash = self.cash - cost

        if position is None:
            self.positions[symbol] = Position(symbol, quantity, price)
        else:
            new_quantity = position.quantity + quantity
            total_cost = position.quantity * position.average_cost + quantity * price
            new_average_cost = total_cost / new_quantity
            position.update_quantity(new_quantity)
            position.average_cost = new_average_cost

    def sell(self, symbol, quantity, price):

        position = self.positions.get(symbol)

        if position is None:
            raise ValueError(f"Cannot sell {symbol} : position does not exist!")

        if quantity > position.quantity:
            raise ValueError(f"Cannot sell {symbol} : we do not have enough quantity!")

        self._validate_trade(price)

        gain = quantity * price
        self.cash = self.cash + gain

        new_quantity = position.quantity - quantity
        position.update_quantity(new_quantity)

        if position.quantity == 0:
            self._remove_position(symbol)

    def _add_position(self, position):
        self.positions[position.symbol] = position #我们用symbol当作key，然后进来的parameter position其实是一个object

    def get_position(self, symbol):
        return self.positions.get(symbol)

    def _remove_position(self, symbol):
        self.positions.pop(symbol, None)

    def show_position(self):
        for symbol, position in self.positions.items():
            print(symbol, position.quantity)

    def total_market_value(self, prices):
        total = 0
        for symbol, position in self.positions.items():

            if symbol not in prices:
                raise ValueError(f"Missing price for {symbol}")
            
            price = prices[symbol]
            value = position.market_value(price)
            total = total + value

        return total

    def total_value(self, prices):

        market_price = self.total_market_value(prices)
        return market_price + self.cash

    def pnl(self, prices):

        return self.total_value(prices) - self.initial_cash

    def return_pct(self, prices):

        return self.pnl(prices) / self.initial_cash

    def process_fill(self, fill):

        if fill.side == "BUY":

            cost = fill.market_value() * (1 + fill.commission_rate)

            if cost > self.cash:

                raise ValueError("we do not have enough cash")

            self.cash = self.cash - cost

            position = self.positions.get(fill.symbol)

            commission = (fill.market_value() * fill.commission_rate)

            self.total_commission += commission

            if position is None:
                self.positions[fill.symbol] = Position(fill.symbol, fill.quantity, fill.price)

            else:
                position.add_quantity(fill.quantity, fill.price)

        else:

            position = self.positions.get(fill.symbol)

            if position is None:
                raise ValueError("we do not have this asset")

            if position.quantity < fill.quantity:
                raise ValueError("we do not have enough quantity")
            
            commission = (fill.market_value() * fill.commission_rate)

            self.total_commission += commission

            proceed = fill.market_value() * (1 - fill.commission_rate)

            self.cash = self.cash + proceed 

            realised = position.reduce_quantity(fill.quantity, fill.price)

            self.realised_pnl += realised

            if position.quantity == 0:
                self._remove_position(fill.symbol)

    def total_unrealised_pnl(self, prices):

        total = 0

        for symbol, position in self.positions.items():

            if symbol not in prices:
                raise ValueError(f"Missing price for {symbol}")
                        
            price = prices[symbol]
            value = position.unrealised_pnl(price)
            total = total + value

        return total

    def gross_exposure(self, prices):

        total = 0
        for symbol, position in self.positions.items():

            if symbol not in prices:
                raise ValueError(f"missing price for {symbol}")

            price = prices[symbol]
            value = position.quantity  * price
            total = total + abs(value)
        return total

    def net_exposure(self, prices):

        total = 0
        for symbol, position in self.positions.items():

            if symbol not in prices:
                raise ValueError(f"missing price for {symbol}")

            price = prices[symbol]
            value = position.quantity * price
            total = total + value
        return total

    def gross_exposure_ratio(self, prices):

        equity = self.total_value(prices)

        if equity <= 0:
            raise ValueError("portfolio equity must be positive")

        return self.gross_exposure(prices) / equity


    def net_exposure_ratio(self, prices):

        equity = self.total_value(prices)

        if equity <= 0:
            raise ValueError("portfolio equity must be positive")

        return self.net_exposure(prices) / equity

    def asset_exposure_ratio(self, symbol, prices):

        if symbol not in prices:
            raise ValueError(f"missing price for {symbol}")

        position = self.get_position(symbol)

        if position is None:
            return 0

        equity = self.total_value(prices)

        if equity <= 0:
            raise ValueError("portfolio equity must be positive")

        exposure = abs(position.quantity * prices[symbol])

        return exposure / equity


