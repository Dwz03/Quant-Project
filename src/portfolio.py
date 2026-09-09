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

        position = self.positions.get(fill.symbol)

        market_value = fill.market_value()

        commission = (
            market_value * fill.commission_rate
        )

        self.total_commission += commission

        # BUY = positive quantity change
        # SELL = negative quantity change
        if fill.side == "BUY":

            cash_required = market_value + commission

            if cash_required > self.cash:
                raise ValueError(
                    "we do not have enough cash"
                )

            self.cash -= cash_required

            signed_quantity = fill.quantity

        else:

            self.cash += market_value - commission

            signed_quantity = -fill.quantity


        # --------------------------------
        # No existing position
        # --------------------------------

        if position is None:

            self.positions[fill.symbol] = Position(
                fill.symbol,
                signed_quantity,
                fill.price
            )

            return


        old_quantity = position.quantity

        new_quantity = (
            old_quantity + signed_quantity
        )


        # --------------------------------
        # Same direction:
        # add to long OR add to short
        # --------------------------------

        if old_quantity * signed_quantity > 0:

            total_cost = (
                abs(old_quantity)
                * position.average_cost
                +
                abs(signed_quantity)
                * fill.price
            )

            new_average_cost = (
                total_cost / abs(new_quantity)
            )

            position.update_quantity(
                new_quantity
            )

            position.average_cost = (
                new_average_cost
            )

            return


        # --------------------------------
        # Opposite direction:
        # reduce / close / flip
        # --------------------------------

        closing_quantity = min(
            abs(old_quantity),
            abs(signed_quantity)
        )


        # Close long
        if old_quantity > 0:

            realised = (
                closing_quantity
                * (
                    fill.price
                    - position.average_cost
                )
            )

        # Cover short
        else:

            realised = (
                closing_quantity
                * (
                    position.average_cost
                    - fill.price
                )
            )


        self.realised_pnl += realised


        # Fully closed
        if new_quantity == 0:

            self._remove_position(
                fill.symbol
            )

            return


        # Still same original direction:
        # partial close
        if old_quantity * new_quantity > 0:

            position.update_quantity(
                new_quantity
            )

            # average cost stays unchanged
            return


        # --------------------------------
        # Crossed through zero
        # long -> short
        # or short -> long
        # --------------------------------

        position.update_quantity(
            new_quantity
        )

        position.average_cost = fill.price

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

    def sync_from_broker(self, account, broker_positions):

        self.cash = float(account.cash)

        self.positions = {}

        for broker_position in broker_positions:

            symbol = broker_position.symbol
            quantity = float(broker_position.qty)
            average_cost = float(
                broker_position.avg_entry_price
            )

            self.positions[symbol] = Position(
                symbol,
                quantity,
                average_cost
            )


