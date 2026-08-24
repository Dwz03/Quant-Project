# list查找是O(n)复杂度， dict是O(1)复杂度

# Import
from .position import Position

# Main script
class Portfolio:
    def __init__(self, cash):
        self.initial_cash = cash
        self.cash = cash
        self.positions = {}

    def _validate_trade(self, quantity, price):

        if quantity <= 0:
            raise ValueError(f"the number of shares must be positive!")

        if price <= 0:
            raise ValueError(f"the price of shares must be positive!") # This could be change later
            
    def buy(self, symbol, quantity, price):

        self._validate_trade(quantity, price)

        position = self.positions.get(symbol)

        cost = quantity * price

        if cost > self.cash:
            raise ValueError(f"we do not have enough cash to proceed the transaction") # forbid lending

        self.cash = self.cash - cost

        if position is None:
            self.positions[symbol] = Position(symbol, quantity)
        else:
            new_quantity = position.quantity + quantity
            position.update_quantity(new_quantity)

    def sell(self, symbol, quantity, price):

        position = self.positions.get(symbol)

        if position is None:
            raise ValueError(f"Cannot sell {symbol} : position does not exist!")

        if quantity > position.quantity:
            raise ValueError(f"Cannot sell {symbol} : we do not have enough quantity!")

        self._validate_trade(quantity,price)

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
    

# Test
if __name__ == "__main__":

    positions = {}
    
    positions["AAPL"] = 100
    positions["MSFT"] = 50
            
    print(positions)
    print(positions["AAPL"])
            
    print(positions.get("AAPL"))
    print(positions.get("NVDA"))
            
    positions['AAPL'] = 150
    print(positions)
            
    print("AAPL" in positions)
    print("NVDA" in positions)

    portfolio = Portfolio(100000)
    print(portfolio.cash)
    print(portfolio.positions)

    aapl = Position("AAPL", 100)
    msft = Position("MSFT", 50)

    portfolio.add_position(aapl)
    portfolio.add_position(msft)

    print(portfolio.positions["AAPL"].quantity)

    aapl_position = portfolio.get_position("AAPL")
    msft_position = portfolio.get_position("MSFT")

    print(aapl_position.quantity)
    print(msft_position.quantity)

    portfolio.remove_position("MSFT")
    print(portfolio.positions)

    aapl.update_quantity(200)

    print(aapl.quantity)
    print(portfolio.get_position("AAPL").quantity)

    print(portfolio.get_position("NVDA"))
    print(portfolio.get_position("AAPL"))

    portfolio.remove_position("NVDA")
    print("Program still running")

    portfolio.add_position(msft)
    portfolio.show_position()

    print(portfolio.positions)

    prices = {
        "AAPL": 230,
        "MSFT": 300
    }

    try:
        total = portfolio.total_market_value(prices)
        print(total)
    except ValueError as e:
        print(f"portfolio valuation failed: {e}")

    print(f"program still running")

    portfolio = Portfolio(100000)

    portfolio.buy("AAPL", 100, 200)

    print(portfolio.cash)
    print(portfolio.get_position("AAPL").quantity)

    portfolio.buy("AAPL", 50, 210)

    print(portfolio.cash)
    print(portfolio.get_position("AAPL").quantity)

    portfolio = Portfolio(100000)

    portfolio.buy("AAPL", 100, 200)

    portfolio.sell("AAPL", 20, 220)

    print(portfolio.cash)
    print(portfolio.get_position("AAPL").quantity)

    portfolio.sell("AAPL", 80, 230)

    print(portfolio.cash)
    print(portfolio.get_position("AAPL"))

    portfolio = Portfolio(100000)

    portfolio.buy("AAPL", 100, 200)
    portfolio.buy("MSFT", 50, 400)

    prices = {
        "AAPL": 220,
        "MSFT": 410
    }

    print(portfolio.cash)
    print(portfolio.total_market_value(prices))
    print(portfolio.total_value(prices))

    portfolio = Portfolio(100000)

    portfolio.buy("AAPL", 100, 200)
    portfolio.buy("MSFT", 50, 400)

    prices = {
        "AAPL": 220,
        "MSFT": 410
    }

    print("Cash:", portfolio.cash)
    print("Portfolio value:", portfolio.total_value(prices))
    print("PnL:", portfolio.pnl(prices))
    print("Return:", portfolio.return_pct(prices))

    portfolio = Portfolio(100000)

    portfolio.buy("AAPL", 100, 200)
    portfolio.buy("MSFT", 50, 400)

    prices_1 = {
        "AAPL": 220,
        "MSFT": 410
    }

    print(portfolio.total_value(prices_1))

    portfolio.sell("AAPL", 20, 220)

    print(portfolio.total_value(prices_1))

    prices_2 = {
        "AAPL": 230,
        "MSFT": 390
    }

    print(portfolio.total_value(prices_2))
    print(portfolio.pnl(prices_2))
    print(portfolio.return_pct(prices_2))











