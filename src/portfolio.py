# list查找是O(n)复杂度， dict是O(1)复杂度

# Import
from .position import Position

# Main script
class Portfolio:
    def __init__(self, cash):
        self.cash = cash
        self.positions = {}

    def add_position(self, position):
        self.positions[position.symbol] = position #我们用symbol当作key，然后进来的parameter position其实是一个object

    def get_position(self, symbol):
        return self.positions.get(symbol)

    def remove_position(self, symbol):
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

            print(symbol, value)

        return total

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
        "AAPL": 230
    }

    print(portfolio.total_market_value(prices))








