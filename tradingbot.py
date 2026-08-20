from abc import ABC, abstractmethod

class Position:

    def __init__(self,symbol,quantity):
        self.symbol = symbol
        self.quantity = quantity

    def market_value(self,price):
        return self.quantity * price

    def update_quantity(self,new_quantity):
        self.quantity = new_quantity

    def is_long(self):
        if self.quantity > 0:
            return True
        else:
            return False

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
            price = prices[symbol]
            value = position.market_value(price)
            total = total + value

            print(symbol, value)

        return total

class Strategy(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_signal(self, prices):
        pass

class MomentumStrategy(Strategy):

    def __init__(self, lookback):
        super().__init__("Momentum")
        self.lookback = lookback

    def generate_signal(self, prices):
        if prices[-1] > prices[-2]:
            return "Buy"
        elif prices[-1] == prices[-2]:
            return "Hold"
        else:
            return "Sell"

class MeanReversionStrategy(Strategy):

    def __init__(self, window):
        super().__init__("MeanReversion")
        self.window = window

    def generate_signal(self, prices):
        historical_prices = prices[:-1]
        average_price = sum(historical_prices) / len(historical_prices)

        if prices[-1] < average_price:
            return "Buy"
        else:
            return "Sell"

class TradingBot: # 我们把一个tradingbot拆成了portfolio和strategy两个模块，portfolio管position，strategy管signal

    def __init__(self,strategy,portfolio):
        self.strategy = strategy
        self.portfolio = portfolio

    def get_signal(self, prices):
        return self.strategy.generate_signal(prices)

    def show_portfolio(self):
        self.portfolio.show_position()

momentum = MomentumStrategy(5)
mean_reversion = MeanReversionStrategy(10)

print(momentum.name)
print(momentum.lookback)

print(mean_reversion.name)
print(mean_reversion.window)
