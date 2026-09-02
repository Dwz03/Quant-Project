# Import
from abc import ABC, abstractmethod
from .events import SignalEvent

# Main script
class Strategy(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_signal(self, prices):
        pass

class AlwaysBuyStrategy(Strategy):

    def __init__(self):
        super().__init__("AlwaysBuys")

    def generate_signal(self, prices):
        return "BUY"

class AlwaysSellStrategy(Strategy):

    def __init__(self):
        super().__init__("AlwaysSell")

    def generate_signal(self, prices):
        return "SELL"

class MomentumStrategy(Strategy):

    def __init__(self, lookback):

        super().__init__("Momentum")
        self.lookback = lookback
        self.last_prices = {}

    def generate_signal(self, prices):

        if len(prices) < 2:
            raise ValueError("Momentum Strategy require at least 2 prices")

        if prices[-1] > prices[-2]:
            return "BUY"
        elif prices[-1] == prices[-2]:
            return "HOLD"
        else:
            return "SELL"
    def on_market_event(self, event):

        symbol = event.symbol
        price = event.price

        if symbol not in self.last_prices:
            self.last_prices[symbol] = price
            return None

        previous_price = self.last_prices[symbol]

        if price > previous_price:
            signal = SignalEvent(symbol, "BUY")

        elif price < previous_price:
            signal = SignalEvent(symbol, "SELL")

        else:
            signal = None

        self.last_prices[symbol] = price

        return signal

class MeanReversionStrategy(Strategy):

    def __init__(self, window):
        super().__init__("MeanReversion")
        self.window = window

    def generate_signal(self, prices):

        if len(prices) < self.window + 1:
            raise ValueError(f"we need at least {self.window} prices to calculate the mean revision strategy")
        
        historical_prices = prices[-self.window - 1: -1]
        average_price = sum(historical_prices) / len(historical_prices)

        if prices[-1] < average_price:
            return "BUY"
        else:
            return "SELL"

class MovingAverageStrategy(Strategy):

    def __init__(self, short_window, long_window):

        super().__init__("MovingAverage")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, data):

        data["Short_MA"] = data["Close"].rolling(window = self.short_window).mean()
        data["Long_MA"] = data["Close"].rolling(window = self.long_window).mean()

        data["Signal"] = 0

        data.loc[data["Short_MA"] > data["Long_MA"], "Signal"] = 1
        data.loc[data["Short_MA"] < data["Long_MA"], "Signal"] = -1

        return data

# Test
if __name__ == "__main__":

    buy_strategy = AlwaysBuyStrategy()
    sell_strategy = AlwaysSellStrategy()

    print(buy_strategy.generate_signal([100, 101, 102]))
    print(sell_strategy.generate_signal([100, 101, 102]))

    strategies = [
        AlwaysBuyStrategy(),
        AlwaysSellStrategy()
        ]

    for strategy in strategies:
        print(strategy.generate_signal([100, 101, 102]))

    momentum = MomentumStrategy(5)
    mean_reversion = MeanReversionStrategy(3)

    prices = [100, 101]

    strategies = [
        momentum,
        mean_reversion
    ]

    for strategy in strategies:
        print(strategy.generate_signal(prices))
