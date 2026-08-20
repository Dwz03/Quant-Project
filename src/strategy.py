# Import
from abc import ABC, abstractmethod

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

    def generate_signal(self, prices):

        if len(prices) < 2:
            raise ValueError("Momentum Strategy require at least 2 prices")

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

        if len(prices) < self.window + 1:
            raise ValueError(f"we need at least {self.window} prices to calculate the mean revision strategy")
        
        historical_prices = prices[-self.window - 1: -1]
        average_price = sum(historical_prices) / len(historical_prices)

        if prices[-1] < average_price:
            return "Buy"
        else:
            return "Sell"

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
