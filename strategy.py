from abc import ABC, abstractmethod

class Strategy(ABC):

    @abstractmethod
    def generate_signal(self, prices):
        pass

class AlwaysBuyStrategy(Strategy):

    def generate_signal(self, prices):
        return "BUY"

class AlwaysSellStrategy(Strategy):

    def generate_signal(self, prices):
        return "SELL"

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

class MomentumStrategy(Strategy):

    def generate_signal(self, prices):
        if prices[-1] > prices[-2]:
            return "Buy"
        elif prices[-1] == prices[-2]:
            return "Hold"
        else:
            return "Sell"

class MeanReversionStrategy(Strategy):

    def generate_signal(self, prices):
        historical_prices = prices[:-1]
        average_price = sum(historical_prices) / len(historical_prices)

        if prices[-1] < average_price:
            return "Buy"
        else:
            return "Sell"

momentum = MomentumStrategy()
mean_reversion = MeanReversionStrategy()

prices = [100, 102, 101, 105]

print(momentum.generate_signal(prices))
print(mean_reversion.generate_signal(prices))

strategies = [
    momentum,
    mean_reversion
]

for strategy in strategies:
    print(strategy.generate_signal(prices))