# Import 
from .strategy import MeanReversionStrategy, MomentumStrategy

# Main Script
class TradingBot: # 我们把一个tradingbot拆成了portfolio和strategy两个模块，portfolio管position，strategy管signal

    def __init__(self,strategy,portfolio):
        self.strategy = strategy
        self.portfolio = portfolio

    def get_signal(self, prices):
        return self.strategy.generate_signal(prices)

    def show_portfolio(self):
        self.portfolio.show_position()

# Test
if __name__ == "__main__":
    momentum = MomentumStrategy(5)
    mean_reversion = MeanReversionStrategy(10)

    print(momentum.name)
    print(momentum.lookback)

    print(mean_reversion.name)
    print(mean_reversion.window)
