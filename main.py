# import
from src.position import Position
from src.portfolio import Portfolio
from src.strategy import MomentumStrategy
from src.tradingbot import TradingBot

# Main script
def main():
    
    aapl = Position("AAPL", 100)

    portfolio = Portfolio(100000)
    portfolio.add_position(aapl)

    strategy = MomentumStrategy(5)

    bot = TradingBot(strategy, portfolio)

    prices = [100, 102, 104]

    signal = bot.get_signal(prices)

    print(signal)

    bot.show_portfolio()

# Test
if __name__ == "__main__":
    main()
    print(__name__)


