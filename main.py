# import
import matplotlib.pyplot as plt
from src.position import Position
from src.portfolio import Portfolio
from src.strategy import MovingAverageStrategy
from src.tradingbot import TradingBot
from src.data_loader import (load_market_data, add_returns, get_data_path,
                             clean_market_data, save_market_data, load_local_market_data)
from src.backtest import (add_strategy_returns, add_equity_curve, add_buy_hold_equity)

# Main script
def main():

    data = load_market_data(
    "AAPL",
    "2020-01-01",
    "2025-01-01")

    strategy = MovingAverageStrategy(short_window=20,long_window=50)

    data = strategy.generate_signal(data)

    print(data[["Close", "Short_MA", "Long_MA", "Signal"]].tail(20))

    print(data["Signal"].value_counts())

# Test
if __name__ == "__main__":
    main()
    print(__name__)


