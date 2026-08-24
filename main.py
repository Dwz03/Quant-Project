# import
import matplotlib.pyplot as plt
from src.position import Position
from src.portfolio import Portfolio
from src.strategy import MomentumStrategy
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

    data = add_returns(data)

    data["Signal"] = 0

    data.loc[data["Return"] > 0, "Signal"] = 1
    data.loc[data["Return"] < 0, "Signal"] = -1

    data = add_strategy_returns(data)

    data = add_equity_curve(data)

    data = add_buy_hold_equity(data)

    print(data.tail())

    plt.plot(data.index, data["equity"], label="Strategy")
    plt.plot(data.index, data["Buy_Hold_Equity"], label="Buy & Hold")

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Strategy vs Buy & Hold")

    plt.show()

# Test
if __name__ == "__main__":
    main()
    print(__name__)


