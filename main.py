# import
import matplotlib.pyplot as plt
from src.position import Position
from src.portfolio import Portfolio
from src.strategy import MovingAverageStrategy
from src.tradingbot import TradingBot
from src.data_loader import (load_market_data, add_returns, get_data_path,
                             clean_market_data, save_market_data, load_local_market_data)
from src.backtest import Backtester

# Main script
def main():

    data = load_market_data(
    "SPY",
    "2020-01-01",
    "2025-01-01")

    strategy = MovingAverageStrategy(short_window=20,long_window=50)

    data = add_returns(data)

    data = strategy.generate_signal(data)

    backtester = Backtester(data)

    result = backtester.run()

    print(result[["Close", "Signal", "Position", "Return",
                  "Strategy_Return", "equity", "Buy_Hold_Equity"]].tail())

    change_dates = result.index[result["Signal"] != result["Signal"].shift(1)]

    print(change_dates[:5])

    print(
        result.loc[ "2020-03-10":"2020-03-18", 
                   ["Close", "Short_MA", "Long_MA", "Signal", "Position", "Return","Strategy_Return"]]
        )
# Test
if __name__ == "__main__":
    main()
    print(__name__)


