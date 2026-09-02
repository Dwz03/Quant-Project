from .trading_engine import TradingEngine
from .events import MarketEvent
import pandas as pd


class EventDrivenBacktester:

    def __init__(self, engine, symbol):

        self.engine = engine
        self.symbol = symbol

    def run(self, prices):

        results = []

        for price in prices:

            current_prices = {self.symbol: price}

            market_event = MarketEvent(self.symbol, price)

            self.engine.add_event(market_event)

            self.engine.run(current_prices)

            portfolio = self.engine.portfolio

            position = portfolio.get_position(self.symbol)

            if position is None:
                quantity = 0
            else:
                quantity = position.quantity

            portfolio_value = portfolio.total_value(current_prices)

            results.append({
                "Price": price,
                "Cash": portfolio.cash,
                "Quantity": quantity,
                "Portfolio_Value": portfolio_value
                })

        results = pd.DataFrame(results)

        return results
        