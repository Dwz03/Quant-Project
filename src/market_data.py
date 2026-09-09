from datetime import datetime, timezone
from alpaca.data.requests import (StockLatestTradeRequest,StockBarsRequest)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


class MarketDataError(Exception):
    pass


class AlpacaMarketDataAdapter:

    def __init__(self, client, max_staleness_seconds=60):

        self.client = client
        self.max_staleness_seconds = max_staleness_seconds


    def get_latest_price(self, symbol, now=None):

        if not symbol:
            raise ValueError("symbol cannot be empty")

        request = StockLatestTradeRequest(
            symbol_or_symbols=symbol,
            feed=DataFeed.IEX
        )

        trades = self.client.get_stock_latest_trade(request)

        if not trades or symbol not in trades:
            raise MarketDataError(
                f"No market data available for {symbol}"
            )

        trade = trades[symbol]

        if trade.price is None:
            raise MarketDataError(
                f"No price available for {symbol}"
            )

        if trade.timestamp is None:
            raise MarketDataError(
                f"No timestamp available for {symbol}"
            )

        current_time = now or datetime.now(timezone.utc)

        trade_time = trade.timestamp

        if trade_time.tzinfo is None:
            trade_time = trade_time.replace(tzinfo=timezone.utc)

        age = (
            current_time - trade_time
        ).total_seconds()

        if age > self.max_staleness_seconds:
            raise MarketDataError(
                f"Market data for {symbol} is stale"
            )

        return float(trade.price)

    def get_history(self, symbols, start, end, timeframe=TimeFrame.Day):

        if not symbols:
            raise ValueError("symbols cannot be empty")

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed.IEX
        )

        bars = self.client.get_stock_bars(request)

        df = bars.df

        if df.empty:
            raise MarketDataError(
                "No historical market data available"
            )

        close_prices = (
            df["close"]
            .unstack(level="symbol")
            .sort_index()
        )

        return close_prices