from datetime import datetime, timezone

from alpaca.data.requests import StockLatestTradeRequest


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
            symbol_or_symbols=symbol
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