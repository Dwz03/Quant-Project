from datetime import datetime, timezone

import numpy as np
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


class MarketDataError(Exception):
    pass


MAX_QUOTE_SPREAD_PCT = 0.02
MAX_FUTURE_TIMESTAMP_SKEW_SECONDS = 15


class AlpacaMarketDataAdapter:

    def __init__(
        self,
        client,
        max_staleness_seconds=60,
        max_quote_spread_pct=MAX_QUOTE_SPREAD_PCT,
        execution_feed=DataFeed.IEX,
    ):

        self.client = client
        self.max_staleness_seconds = max_staleness_seconds
        self.max_quote_spread_pct = max_quote_spread_pct
        self.execution_feed = execution_feed


    def _timestamp_ages(self, timestamp, now):
        if timestamp is None:
            return None, None, None
        observed_at = timestamp
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        raw_age = (now - observed_at).total_seconds()
        effective_age = (
            0.0
            if -MAX_FUTURE_TIMESTAMP_SKEW_SECONDS <= raw_age < 0
            else raw_age
        )
        return raw_age, effective_age, observed_at

    def _feed_name(self):
        value = getattr(self.execution_feed, "value", self.execution_feed)
        return str(value).upper()

    @staticmethod
    def _is_entitlement_failure(error):
        if getattr(error, "status_code", None) == 403:
            return True
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "entitlement",
                "insufficient subscription",
                "subscription does not permit",
                "not subscribed",
            )
        )

    def _request_execution_data(self, loader, request):
        try:
            return loader(request)
        except Exception as error:
            if (
                self.execution_feed == DataFeed.SIP
                and self._is_entitlement_failure(error)
            ):
                raise MarketDataError(
                    "Real-time SIP market data is required for volatility_20 "
                    "V1 paper execution under the current execution-mark "
                    "architecture: requested_feed=SIP, "
                    "failure=Alpaca entitlement failure. Obtain real-time SIP "
                    "entitlement or make an explicit execution-architecture "
                    "decision; no IEX or delayed-SIP fallback was attempted."
                ) from error
            raise


    def get_latest_price(self, symbol, now=None):

        if not symbol:
            raise ValueError("symbol cannot be empty")

        request = StockLatestTradeRequest(
            symbol_or_symbols=symbol,
            feed=self.execution_feed,
        )

        trades = self._request_execution_data(
            self.client.get_stock_latest_trade,
            request,
        )

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

        raw_age, effective_age, trade_time = self._timestamp_ages(
            trade.timestamp,
            current_time,
        )

        if raw_age < -MAX_FUTURE_TIMESTAMP_SKEW_SECONDS:
            raise MarketDataError(
                f"Market data timestamp for {symbol} is materially in the "
                f"future: feed={self._feed_name()}, raw_age={raw_age:.1f}s, "
                f"effective_age={effective_age:.1f}s, "
                f"max_future_skew={MAX_FUTURE_TIMESTAMP_SKEW_SECONDS:g}s, "
                f"trade_timestamp={trade_time.isoformat()}"
            )

        if effective_age > self.max_staleness_seconds:
            raise MarketDataError(
                f"Market data for {symbol} is stale: "
                f"feed={self._feed_name()}, age={effective_age:.1f}s, "
                f"raw_age={raw_age:.1f}s, "
                f"effective_age={effective_age:.1f}s, "
                f"max_allowed={self.max_staleness_seconds:g}s, "
                f"trade_timestamp={trade_time.isoformat()}"
            )

        return float(trade.price)

    def _quote_mark(self, symbol, quote, current_time):
        if quote is None:
            return None, {
                "status": "missing",
                "raw_age": None,
                "effective_age": None,
            }

        try:
            raw_age, effective_age, quote_time = self._timestamp_ages(
                getattr(quote, "timestamp", None),
                current_time,
            )
        except (AttributeError, TypeError):
            raw_age, effective_age, quote_time = None, None, None

        try:
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
        except (AttributeError, TypeError, ValueError):
            return None, {
                "status": "invalid bid/ask",
                "raw_age": raw_age,
                "effective_age": effective_age,
            }

        if (
            not np.isfinite(bid)
            or not np.isfinite(ask)
            or bid <= 0
            or ask <= 0
            or ask < bid
        ):
            return None, {
                "status": "invalid bid/ask",
                "raw_age": raw_age,
                "effective_age": effective_age,
                "bid": bid,
                "ask": ask,
            }

        if raw_age is None:
            return None, {
                "status": "invalid timestamp",
                "raw_age": None,
                "effective_age": None,
                "bid": bid,
                "ask": ask,
            }
        if raw_age < -MAX_FUTURE_TIMESTAMP_SKEW_SECONDS:
            raise MarketDataError(
                f"Execution quote timestamp for {symbol} is materially in "
                f"the future: feed={self._feed_name()}, "
                f"raw_age={raw_age:.1f}s, "
                f"effective_age={effective_age:.1f}s, "
                f"max_future_skew={MAX_FUTURE_TIMESTAMP_SKEW_SECONDS:g}s, "
                f"quote_timestamp={quote_time.isoformat()}, "
                "trade_age=not_checked"
            )
        if effective_age > self.max_staleness_seconds:
            raise MarketDataError(
                f"Execution quote for {symbol} is stale: "
                f"feed={self._feed_name()}, quote_status=stale, "
                f"quote_age={effective_age:.1f}s, "
                f"raw_age={raw_age:.1f}s, "
                f"effective_age={effective_age:.1f}s, "
                f"max_allowed={self.max_staleness_seconds:g}s, "
                f"quote_timestamp={quote_time.isoformat()}, "
                "trade_age=not_checked"
            )

        midpoint = (bid + ask) / 2.0
        spread_pct = (ask - bid) / midpoint
        if spread_pct > self.max_quote_spread_pct:
            raise MarketDataError(
                f"Execution quote for {symbol} has excessive spread: "
                f"feed={self._feed_name()}, "
                f"quote_status=spread too wide, "
                f"quote_age={effective_age:.1f}s, "
                f"raw_age={raw_age:.1f}s, "
                f"effective_age={effective_age:.1f}s, "
                f"bid={bid:g}, ask={ask:g}, "
                f"spread_pct={spread_pct:.6f}, "
                f"max_spread_pct={self.max_quote_spread_pct:.6f}"
            )

        return midpoint, {
            "status": "valid",
            "raw_age": raw_age,
            "effective_age": effective_age,
            "bid": bid,
            "ask": ask,
            "spread_pct": spread_pct,
        }

    def _trade_fallback(self, symbol, trade, current_time, quote_status):
        quote_diagnostics = self._format_quote_diagnostics(quote_status)
        if trade is None:
            raise MarketDataError(
                f"No valid execution price for {symbol}: "
                f"feed={self._feed_name()}, {quote_diagnostics}, "
                "trade_status=missing, trade_age=unavailable, "
                f"max_allowed={self.max_staleness_seconds:g}s"
            )

        try:
            price = float(trade.price)
        except (AttributeError, TypeError, ValueError):
            price = np.nan
        try:
            raw_age, effective_age, trade_time = self._timestamp_ages(
                getattr(trade, "timestamp", None),
                current_time,
            )
        except (AttributeError, TypeError):
            raw_age, effective_age, trade_time = None, None, None
        if (
            not np.isfinite(price)
            or price <= 0
            or raw_age is None
            or raw_age < -MAX_FUTURE_TIMESTAMP_SKEW_SECONDS
            or effective_age > self.max_staleness_seconds
        ):
            if (
                effective_age is not None
                and effective_age > self.max_staleness_seconds
            ):
                status = "stale"
            elif (
                raw_age is not None
                and raw_age < -MAX_FUTURE_TIMESTAMP_SKEW_SECONDS
            ):
                status = "future timestamp"
            else:
                status = "invalid"
            raise MarketDataError(
                f"No valid execution price for {symbol}: "
                f"feed={self._feed_name()}, {quote_diagnostics}, "
                f"trade_status={status}, "
                f"trade_age={self._format_age(effective_age)}, "
                f"trade_raw_age={self._format_age(raw_age)}, "
                f"trade_effective_age={self._format_age(effective_age)}, "
                f"max_allowed={self.max_staleness_seconds:g}s, "
                f"trade_timestamp={self._format_timestamp(trade_time)}"
            )
        return price

    @classmethod
    def _format_quote_diagnostics(cls, quote_status):
        details = [
            f"quote_status={quote_status['status']}",
            "quote_age="
            f"{cls._format_age(quote_status.get('effective_age'))}",
            f"raw_age={cls._format_age(quote_status.get('raw_age'))}",
            "effective_age="
            f"{cls._format_age(quote_status.get('effective_age'))}",
        ]
        if "bid" in quote_status:
            details.append(f"bid={quote_status['bid']:g}")
        if "ask" in quote_status:
            details.append(f"ask={quote_status['ask']:g}")
        if "spread_pct" in quote_status:
            details.append(f"spread_pct={quote_status['spread_pct']:.6f}")
        return ", ".join(details)

    @staticmethod
    def _format_age(age):
        return "unavailable" if age is None else f"{age:.1f}s"

    @staticmethod
    def _format_timestamp(timestamp):
        return "unavailable" if timestamp is None else timestamp.isoformat()

    def get_execution_prices(self, symbols, now=None):
        """Return quote midpoints with a fresh-trade fallback, as one batch."""
        symbols_used = list(symbols)
        if not symbols_used:
            raise ValueError("symbols cannot be empty")
        if len(set(symbols_used)) != len(symbols_used):
            raise ValueError("symbols must be unique")

        current_time = now or datetime.now(timezone.utc)
        quote_request = StockLatestQuoteRequest(
            symbol_or_symbols=symbols_used,
            feed=self.execution_feed,
        )
        quotes = self._request_execution_data(
            self.client.get_stock_latest_quote,
            quote_request,
        ) or {}
        prices = {}
        fallback_statuses = {}
        for symbol in symbols_used:
            midpoint, status = self._quote_mark(
                symbol,
                quotes.get(symbol),
                current_time,
            )
            if midpoint is None:
                fallback_statuses[symbol] = status
            else:
                prices[symbol] = midpoint

        if fallback_statuses:
            fallback_symbols = list(fallback_statuses)
            trade_request = StockLatestTradeRequest(
                symbol_or_symbols=fallback_symbols,
                feed=self.execution_feed,
            )
            trades = self._request_execution_data(
                self.client.get_stock_latest_trade,
                trade_request,
            ) or {}
            for symbol, quote_status in fallback_statuses.items():
                prices[symbol] = self._trade_fallback(
                    symbol,
                    trades.get(symbol),
                    current_time,
                    quote_status,
                )

        return {symbol: prices[symbol] for symbol in symbols_used}

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
