import numpy as np
import pytest
import pandas as pd

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

from alpaca.data.enums import DataFeed

from src.market_data import (
    AlpacaMarketDataAdapter,
    MAX_FUTURE_TIMESTAMP_SKEW_SECONDS,
    MarketDataError
)


class FakeDataClient:

    def __init__(self, response):
        self.response = response

    def get_stock_latest_trade(self, request):
        return self.response


class FakeExecutionClient:

    def __init__(self, quotes=None, trades=None):
        self.quotes = quotes or {}
        self.trades = trades or {}
        self.quote_requests = []
        self.trade_requests = []

    def get_stock_latest_quote(self, request):
        self.quote_requests.append(request)
        return self.quotes

    def get_stock_latest_trade(self, request):
        self.trade_requests.append(request)
        return self.trades


def quote(bid, ask, timestamp):
    return SimpleNamespace(
        bid_price=bid,
        ask_price=ask,
        timestamp=timestamp,
    )


def test_get_latest_price():

    now = datetime.now(timezone.utc)

    fake_trade = SimpleNamespace(
        price=200,
        timestamp=now
    )

    client = FakeDataClient({
        "AAPL": fake_trade
    })

    adapter = AlpacaMarketDataAdapter(client)

    price = adapter.get_latest_price(
        "AAPL",
        now=now
    )

    assert price == 200


def test_missing_market_data():

    now = datetime.now(timezone.utc)

    client = FakeDataClient({})

    adapter = AlpacaMarketDataAdapter(client)

    with pytest.raises(MarketDataError):

        adapter.get_latest_price(
            "AAPL",
            now=now
        )


def test_stale_market_data():

    now = datetime.now(timezone.utc)

    old_time = now - timedelta(seconds=120)

    fake_trade = SimpleNamespace(
        price=200,
        timestamp=old_time
    )

    client = FakeDataClient({
        "AAPL": fake_trade
    })

    adapter = AlpacaMarketDataAdapter(
        client,
        max_staleness_seconds=60
    )

    with pytest.raises(MarketDataError):

        adapter.get_latest_price(
            "AAPL",
            now=now
        )


def test_recent_market_data():

    now = datetime.now(timezone.utc)

    recent_time = now - timedelta(seconds=10)

    fake_trade = SimpleNamespace(
        price=200,
        timestamp=recent_time
    )

    client = FakeDataClient({
        "AAPL": fake_trade
    })

    adapter = AlpacaMarketDataAdapter(
        client,
        max_staleness_seconds=60
    )

    price = adapter.get_latest_price(
        "AAPL",
        now=now
    )

    assert price == 200


def test_generic_default_accepts_trade_59_seconds_old():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    trade = SimpleNamespace(
        price=200,
        timestamp=now - timedelta(seconds=59),
    )

    adapter = AlpacaMarketDataAdapter(FakeDataClient({"SBUX": trade}))

    assert adapter.get_latest_price("SBUX", now=now) == 200


def test_trade_between_strict_and_paper_thresholds_uses_profile_policy():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    trade = SimpleNamespace(
        price=200,
        timestamp=now - timedelta(seconds=83.4),
    )
    client = FakeDataClient({"SBUX": trade})

    strict_adapter = AlpacaMarketDataAdapter(client)
    paper_adapter = AlpacaMarketDataAdapter(
        client,
        max_staleness_seconds=300,
    )

    with pytest.raises(MarketDataError, match="SBUX is stale"):
        strict_adapter.get_latest_price("SBUX", now=now)
    assert paper_adapter.get_latest_price("SBUX", now=now) == 200


def test_paper_threshold_fails_over_300_seconds_with_diagnostics():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    trade_timestamp = now - timedelta(seconds=301.2)
    trade = SimpleNamespace(price=200, timestamp=trade_timestamp)
    adapter = AlpacaMarketDataAdapter(
        FakeDataClient({"SBUX": trade}),
        max_staleness_seconds=300,
    )

    with pytest.raises(MarketDataError) as error:
        adapter.get_latest_price("SBUX", now=now)

    message = str(error.value)
    assert "SBUX" in message
    assert "age=301.2s" in message
    assert "max_allowed=300s" in message
    assert f"trade_timestamp={trade_timestamp.isoformat()}" in message


def test_execution_price_uses_fresh_quote_midpoint_without_trade_request():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    client = FakeExecutionClient(
        quotes={"AAPL": quote(99.0, 101.0, now - timedelta(seconds=12))},
    )
    adapter = AlpacaMarketDataAdapter(client, max_staleness_seconds=300)

    result = adapter.get_execution_prices(["AAPL"], now=now)

    assert result == {"AAPL": 100.0}
    assert len(client.quote_requests) == 1
    assert client.quote_requests[0].feed == DataFeed.IEX
    assert client.trade_requests == []


def test_sip_execution_profile_requests_sip_and_uses_quote_midpoint():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    client = FakeExecutionClient(
        quotes={"AAPL": quote(99.0, 101.0, now)},
    )
    adapter = AlpacaMarketDataAdapter(
        client,
        max_staleness_seconds=300,
        execution_feed=DataFeed.SIP,
    )

    assert adapter.get_execution_prices(["AAPL"], now=now) == {
        "AAPL": 100.0,
    }
    assert len(client.quote_requests) == 1
    assert client.quote_requests[0].feed == DataFeed.SIP
    assert client.trade_requests == []


def test_sip_entitlement_failure_is_distinct_and_never_retries_iex():
    class EntitlementError(Exception):
        status_code = 403

    class EntitlementClient:
        def __init__(self):
            self.quote_requests = []
            self.trade_requests = []

        def get_stock_latest_quote(self, request):
            self.quote_requests.append(request)
            raise EntitlementError("forbidden")

        def get_stock_latest_trade(self, request):
            self.trade_requests.append(request)
            raise AssertionError("quote entitlement failure must stop the batch")

    client = EntitlementClient()
    adapter = AlpacaMarketDataAdapter(
        client,
        max_staleness_seconds=300,
        execution_feed=DataFeed.SIP,
    )

    with pytest.raises(MarketDataError) as error:
        adapter.get_execution_prices(
            ["AAPL"],
            now=datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc),
        )

    message = str(error.value)
    assert "Real-time SIP market data is required" in message
    assert "requested_feed=SIP" in message
    assert "entitlement failure" in message
    assert "no IEX or delayed-SIP fallback" in message
    assert [request.feed for request in client.quote_requests] == [DataFeed.SIP]
    assert client.trade_requests == []


def test_small_future_quote_clock_skew_is_treated_as_zero_age():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    client = FakeExecutionClient(
        quotes={"AAPL": quote(99.0, 101.0, now + timedelta(seconds=6))},
    )
    adapter = AlpacaMarketDataAdapter(
        client,
        max_staleness_seconds=300,
        execution_feed=DataFeed.SIP,
    )

    assert adapter.get_execution_prices(["AAPL"], now=now) == {
        "AAPL": 100.0,
    }
    assert MAX_FUTURE_TIMESTAMP_SKEW_SECONDS == 15


def test_material_future_quote_timestamp_fails_closed():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    client = FakeExecutionClient(
        quotes={"AAPL": quote(99.0, 101.0, now + timedelta(seconds=30))},
        trades={"AAPL": SimpleNamespace(price=100.0, timestamp=now)},
    )
    adapter = AlpacaMarketDataAdapter(
        client,
        max_staleness_seconds=300,
        execution_feed=DataFeed.SIP,
    )

    with pytest.raises(MarketDataError) as error:
        adapter.get_execution_prices(["AAPL"], now=now)

    message = str(error.value)
    assert "AAPL" in message
    assert "feed=SIP" in message
    assert "raw_age=-30.0s" in message
    assert "effective_age=-30.0s" in message
    assert "max_future_skew=15s" in message
    assert client.trade_requests == []


def test_execution_quote_freshness_uses_fixed_paper_threshold():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    fresh_client = FakeExecutionClient(
        quotes={"AAPL": quote(99.5, 100.5, now - timedelta(seconds=299))},
    )
    stale_client = FakeExecutionClient(
        quotes={"AAPL": quote(99.5, 100.5, now - timedelta(seconds=301))},
        trades={
            "AAPL": SimpleNamespace(price=100.0, timestamp=now),
        },
    )

    fresh = AlpacaMarketDataAdapter(
        fresh_client,
        max_staleness_seconds=300,
        execution_feed=DataFeed.SIP,
    )
    stale = AlpacaMarketDataAdapter(
        stale_client,
        max_staleness_seconds=300,
        execution_feed=DataFeed.SIP,
    )

    assert fresh.get_execution_prices(["AAPL"], now=now) == {"AAPL": 100.0}
    with pytest.raises(MarketDataError, match="quote_age=301.0s"):
        stale.get_execution_prices(["AAPL"], now=now)
    assert fresh_client.quote_requests[0].feed == DataFeed.SIP
    assert stale_client.quote_requests[0].feed == DataFeed.SIP
    assert stale_client.trade_requests == []


@pytest.mark.parametrize(
    ("bid", "ask"),
    [
        (0.0, 100.0),
        (99.0, 0.0),
        (101.0, 100.0),
        (np.nan, 100.0),
        (99.0, np.inf),
    ],
)
def test_invalid_quote_bid_ask_fails_without_fresh_trade(bid, ask):
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    client = FakeExecutionClient(
        quotes={"AAPL": quote(bid, ask, now)},
    )
    adapter = AlpacaMarketDataAdapter(client, max_staleness_seconds=300)

    with pytest.raises(MarketDataError, match="quote_status=invalid bid/ask"):
        adapter.get_execution_prices(["AAPL"], now=now)


def test_invalid_quote_timestamp_fails_without_fresh_trade():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    client = FakeExecutionClient(
        quotes={"AAPL": quote(99.0, 101.0, None)},
    )
    adapter = AlpacaMarketDataAdapter(client, max_staleness_seconds=300)

    with pytest.raises(MarketDataError, match="quote_status=invalid timestamp"):
        adapter.get_execution_prices(["AAPL"], now=now)


def test_quote_above_two_percent_spread_fails_with_diagnostics():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    client = FakeExecutionClient(
        quotes={
            "AAPL": quote(98.0, 102.0, now + timedelta(seconds=6)),
        },
        trades={"AAPL": SimpleNamespace(price=100.0, timestamp=now)},
    )
    adapter = AlpacaMarketDataAdapter(
        client,
        max_staleness_seconds=300,
        execution_feed=DataFeed.SIP,
    )

    with pytest.raises(MarketDataError) as error:
        adapter.get_execution_prices(["AAPL"], now=now)

    message = str(error.value)
    assert "AAPL" in message
    assert "feed=SIP" in message
    assert "raw_age=-6.0s" in message
    assert "effective_age=0.0s" in message
    assert "quote_status=spread too wide" in message
    assert "bid=98" in message
    assert "ask=102" in message
    assert "spread_pct=0.040000" in message
    assert client.trade_requests == []


@pytest.mark.parametrize(
    "quotes",
    [
        {},
        {"AAPL": quote(0.0, 100.0, None)},
    ],
)
def test_missing_or_invalid_quote_falls_back_to_fresh_trade(quotes):
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    client = FakeExecutionClient(
        quotes=quotes,
        trades={
            "AAPL": SimpleNamespace(
                price=100.25,
                timestamp=now - timedelta(seconds=20),
            ),
        },
    )
    adapter = AlpacaMarketDataAdapter(
        client,
        max_staleness_seconds=300,
        execution_feed=DataFeed.SIP,
    )

    assert adapter.get_execution_prices(["AAPL"], now=now) == {
        "AAPL": 100.25,
    }
    assert len(client.trade_requests) == 1
    assert client.quote_requests[0].feed == DataFeed.SIP
    assert client.trade_requests[0].feed == DataFeed.SIP


def test_missing_quote_with_stale_trade_fails_closed_with_both_ages():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    client = FakeExecutionClient(
        trades={
            "AAPL": SimpleNamespace(
                price=100.25,
                timestamp=now - timedelta(seconds=301),
            ),
        },
    )
    adapter = AlpacaMarketDataAdapter(client, max_staleness_seconds=300)

    with pytest.raises(MarketDataError) as error:
        adapter.get_execution_prices(["AAPL"], now=now)

    message = str(error.value)
    assert "AAPL" in message
    assert "quote_status=missing" in message
    assert "quote_age=unavailable" in message
    assert "trade_status=stale" in message
    assert "trade_age=301.0s" in message


def test_all_88_execution_quotes_are_loaded_in_one_batch():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    symbols = [f"S{number:02d}" for number in range(88)]
    quotes = {
        symbol: quote(99.5, 100.5, now)
        for symbol in symbols
    }
    client = FakeExecutionClient(quotes=quotes)
    adapter = AlpacaMarketDataAdapter(client, max_staleness_seconds=300)

    result = adapter.get_execution_prices(symbols, now=now)

    assert result == {symbol: 100.0 for symbol in symbols}
    assert len(client.quote_requests) == 1
    assert client.trade_requests == []


def test_one_unpriced_symbol_fails_the_entire_execution_price_batch():
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    symbols = [f"S{number:02d}" for number in range(88)]
    missing_symbol = symbols[-1]
    quotes = {
        symbol: quote(99.5, 100.5, now)
        for symbol in symbols[:-1]
    }
    client = FakeExecutionClient(quotes=quotes)
    adapter = AlpacaMarketDataAdapter(client, max_staleness_seconds=300)

    with pytest.raises(MarketDataError, match=missing_symbol):
        adapter.get_execution_prices(symbols, now=now)

    assert len(client.quote_requests) == 1
    assert len(client.trade_requests) == 1


def test_get_history():

    class FakeBars:

        @property
        def df(self):

            index = pd.MultiIndex.from_tuples(
                [
                    ("AAPL", pd.Timestamp("2026-09-01")),
                    ("AAPL", pd.Timestamp("2026-09-02")),
                    ("MSFT", pd.Timestamp("2026-09-01")),
                    ("MSFT", pd.Timestamp("2026-09-02")),
                ],
                names=["symbol", "timestamp"]
            )

            return pd.DataFrame(
                {
                    "close": [
                        100,
                        101,
                        200,
                        202
                    ]
                },
                index=index
            )

    class FakeClient:

        def get_stock_bars(self, request):

            return FakeBars()

    adapter = AlpacaMarketDataAdapter(
        FakeClient()
    )

    result = adapter.get_history(
        ["AAPL", "MSFT"],
        start=datetime(2026, 9, 1),
        end=datetime(2026, 9, 3)
    )

    assert list(result.columns) == [
        "AAPL",
        "MSFT"
    ]

    assert result.iloc[0]["AAPL"] == 100

    assert result.iloc[1]["MSFT"] == 202
