import pytest
import pandas as pd

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

from src.market_data import (
    AlpacaMarketDataAdapter,
    MarketDataError
)


class FakeDataClient:

    def __init__(self, response):
        self.response = response

    def get_stock_latest_trade(self, request):
        return self.response


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