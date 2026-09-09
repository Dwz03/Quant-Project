from src.portfolio import Portfolio
from src.position import Position
from src.fill import Fill
import pytest

@pytest.fixture
def sample_portfolio():

    portfolio = Portfolio(100000)

    portfolio.buy("AAPL", 10, 100)
    portfolio.buy("MSFT", 5, 200)

    prices = {
        "AAPL": 120,
        "MSFT": 250
    }

    return portfolio, prices

def test_portfolio_initialization():

    portfolio = Portfolio(100000)

    assert portfolio.cash == 100000

    assert portfolio.initial_cash == 100000

    assert portfolio.positions == {}

def test_buy_new_position():

    portfolio = Portfolio(100000)

    portfolio.buy("AAPL", 10, 100)

    assert portfolio.cash == 99000

    assert "AAPL" in portfolio.positions

    assert portfolio.positions["AAPL"].quantity == 10

def test_buy_existing_position():

    portfolio = Portfolio(100000)

    portfolio.buy("AAPL", 10, 100)
    portfolio.buy("AAPL", 5, 120)

    assert portfolio.positions["AAPL"].quantity == 15

    assert portfolio.cash == 98400

def test_buy_insufficient_cash():

    portfolio = Portfolio(1000)

    with pytest.raises(ValueError):
        portfolio.buy("AAPL", 20, 100)

    assert portfolio.cash == 1000
    assert portfolio.positions == {}

def test_buy_invalid_price():

    portfolio = Portfolio(100000)

    with pytest.raises(ValueError):
        portfolio.buy("AAPL", 10, 0)

def test_sell_partial_position():

    portfolio = Portfolio(100000)

    portfolio.buy("AAPL", 10, 100)
    portfolio.sell("AAPL", 4, 120)

    assert portfolio.cash == 99480
    assert portfolio.positions["AAPL"].quantity == 6

def test_sell_full_position():

    portfolio = Portfolio(100000)
    portfolio.buy("AAPL", 10, 100)
    portfolio.sell("AAPL", 10, 120)

    assert "AAPL" not in portfolio.positions
    assert portfolio.cash == 100200

def test_sell_missing_position():
    portfolio = Portfolio(100000)

    with pytest.raises(ValueError):
        portfolio.sell("AAPL", 10, 100)

def test_sell_too_many_shares():

    portfolio = Portfolio(100000)
    portfolio.buy("AAPL", 10, 100)

    with pytest.raises(ValueError):
        portfolio.sell("AAPL", 20, 100)

    assert portfolio.cash == 99000
    assert portfolio.positions["AAPL"].quantity == 10

def test_total_market_value():

    portfolio = Portfolio(100000)

    portfolio.buy("AAPL", 10, 100)
    portfolio.buy("MSFT", 5, 200)

    prices = {
        "AAPL": 120,
        "MSFT": 250
        }

    assert portfolio.total_market_value(prices) == 2450

def test_total_market_value_missing_price():

    portfolio = Portfolio(100000)

    portfolio.buy("AAPL", 10, 100)
    portfolio.buy("MSFT", 5, 200)

    prices = {
        "AAPL": 120
        }

    with pytest.raises(ValueError):
        portfolio.total_market_value(prices)

def test_total_value(sample_portfolio):

    portfolio, prices = sample_portfolio

    assert portfolio.total_value(prices) == 100450

def test_pnl(sample_portfolio):

    portfolio, prices = sample_portfolio

    assert portfolio.pnl(prices) == 450


def test_return_pct(sample_portfolio):

    portfolio, prices = sample_portfolio

    assert portfolio.return_pct(prices) == pytest.approx(0.0045)

def test_net_exposure():

    portfolio = Portfolio(10000)

    portfolio.positions["AAPL"] = Position("AAPL", 40, 100)
    portfolio.positions["MSFT"] = Position("MSFT", -20, 200)

    prices = {
        "AAPL": 100,
        "MSFT": 200
    }

    result = portfolio.net_exposure(prices)

    assert result == 0

def test_exposure_ratios():

    portfolio = Portfolio(10000)

    portfolio.positions["AAPL"] = Position("AAPL", 40, 100)
    portfolio.positions["MSFT"] = Position("MSFT", -20, 200)

    prices = {
        "AAPL": 100,
        "MSFT": 200
    }

    gross_ratio = portfolio.gross_exposure_ratio(prices)
    net_ratio = portfolio.net_exposure_ratio(prices)

    assert gross_ratio == pytest.approx(0.8)
    assert net_ratio == pytest.approx(0.0)

def test_asset_exposure_ratio():

    portfolio = Portfolio(10000)

    portfolio.cash = 6000
    portfolio.positions["AAPL"] = Position("AAPL", 40, 100)

    prices = {
        "AAPL": 100
    }

    result = portfolio.asset_exposure_ratio("AAPL", prices)

    assert result == pytest.approx(0.4)

def test_sync_from_broker():

    class FakeAccount:

        cash = "8000.0"

    class FakeBrokerPosition:

        def __init__(
            self,
            symbol,
            qty,
            avg_entry_price
        ):
            self.symbol = symbol
            self.qty = qty
            self.avg_entry_price = avg_entry_price

    account = FakeAccount()

    broker_positions = [
        FakeBrokerPosition(
            "AAPL",
            "10",
            "200.0"
        ),
        FakeBrokerPosition(
            "MSFT",
            "5",
            "400.0"
        )
    ]

    portfolio = Portfolio(10000)

    portfolio.sync_from_broker(
        account,
        broker_positions
    )

    assert portfolio.cash == 8000.0

    assert (
        portfolio.get_position("AAPL").quantity
        == 10
    )

    assert (
        portfolio.get_position("AAPL").average_cost
        == 200.0
    )

    assert (
        portfolio.get_position("MSFT").quantity
        == 5
    )

def test_short_position_open_and_cover():

    portfolio = Portfolio(10000)

    # Short 10 shares @ 100
    sell_fill = Fill(
        "AAPL",
        10,
        "SELL",
        100,
        0.0
    )

    portfolio.process_fill(
        sell_fill
    )

    position = portfolio.get_position(
        "AAPL"
    )

    assert position.quantity == -10
    assert position.average_cost == 100

    # short sale proceeds increase cash
    assert portfolio.cash == 11000

    # Equity does NOT magically increase
    assert portfolio.total_value({
        "AAPL": 100
    }) == 10000


    # Cover @ 90
    buy_fill = Fill(
        "AAPL",
        10,
        "BUY",
        90,
        0.0
    )

    portfolio.process_fill(
        buy_fill
    )

    assert (
        portfolio.get_position("AAPL")
        is None
    )

    assert portfolio.cash == 10100

    assert portfolio.realised_pnl == 100