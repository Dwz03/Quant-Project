from src.portfolio import Portfolio
from src.position import Position
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

def test_buy_invalid_quantity():

    portfolio = Portfolio(100000)
    
    with pytest.raises(ValueError):
        portfolio.buy("AAPL", 0, 100)


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

