from src.position import Position
import pytest

def test_market_value():

    symbol = "AAPL"
    quantity = 10
    price = 100
    average_cost = 100

    position = Position(symbol, quantity, average_cost)

    value = position.market_value(price)

    assert value == 1000

def test_empty_symbol():

    with pytest.raises(ValueError):

        Position("", 10, 100)

def test_invalid_quantity():

    with pytest.raises(TypeError):

        Position("AAPL", "10", 100)

def test_update_quantity():

    position = Position("AAPL", 10, 100)

    position.update_quantity(20)

    assert position.quantity == 20

def test_update_quantity_invalid():

    position = Position("AAPL", 10, 100)

    with pytest.raises(TypeError):
        position.update_quantity("20")

def test_is_long_positive():
    position = Position("AAPL", 10, 100)
    assert position.is_long()

# def test_is_long_negative():
#    position = Position("AAPL", -10, 100)
#    assert not position.is_long()


# def test_is_long_zero():
#     position = Position("AAPL", 0, 100)
#     assert not position.is_long()