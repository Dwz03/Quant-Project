from src.research import split_data
import pandas as pd
import pytest

def test_split_data():

    data = pd.DataFrame({"price": range(100)})

    train, validation, test = split_data(data, 0.5, 0.3)

    assert len(train) == 50
    assert len(validation) == 30
    assert len(test) == 20

    assert train["price"].iloc[-1] == 49
    assert validation["price"].iloc[0] == 50
    assert validation["price"].iloc[-1] == 79
    assert test["price"].iloc[0] == 80

def test_split_data_invalid_ratio():

    data = pd.DataFrame({"price": range(100)})

    with pytest.raises(ValueError):
        split_data(data, 0.8, 0.3)

    