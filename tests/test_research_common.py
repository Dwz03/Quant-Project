import pandas as pd
import numpy as np
import pytest

from src.research import (split_data, calculate_zscore, calculate_equity_curve, normalize_positions,
                          calculate_zscore_with_history)

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

def test_calculate_zscore():

    data = pd.DataFrame({
        "Close": [1, 2, 3, 4, 5]
    })

    zscore = calculate_zscore(data, 3)

    assert pd.isna(zscore.iloc[0])
    assert pd.isna(zscore.iloc[1])
    assert zscore.iloc[2] == pytest.approx(1.0)

def test_calculate_equity_curve():

    strategy_returns = pd.Series([np.nan, 0.1, 0.1])

    result = calculate_equity_curve(strategy_returns)

    assert result[0] == 1
    assert result[1] == pytest.approx(1.1)
    assert result[2] == pytest.approx(1.21)

def test_normalize_positions():

    positions = pd.DataFrame({
        "AAPL": [1, 1, 0],
        "MSFT": [-1, 0, 0],
        "GOOG": [1, -1, 0]
    })

    result = normalize_positions(positions)

    assert result.loc[0, "AAPL"] == pytest.approx(1 / 3)
    assert result.loc[0, "MSFT"] == pytest.approx(-1 / 3)
    assert result.loc[0, "GOOG"] == pytest.approx(1 / 3)

    assert result.loc[1, "AAPL"] == pytest.approx(0.5)
    assert result.loc[1, "MSFT"] == pytest.approx(0.0)
    assert result.loc[1, "GOOG"] == pytest.approx(-0.5)

    assert result.loc[2].abs().sum() == pytest.approx(0.0)

def test_calculate_zscore_with_history():

    historical_data = pd.DataFrame({
        "Close": [100, 101, 102, 103]
    })

    data = pd.DataFrame({
        "Close": [104, 105]
    })

    result = calculate_zscore_with_history(
        historical_data,
        data,
        window=3,
        column="Close"
    )

    assert len(result) == len(data)

    assert result.notna().all()

