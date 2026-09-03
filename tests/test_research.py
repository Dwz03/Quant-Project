from src.research import (split_data, calculate_zscore, generate_signal, signal_to_position,
                          calculate_strategy_returns, calculate_equity_curve, run_mean_reversion)
import pandas as pd
import numpy as np
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

def test_calculate_zscore():

    data = pd.DataFrame({
        "Close": [1, 2, 3, 4, 5]
    })

    zscore = calculate_zscore(data, 3)

    assert pd.isna(zscore.iloc[0])
    assert pd.isna(zscore.iloc[1])
    assert zscore.iloc[2] == pytest.approx(1.0)

def test_generate_signal():

    zscore = pd.Series([np.nan, 1.5, -1.5, 0.5])

    signals = generate_signal(zscore, 1.0)

    assert signals[0] == "HOLD"
    assert signals[1] == "SELL"
    assert signals[2] == "BUY"
    assert signals[3] == "HOLD"

def test_signal_to_position():

    signals = pd.Series(["BUY", "HOLD", "SELL"])

    positions = signal_to_position(signals)

    assert positions.iloc[0] == 1
    assert positions.iloc[1] == 0
    assert positions.iloc[2] == -1

def test_calculate_strategy_returns():

    data = pd.DataFrame({
        "Close": [100, 110, 99]
    })

    positions = pd.Series([1, -1, 0])

    result = calculate_strategy_returns(data, positions)

    assert pd.isna(result["strategy_return"][0])
    assert result["strategy_return"][1] == pytest.approx(0.1)
    assert result["strategy_return"][2] == pytest.approx(0.1)

def test_calculate_equity_curve():

    strategy_returns = pd.Series([np.nan, 0.1, 0.1])

    result = calculate_equity_curve(strategy_returns)

    assert result[0] == 1
    assert result[1] == pytest.approx(1.1)
    assert result[2] == pytest.approx(1.21)

def test_end_to_end():

    data = pd.DataFrame({
        "Close": [100, 110, 99]
    })

    result = run_mean_reversion(data, window=3, threshold=1)

    assert "zscore" in result.columns
    assert "signal" in result.columns
    assert "position" in result.columns
    assert "strategy_return" in result.columns
    assert "equity" in result.columns

    assert len(result) == len(data)






    