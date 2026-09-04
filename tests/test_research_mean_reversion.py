import pandas as pd
import numpy as np
import pytest

from src.research.mean_reversion import (generate_signal, signal_to_position, run_mean_reversion,
                                         run_mean_reversion_with_history, calculate_strategy_returns)

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

def test_run_mean_reversion_with_history():

    historical_data = pd.DataFrame({
        "Close": [100, 101, 99, 102, 100]
    })

    data = pd.DataFrame({
        "Close": [101, 103, 102]
    })

    result = run_mean_reversion_with_history(
        data,
        historical_data,
        window=3,
        threshold=1
    )

    assert len(result) == len(data)
    assert pd.notna(result["zscore"].iloc[0])
    assert "strategy_return" in result.columns
    assert "equity" in result.columns

def test_strategy_return_starts_flat():

    data = pd.DataFrame({
        "Close": [100, 110, 105]
    })

    positions = pd.Series([1, 1, 0])

    result = calculate_strategy_returns(data, positions)

    assert result["strategy_return"].iloc[0] == pytest.approx(0.0)