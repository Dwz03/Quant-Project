from src.strategy import (MeanReversionTradingStrategy, PairsTradingStrategy, MomentumTradingStrategy,
                          MovingAverageTradingStrategy)
import pandas as pd
import pytest
import numpy as np

def test_mean_reversion_target_weights():

    history = pd.DataFrame({
        "AAPL": [100, 102, 101, 95]
    })

    strategy = MeanReversionTradingStrategy(
        window=3,
        target_weight=0.5
    )

    result = strategy.generate_target_weights(history)

    assert result == {
        "AAPL": 0.5
    }

def test_mean_reversion_normalizes_target_weights():

    history = pd.DataFrame({
        "AAPL": [100, 102, 101, 95],
        "MSFT": [200, 202, 201, 190],
        "GOOG": [150, 152, 151, 140]
    })

    strategy = MeanReversionTradingStrategy(
        window=3,
        target_weight=0.5,
        max_gross_exposure=1.0
    )

    result = strategy.generate_target_weights(history)

    gross_exposure = sum(
        abs(weight) for weight in result.values()
    )

    assert gross_exposure == pytest.approx(1.0)

def test_pairs_trading_strategy():

    history = pd.DataFrame({
        "AAPL": [
            100, 101, 102, 103,
            104, 105, 106, 90
        ],
        "MSFT": [
            200, 202, 204, 206,
            208, 210, 212, 214
        ]
    })

    strategy = PairsTradingStrategy(
        symbol_1="AAPL",
        symbol_2="MSFT",
        window=3,
        threshold=1.0,
        target_gross_exposure=0.10
    )

    weights = (
        strategy.generate_target_weights(
            history
        )
    )

    assert "AAPL" in weights
    assert "MSFT" in weights

    assert (
        abs(weights["AAPL"])
        + abs(weights["MSFT"])
        <= 0.100001
    )

    print(weights)

def test_moving_average_trading_strategy():

    history = pd.DataFrame({
        "AAPL": [
            100, 101, 102,
            103, 104
        ]
    })

    strategy = MovingAverageTradingStrategy(
        short_window=2,
        long_window=4,
        target_weight=0.10
    )

    weights = (
        strategy.generate_target_weights(
            history
        )
    )

    assert weights["AAPL"] == 0.10

def test_momentum_trading_strategy():

    history = pd.DataFrame({
        "AAPL": [
            100,
            101,
            102,
            105
        ]
    })

    strategy = MomentumTradingStrategy(
        lookback=3,
        target_weight=0.10
    )

    weights = (
        strategy.generate_target_weights(
            history
        )
    )

    assert weights["AAPL"] == 0.10