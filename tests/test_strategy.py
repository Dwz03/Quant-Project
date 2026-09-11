from src.strategy import (MeanReversionTradingStrategy, PairsTradingStrategy, MomentumTradingStrategy,
                          MovingAverageStrategy, MovingAverageTradingStrategy)
from src.backtest import Backtester
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

def test_moving_average_trading_strategy_bullish_target():

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


def test_moving_average_trading_strategy_bearish_short_target():

    history = pd.DataFrame({
        "AAPL": [104, 103, 102, 101]
    })

    strategy = MovingAverageTradingStrategy(
        short_window=2,
        long_window=4,
        target_weight=0.10,
        allow_short=True
    )

    assert strategy.generate_target_weights(history) == {
        "AAPL": -0.10
    }


def test_moving_average_trading_strategy_bearish_long_only_target():

    history = pd.DataFrame({
        "AAPL": [104, 103, 102, 101]
    })

    strategy = MovingAverageTradingStrategy(
        short_window=2,
        long_window=4,
        target_weight=0.10,
        allow_short=False
    )

    assert strategy.generate_target_weights(history) == {
        "AAPL": 0.0
    }


def test_moving_average_trading_strategy_equal_averages():

    history = pd.DataFrame({
        "AAPL": [1, 3, 3, 1]
    })

    strategy = MovingAverageTradingStrategy(
        short_window=2,
        long_window=4,
        target_weight=0.10
    )

    assert strategy.generate_target_weights(history) == {
        "AAPL": 0.0
    }


def test_moving_average_trading_strategy_multi_symbol_outputs():

    history = pd.DataFrame({
        "AAPL": [100, 101, 102, 103],
        "MSFT": [203, 202, 201, 200],
        "GOOG": [1, 3, 3, 1]
    })

    strategy = MovingAverageTradingStrategy(
        short_window=2,
        long_window=4,
        target_weight=0.10,
        allow_short=True
    )

    assert strategy.generate_target_weights(history) == {
        "AAPL": 0.10,
        "MSFT": -0.10,
        "GOOG": 0.0
    }


@pytest.mark.parametrize(
    ("short_window", "long_window"),
    [
        (4, 4),
        (5, 4),
        (0, 4),
        (-1, 4),
        (2, 0),
        (2, -1),
        (2.0, 4),
        (2, 4.0),
    ]
)
def test_moving_average_trading_strategy_rejects_invalid_windows(
    short_window,
    long_window
):

    with pytest.raises(ValueError):
        MovingAverageTradingStrategy(
            short_window=short_window,
            long_window=long_window
        )


@pytest.mark.parametrize(
    "target_weight",
    [-0.01, 1.01, np.nan, np.inf, "0.10", True]
)
def test_moving_average_trading_strategy_rejects_invalid_target_weight(
    target_weight
):

    with pytest.raises(ValueError, match="target_weight"):
        MovingAverageTradingStrategy(
            short_window=2,
            long_window=4,
            target_weight=target_weight
        )


def test_moving_average_trading_strategy_insufficient_history_fails_closed():

    strategy = MovingAverageTradingStrategy(
        short_window=2,
        long_window=4
    )

    with pytest.raises(ValueError, match="insufficient completed history"):
        strategy.generate_target_weights(
            pd.DataFrame({"AAPL": [100, 101, 102]})
        )


@pytest.mark.parametrize("invalid_price", [np.nan, np.inf, -np.inf])
def test_moving_average_trading_strategy_non_finite_history_fails_closed(
    invalid_price
):

    strategy = MovingAverageTradingStrategy(
        short_window=2,
        long_window=4
    )
    history = pd.DataFrame({
        "AAPL": [100, invalid_price, 102, 103]
    })

    with pytest.raises(ValueError, match="must be finite"):
        strategy.generate_target_weights(history)


def test_moving_average_legacy_and_unified_direction_parity():

    close_prices = pd.Series(
        [100, 101, 103, 102, 99, 98, 100, 104],
        name="Close"
    )
    legacy_data = pd.DataFrame({
        "Close": close_prices,
        "Return": close_prices.pct_change().fillna(0.0)
    })
    legacy_strategy = MovingAverageStrategy(
        short_window=2,
        long_window=4
    )
    legacy_signals = legacy_strategy.generate_signal(
        legacy_data
    )
    backtest = Backtester(legacy_signals)
    backtest.add_strategy_returns()

    unified_strategy = MovingAverageTradingStrategy(
        short_window=2,
        long_window=4,
        target_weight=0.10,
        allow_short=True
    )

    for execution_offset in range(4, len(close_prices)):
        completed_history = pd.DataFrame({
            "AAPL": close_prices.iloc[:execution_offset]
        })
        target = unified_strategy.generate_target_weights(
            completed_history
        )["AAPL"]
        unified_direction = int(np.sign(target))
        legacy_executed_direction = int(
            backtest.data["Position"].iloc[execution_offset]
        )

        assert unified_direction == legacy_executed_direction

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
