import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations
import pytest

from src.research.pairs import (estimate_beta, calculate_spread, estimate_hedge_ratio, generate_pair_positions,
                                calculate_pair_weights, calculate_pair_returns, check_spread_stationarity,
                                run_pairs_trading, run_pairs_trading_with_history, generate_pairs,
                                screen_pairs, select_pairs)

def test_estimate_beta():

    data = pd.DataFrame({
        "symbol_2" : [100, 120, 150],
        "symbol_1" : [200, 240, 300]
    })

    result = estimate_beta(data)

    assert result == pytest.approx(2)

def test_calculate_spread():

    data = pd.DataFrame({
        "symbol_1": [210, 252, 288],
        "symbol_2": [100, 121, 139]
    })

    result = calculate_spread(data, 2.0, 10.0)

    assert result["spread"].iloc[0] == pytest.approx(0)
    assert result["spread"].iloc[1] == pytest.approx(0)
    assert result["spread"].iloc[2] == pytest.approx(0)

def test_hedge_ratio():

    data = pd.DataFrame({
        "symbol_1": [210, 252, 288, 372, 408],
        "symbol_2": [100, 121, 139, 181, 199]
    })

    result = estimate_hedge_ratio(data)

    assert result["alpha"] == pytest.approx(10)
    assert result["beta"] == pytest.approx(2)

def test_generate_pair_position():

    zscore = pd.Series([0, 2, -2])
    threshold = 1
    beta = 0.5

    result = generate_pair_positions(zscore, threshold, beta)

    assert result["position_1"].iloc[0] == 0
    assert result["position_2"].iloc[0] == 0
    assert result["position_1"].iloc[1] == -1
    assert result["position_2"].iloc[1] == pytest.approx(0.5)
    assert result["position_1"].iloc[2] == 1
    assert result["position_2"].iloc[2] == pytest.approx(-0.5)

def test_calculate_pair_weights():

    data = pd.DataFrame({
        "symbol_1": [100, 110],
        "symbol_2": [200, 180]
    })

    positions = pd.DataFrame({
        "position_1": [1.0, 0.0],
        "position_2": [-0.5, 0.0]
    })

    result = calculate_pair_weights(data, positions)

    assert result.loc[0, "symbol_1"] == pytest.approx(0.5)
    assert result.loc[0, "symbol_2"] == pytest.approx(-0.5)

    assert result.loc[0].abs().sum() == pytest.approx(1.0)
    assert result.loc[1].abs().sum() == pytest.approx(0.0)

def test_calculate_pair_returns():

    data = pd.DataFrame({
        "symbol_1": [100, 110],
        "symbol_2": [200, 180]
    })

    weights = pd.DataFrame({
        "symbol_1": [0.5, 0.0],
        "symbol_2": [-0.5, 0.0]
    })

    result = calculate_pair_returns(data, weights)

    assert result["strategy_return"].iloc[1] == pytest.approx(0.10)

def test_check_spread_stationarity():

    np.random.seed(42)

    spread = [0]

    for _ in range(100):
        new_value = 0.5 * spread[-1] + np.random.normal()
        spread.append(new_value)

    spread = pd.Series(spread)

    result = check_spread_stationarity(spread)

    assert "adf_statistic" in result
    assert "p_value" in result
    assert "is_stationary" in result

    assert result["p_value"] < 0.05
    assert result["is_stationary"] == True

def test_run_pairs_trading():

    data = pd.DataFrame({
        "symbol_1": [100, 121, 139, 181, 199],
        "symbol_2": [210, 240, 280, 360, 400]
    })

    result = run_pairs_trading(data, alpha = 10, beta=0.5, window=3, threshold=1)

    assert "spread" in result.columns
    assert "zscore" in result.columns
    assert "position_1" in result.columns
    assert "position_2" in result.columns
    assert "strategy_return" in result.columns
    assert "equity" in result.columns

    assert len(result) == len(data)

    assert result["spread"].notna().all()
    assert result["zscore"].notna().sum() > 0
    assert result["strategy_return"].notna().sum() > 0

def test_run_pairs_trading_with_history():

    historical_data = pd.DataFrame({
        "symbol_1": [100, 101, 99, 102, 100],
        "symbol_2": [200, 201, 198, 203, 201]
    })

    data = pd.DataFrame({
        "symbol_1": [101, 103, 102],
        "symbol_2": [202, 204, 203]
    })

    result = run_pairs_trading_with_history(
        data,
        historical_data,
        alpha=0.0,
        beta=0.5,
        window=3,
        threshold=1
    )

    assert len(result) == len(data)
    assert pd.notna(result["zscore"].iloc[0])
    assert "strategy_return" in result.columns
    assert "equity" in result.columns

def test_generate_pairs():

    symbols = ["AAPL", "MSFT", "GOOG"]

    result = generate_pairs(symbols)

    assert result == [
        ("AAPL", "MSFT"),
        ("AAPL", "GOOG"),
        ("MSFT", "GOOG")
    ]

def test_generate_pairs_count():

    symbols = ["AAPL", "MSFT", "GOOG", "NVDA", "AMZN"]

    result = generate_pairs(symbols)

    assert len(result) == 10

def test_screen_pairs():

    t = np.arange(30)

    train_prices = pd.DataFrame({
        "AAPL": 100 + t + np.array([0, 1, -1, 0, 2, -2] * 5),
        "MSFT": 200 + 1.8 * t + np.array([1, -1, 2, -2, 0, 1] * 5),
        "GOOG": 150 + 0.6 * t + np.array([2, 0, -2, 1, -1, 0] * 5)
    })

    candidate_pairs = [
        ("AAPL", "MSFT"),
        ("AAPL", "GOOG")
    ]

    result = screen_pairs(
        train_prices,
        candidate_pairs
    )

    assert len(result) == 2

    expected_columns = {
        "symbol_1",
        "symbol_2",
        "alpha",
        "beta",
        "coint_pvalue",
        "adf_pvalue",
        "is_cointegrated",
        "is_stationary"
    }

    assert expected_columns.issubset(result.columns)

    assert result.iloc[0]["symbol_1"] == "AAPL"
    assert result.iloc[0]["symbol_2"] == "MSFT"

    assert result.iloc[1]["symbol_1"] == "AAPL"
    assert result.iloc[1]["symbol_2"] == "GOOG"

def test_select_pairs():

    screening_results = pd.DataFrame({
        "symbol_1": ["AAPL", "AAPL", "MSFT"],
        "symbol_2": ["MSFT", "GOOG", "GOOG"],
        "coint_pvalue": [0.01, 0.30, 0.03],
        "adf_pvalue": [0.02, 0.20, 0.08],
        "is_cointegrated": [True, False, True],
        "is_stationary": [True, False, False]
    })

    result = select_pairs(screening_results)

    assert len(result) == 2

    assert result.iloc[0]["symbol_1"] == "AAPL"
    assert result.iloc[0]["symbol_2"] == "MSFT"

    assert result.iloc[1]["symbol_1"] == "MSFT"
    assert result.iloc[1]["symbol_2"] == "GOOG"

def test_select_pairs_none_selected():

    screening_results = pd.DataFrame({
        "symbol_1": ["AAPL"],
        "symbol_2": ["MSFT"],
        "coint_pvalue": [0.30],
        "adf_pvalue": [0.20],
        "is_cointegrated": [False],
        "is_stationary": [False]
    })

    result = select_pairs(screening_results)

    assert result.empty

def test_calculate_pair_returns_normalized():

    data = pd.DataFrame({
        "symbol_1": [100, 110],
        "symbol_2": [200, 180]
    })

    weights = pd.DataFrame({
        "symbol_1": [0.5, 0.0],
        "symbol_2": [-0.5, 0.0]
    })

    result = calculate_pair_returns(data, weights)

    assert result["strategy_return"].iloc[1] == pytest.approx(0.10)




