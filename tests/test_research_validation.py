import pandas as pd
import numpy as np
import pytest

from src.research.validation import (run_validation_comparison, compare_strategies, build_strategy_comparison)

def test_compare_strategies():

    strategy_returns = {
        "strategy_a": pd.Series([0.01, 0.02, -0.01]),
        "strategy_b": pd.Series([0.00, 0.01, 0.01])
    }

    result = compare_strategies(strategy_returns)

    assert "strategy_a" in result.index
    assert "strategy_b" in result.index

    assert "Total Return" in result.columns
    assert "Annualized Volatility" in result.columns
    assert "Sharpe Ratio" in result.columns
    assert "Max Drawdown" in result.columns

def test_build_strategy_comparison():

    mean_result = {
        "strategy_return": pd.Series([0.01, 0.02, -0.01])
    }

    pair_result = {
        "strategy_return": pd.Series([0.00, 0.01, 0.01])
    }

    pca_result = {
        "strategy_return": pd.Series([-0.01, 0.02, 0.03])
    }

    result = build_strategy_comparison(
        mean_result,
        pair_result,
        pca_result
    )

    assert "mean_reversion" in result.index
    assert "pairs" in result.index
    assert "pca" in result.index

    assert "Total Return" in result.columns
    assert "Annualized Volatility" in result.columns
    assert "Sharpe Ratio" in result.columns
    assert "Max Drawdown" in result.columns

def test_run_validation_comparison():

    rng = np.random.default_rng(42)

    n_train = 250
    n_validation = 60

    # Common non-stationary trend
    train_base = 100 + np.cumsum(
        rng.normal(0, 1, n_train)
    )

    validation_base = (
        train_base[-1]
        + np.cumsum(rng.normal(0, 1, n_validation))
    )

    # Stationary noises
    train_noise_1 = np.zeros(n_train)
    train_noise_2 = np.zeros(n_train)

    for t in range(1, n_train):
        train_noise_1[t] = (
            0.6 * train_noise_1[t - 1]
            + rng.normal(0, 0.8)
        )

        train_noise_2[t] = (
            0.3 * train_noise_2[t - 1]
            + rng.normal(0, 1.2)
        )

    validation_noise_1 = np.zeros(n_validation)
    validation_noise_2 = np.zeros(n_validation)

    for t in range(1, n_validation):
        validation_noise_1[t] = (
            0.6 * validation_noise_1[t - 1]
            + rng.normal(0, 0.8)
        )

        validation_noise_2[t] = (
            0.3 * validation_noise_2[t - 1]
            + rng.normal(0, 1.2)
        )

    train_prices = pd.DataFrame({
        "AAPL": train_base,
        "MSFT": 20 + 1.8 * train_base + train_noise_1,
        "GOOG": 50 + 0.7 * train_base + train_noise_2
    })

    validation_prices = pd.DataFrame({
        "AAPL": validation_base,
        "MSFT": 20 + 1.8 * validation_base + validation_noise_1,
        "GOOG": 50 + 0.7 * validation_base + validation_noise_2
    })

    result = run_validation_comparison(
        train_prices=train_prices,
        validation_prices=validation_prices,
        mean_symbol="AAPL",
        candidate_pairs=[
            ("AAPL", "MSFT"),
            ("AAPL", "GOOG"),
            ("MSFT", "GOOG")
        ],
        pca_symbols=[
            "AAPL",
            "MSFT",
            "GOOG"
        ],
        window=20,
        threshold=1.5,
        n_components=1
    )

    assert "comparison" in result
    assert "mean_result" in result
    assert "pair_result" in result
    assert "pca_result" in result
    assert "screening_results" in result
    assert "selected_pairs" in result

    comparison = result["comparison"]

    assert "mean_reversion" in comparison.index
    assert "pairs" in comparison.index
    assert "pca" in comparison.index

    assert len(result["mean_result"]) == n_validation
    assert len(result["pair_result"]) == n_validation
    assert len(result["pca_result"]["strategy_return"]) == n_validation

    assert not result["selected_pairs"].empty

