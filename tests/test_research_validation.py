import pandas as pd
import numpy as np
import pytest

from src.research.validation import (run_validation_comparison, compare_strategies, 
                                     build_strategy_comparison, walk_forward_split, 
                                     combine_history_and_test, walk_forward_mean_reversion,
                                     walk_forward_pca, calculate_pca_stability,
                                     calculate_signal_disagreement)

def test_walk_forward_split():

    data = pd.DataFrame({"price": range(10)})

    splits = list(walk_forward_split(data, initial_train_size=4, test_size=2))

    assert len(splits) == 3

    train_1, test_1 = splits[0]

    assert train_1["price"].tolist() == [0, 1, 2, 3]
    assert test_1["price"].tolist() == [4, 5]

    train_2, test_2 = splits[1]

    assert train_2["price"].tolist() == [0, 1, 2, 3, 4, 5]
    assert test_2["price"].tolist() == [6, 7]

    train_3, test_3 = splits[2]

    assert train_3["price"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert test_3["price"].tolist() == [8, 9]

def test_calculate_signal_disagreement():

    fixed = pd.DataFrame({
        "A": [1, 0, -1],
        "B": [0, 1, 0]
    })

    refitted = pd.DataFrame({
        "A": [1, -1, -1],
        "B": [0, 1, 1]
    })

    result = calculate_signal_disagreement(
        fixed,
        refitted
    )

    assert result["overall_disagreement_rate"] == pytest.approx(2 / 6)

    assert result["daily_disagreement_rate"] == pytest.approx(2 / 3)

    assert result["per_symbol_disagreement"]["A"] == pytest.approx(1 / 3)

    assert result["per_symbol_disagreement"]["B"] == pytest.approx(1 / 3)

def test_combine_history_and_test():

    train = pd.DataFrame({ "Close": [100, 101, 102, 103, 104, 105]})

    test = pd.DataFrame({"Close": [106, 107]}, index=[6, 7])

    train.index = range(6)

    result = combine_history_and_test(train, test, window=3)

    assert result["Close"].tolist() == [103, 104, 105, 106, 107]

    assert result.index.tolist() == [3, 4, 5, 6, 7]

def test_walk_forward_mean_reversion():

    data = pd.DataFrame({"Close": [100, 101, 102, 101, 100, 99, 100, 101, 102, 103]})

    result = walk_forward_mean_reversion(data, initial_train_size=4, test_size=2, window=3, threshold=1)

    assert len(result) == 6

    assert result.index.tolist() == [4, 5, 6, 7, 8, 9]

    assert "strategy_return" in result.columns

    assert "equity" in result.columns

    expected_equity = (1 + result["strategy_return"]).cumprod()

    pd.testing.assert_series_equal(result["equity"], expected_equity, check_names=False)

def test_walk_forward_pca():

    prices = pd.DataFrame({
        "A": [100, 101, 102, 101, 103, 104, 105, 104, 106, 107],
        "B": [50, 51, 50, 52, 53, 52, 54, 55, 54, 56],
        "C": [80, 81, 82, 81, 83, 84, 83, 85, 86, 87]
    })

    result = walk_forward_pca(
        prices,
        initial_train_size=6,
        test_size=2,
        n_components=2,
        window=3,
        threshold=1
    )

    assert len(result["strategy_return"]) == 4

    assert result["strategy_return"].index.tolist() == [
        6, 7, 8, 9
    ]

    assert len(result["equity"]) == 4

    assert len(result["folds"]) == 2

def test_calculate_pca_stability():

    folds = [
        {
            "pca_components": np.array([
                [1.0, 0.0],
                [0.0, 1.0]
            ])
        },
        {
            "pca_components": np.array([
                [-1.0, 0.0],
                [0.0, 1.0]
            ])
        }
    ]

    result = calculate_pca_stability(folds)

    assert len(result) == 1

    assert result["pc1_similarity"].iloc[0] == pytest.approx(1.0)

    assert result["pc2_similarity"].iloc[0] == pytest.approx(1.0)

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

