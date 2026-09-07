import pandas as pd
import numpy as np
import pytest

from src.research.validation import (run_validation_comparison, compare_strategies, 
                                     build_strategy_comparison, walk_forward_split, 
                                     combine_history_and_test, walk_forward_mean_reversion,
                                     walk_forward_pca, calculate_pca_stability,
                                     calculate_signal_disagreement, select_best_mean_reversion_parameters,
                                     evaluate_mean_reversion_on_test, run_mean_reversion_research,
                                     build_parameter_surface, calculate_local_robustness,
                                     is_parameter_robust, summarize_parameter_robustness, 
                                     apply_transaction_costs, run_cost_sensitivity, find_break_even_cost,
                                     classify_volatility_regime, compare_performance_by_regime, 
                                     run_regime_analysis, attribute_performance_by_position,
                                     attribute_performance_by_month, attribute_performance_by_asset,
                                     run_performance_attribution)
from src.metrics import performance_summary

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

def test_select_best_mean_reversion_parameters():

    tuning_results = pd.DataFrame({
        "window": [10, 20, 30],
        "threshold": [1.0, 1.5, 2.0],
        "Sharpe Ratio": [0.5, 1.2, 0.8]
    })

    result = select_best_mean_reversion_parameters(tuning_results)

    assert result["window"] == 20
    assert result["threshold"] == 1.5

def test_evaluate_mean_reversion_on_test():

    train_data = pd.DataFrame({
        "Close": [100, 102, 101, 103, 105]
    })

    validation_data = pd.DataFrame({
        "Close": [104, 106, 105]
    })

    test_data = pd.DataFrame({
        "Close": [107, 106, 108]
    })

    best_parameters = {
        "window": 3,
        "threshold": 1.0
    }

    result = evaluate_mean_reversion_on_test(
        train_data,
        validation_data,
        test_data,
        best_parameters
    )

    assert isinstance(result, pd.DataFrame)

    assert "strategy_return" in result.columns

    assert len(result) == len(test_data)

    assert result.index.equals(test_data.index)

def test_run_mean_reversion_research():

    train_data = pd.DataFrame({
        "Close": [100, 102, 101, 103, 105, 104]
    })

    validation_data = pd.DataFrame({
        "Close": [106, 105, 107, 104]
    })

    test_data = pd.DataFrame({
        "Close": [108, 106, 109, 107]
    })

    result = run_mean_reversion_research(
        train_data,
        validation_data,
        test_data,
        windows=[2, 3],
        thresholds=[0.5, 1.0]
    )

    assert "tuning_results" in result
    assert "best_parameters" in result
    assert "test_result" in result
    assert "test_performance" in result

    assert len(result["tuning_results"]) == 4

    assert "window" in result["best_parameters"]
    assert "threshold" in result["best_parameters"]

    assert len(result["test_result"]) == len(test_data)

def test_build_parameter_surface():

    tuning_results = pd.DataFrame({
        "window": [10, 10, 20, 20],
        "threshold": [1.0, 1.5, 1.0, 1.5],
        "Sharpe Ratio": [0.8, 1.0, 1.1, 1.2]
    })

    result = build_parameter_surface(
        tuning_results
    )

    assert result.loc[10, 1.0] == 0.8
    assert result.loc[10, 1.5] == 1.0
    assert result.loc[20, 1.0] == 1.1
    assert result.loc[20, 1.5] == 1.2

    assert result.shape == (2, 2)

def test_calculate_local_robustness():

    tuning_results = pd.DataFrame({
        "window": [
            10, 10, 10,
            20, 20, 20,
            30, 30, 30
        ],
        "threshold": [
            1.0, 1.5, 2.0,
            1.0, 1.5, 2.0,
            1.0, 1.5, 2.0
        ],
        "Sharpe Ratio": [
            1.0, 1.1, 1.0,
            1.2, 1.5, 1.1,
            1.0, 1.2, 1.1
        ]
    })

    result = calculate_local_robustness(
        tuning_results
    )

    assert result["best_window"] == 20
    assert result["best_threshold"] == 1.5
    assert result["best_metric"] == 1.5

    assert result["neighbor_mean"] == pytest.approx(
        (1.0 + 1.1 + 1.0 +
         1.2 + 1.1 +
         1.0 + 1.2 + 1.1) / 8
    )

    assert result["performance_drop"] == pytest.approx(
        1.5 - result["neighbor_mean"]
    )

def test_is_parameter_robust():

    robust_result = {
        "performance_drop": 0.15,
        "neighbor_std": 0.20
    }

    unstable_result = {
        "performance_drop": 0.8,
        "neighbor_std": 0.6
    }

    assert is_parameter_robust(
        robust_result,
        max_performance_drop=0.3,
        max_neighbor_std=0.3
    )

    assert not is_parameter_robust(
        unstable_result,
        max_performance_drop=0.3,
        max_neighbor_std=0.3
    )

def test_summarize_parameter_robustness():

    tuning_results = pd.DataFrame({
        "window": [
            10, 10, 10,
            20, 20, 20,
            30, 30, 30
        ],
        "threshold": [
            1.0, 1.5, 2.0,
            1.0, 1.5, 2.0,
            1.0, 1.5, 2.0
        ],
        "Sharpe Ratio": [
            1.1, 1.2, 1.1,
            1.2, 1.3, 1.2,
            1.1, 1.2, 1.1
        ]
    })

    result = summarize_parameter_robustness(
        tuning_results,
        max_performance_drop=0.3,
        max_neighbor_std=0.3
    )

    assert result["best_window"] == 20
    assert result["best_threshold"] == 1.5

    assert "neighbor_mean" in result
    assert "neighbor_std" in result
    assert "performance_drop" in result
    assert "robust" in result

    assert result["robust"] is True

def test_apply_transaction_costs():

    strategy_returns = pd.Series([
        0.00,
        0.02,
        0.01,
        -0.03
    ])

    positions = pd.Series([
        0,
        1,
        1,
        -1
    ])

    result = apply_transaction_costs(
        strategy_returns,
        positions,
        cost_rate=0.001
    )

    assert result["turnover"].tolist() == [
        0, 1, 0, 2
    ]

    assert result["cost"].tolist() == pytest.approx([
        0,
        0.001,
        0,
        0.002
    ])

    assert result["net_strategy_return"].tolist() == pytest.approx([
        0,
        0.019,
        0.01,
        -0.032
    ])

def test_run_cost_sensitivity():

    strategy_returns = pd.Series([
        0.00,
        0.02,
        0.01,
        -0.03,
        0.02
    ])

    positions = pd.Series([
        0,
        1,
        1,
        -1,
        0
    ])

    cost_rates = [
        0,
        0.0005,
        0.001
    ]

    result = run_cost_sensitivity(
        strategy_returns,
        positions,
        cost_rates
    )

    assert isinstance(result, pd.DataFrame)

    assert len(result) == 3

    assert result["cost_rate"].tolist() == cost_rates

    assert "Total Return" in result.columns
    assert "Sharpe Ratio" in result.columns
    assert "Max Drawdown" in result.columns

    assert (result.loc[result["cost_rate"] == 0.001, "Total Return"].iloc[0]
            <=
            result.loc[result["cost_rate"] == 0, "Total Return"].iloc[0])

def test_find_break_even_cost():

    results = pd.DataFrame({
        "cost_rate": [
            0,
            0.0005,
            0.001,
            0.002
        ],
        "Total Return": [
            0.10,
            0.06,
            0.02,
            -0.03
        ]
    })

    break_even_cost = find_break_even_cost(results)

    assert break_even_cost == pytest.approx(0.002)

def test_find_break_even_cost_not_reached():

    results = pd.DataFrame({
        "cost_rate": [
            0,
            0.0005,
            0.001
        ],
        "Total Return": [
            0.10,
            0.08,
            0.05
        ]
    })

    break_even_cost = find_break_even_cost(results)

    assert break_even_cost is None

def test_classify_volatility_regime():

    returns = pd.Series([
        0.01,
        0.01,
        0.01,
        0.10,
        -0.10,
        0.10
    ])

    result = classify_volatility_regime(
        returns,
        window=3,
        volatility_threshold=0.05
    )

    assert "rolling_volatility" in result.columns
    assert "regime" in result.columns

    assert len(result) == len(returns)

    assert pd.isna(result["regime"].iloc[0])
    assert pd.isna(result["regime"].iloc[1])
    assert pd.isna(result["regime"].iloc[2])

    assert result["regime"].iloc[3] == "calm"

    assert result["regime"].iloc[5] == "volatile"

def test_compare_performance_by_regime():

    strategy_returns = pd.Series([
        0.01,
        0.02,
        -0.01,
        0.03,
        -0.02,
        0.01
    ])

    regimes = pd.Series([
        "calm",
        "calm",
        "volatile",
        "volatile",
        "volatile",
        "calm"
    ])

    result = compare_performance_by_regime(
        strategy_returns,
        regimes
    )

    assert isinstance(result, pd.DataFrame)

    assert "calm" in result.index
    assert "volatile" in result.index

    assert "Total Return" in result.columns
    assert "Sharpe Ratio" in result.columns
    assert "Max Drawdown" in result.columns

    calm_expected = performance_summary(
        pd.Series([0.01, 0.02, 0.01])
    )

    assert result.loc["calm", "Total Return"] == pytest.approx(
        calm_expected["Total Return"]
    )

def test_run_regime_analysis():

    market_returns = pd.Series([
        0.01,
        0.01,
        0.01,
        0.10,
        -0.10,
        0.10,
        0.01
    ])

    strategy_returns = pd.Series([
        0.01,
        0.02,
        0.01,
        -0.03,
        0.02,
        -0.01,
        0.01
    ])

    result = run_regime_analysis(
        market_returns,
        strategy_returns,
        window=3,
        volatility_threshold=0.05
    )

    assert "regime_result" in result
    assert "comparison" in result

    assert len(result["regime_result"]) == len(market_returns)

    assert "regime" in result["regime_result"].columns

    assert "calm" in result["comparison"].index
    assert "volatile" in result["comparison"].index

    assert "Sharpe Ratio" in result["comparison"].columns

def test_attribute_performance_by_position():

    strategy_returns = pd.Series([
        0.00,
        0.02,
        0.01,
        -0.01,
        0.03,
        -0.02
    ])

    positions = pd.Series([
        0,
        1,
        1,
        -1,
        1,
        -1
    ])

    result = attribute_performance_by_position(
        strategy_returns,
        positions
    )

    assert result.loc["long", "count"] == 3
    assert result.loc["short", "count"] == 2
    assert result.loc["flat", "count"] == 1

    assert result.loc["long", "sum"] == pytest.approx(0.06)
    assert result.loc["short", "sum"] == pytest.approx(-0.03)
    assert result.loc["flat", "sum"] == pytest.approx(0.00)

def test_attribute_performance_by_month():

    dates = pd.to_datetime([
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
        "2025-02-01",
        "2025-02-02"
    ])

    strategy_returns = pd.Series(
        [
            0.01,
            0.02,
            0.01,
            -0.02,
            0.01
        ],
        index=dates
    )

    result = attribute_performance_by_month(
        strategy_returns
    )

    assert len(result) == 2

    assert result.loc[pd.Period("2025-01"), "count"] == 3
    assert result.loc[pd.Period("2025-02"), "count"] == 2

    assert result.loc[pd.Period("2025-01"), "sum"] == pytest.approx(0.04)

    assert result.loc[pd.Period("2025-02"), "sum"] == pytest.approx(-0.01)

def test_attribute_performance_by_asset():

    asset_strategy_returns = pd.DataFrame({
        "AAPL": [0.01, 0.02, -0.01],
        "MSFT": [0.00, -0.01, 0.02],
        "GOOG": [0.02, 0.00, 0.01]
    })

    result = attribute_performance_by_asset(
        asset_strategy_returns
    )

    assert result.loc["AAPL", "count"] == 3
    assert result.loc["MSFT", "count"] == 3

    assert result.loc["AAPL", "sum"] == pytest.approx(0.02)
    assert result.loc["MSFT", "sum"] == pytest.approx(0.01)
    assert result.loc["GOOG", "sum"] == pytest.approx(0.03)

    assert result.loc["AAPL", "mean"] == pytest.approx(
        0.02 / 3
    )

def test_run_performance_attribution():

    dates = pd.to_datetime([
        "2025-01-01",
        "2025-01-02",
        "2025-02-01",
        "2025-02-02"
    ])

    strategy_returns = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=dates
    )

    positions = pd.Series(
        [1, 1, -1, 1],
        index=dates
    )

    asset_strategy_returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.01, -0.01, 0.02],
            "MSFT": [0.00, 0.01, 0.00, 0.01]
        },
        index=dates
    )

    result = run_performance_attribution(
        strategy_returns,
        positions,
        asset_strategy_returns
    )

    assert "position_attribution" in result
    assert "monthly_attribution" in result
    assert "asset_attribution" in result

    assert "long" in result["position_attribution"].index
    assert "short" in result["position_attribution"].index

    assert pd.Period("2025-01") in result["monthly_attribution"].index
    assert pd.Period("2025-02") in result["monthly_attribution"].index

    assert "AAPL" in result["asset_attribution"].index
    assert "MSFT" in result["asset_attribution"].index

def test_run_performance_attribution_without_assets():

    dates = pd.to_datetime([
        "2025-01-01",
        "2025-01-02"
    ])

    strategy_returns = pd.Series(
        [0.01, 0.02],
        index=dates
    )

    positions = pd.Series(
        [1, 1],
        index=dates
    )

    result = run_performance_attribution(
        strategy_returns,
        positions
    )

    assert result["asset_attribution"] is None
