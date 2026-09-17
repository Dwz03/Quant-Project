import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations
import pytest

from src.research.pairs import (estimate_beta, calculate_spread, estimate_hedge_ratio, generate_pair_positions,
                                calculate_pair_weights, calculate_pair_returns, check_spread_stationarity,
                                run_pairs_trading, run_pairs_trading_with_history, generate_pairs,
                                screen_pairs, select_pairs, check_cointegration,
                                estimate_spread_half_life, rolling_hedge_ratio,
                                summarize_beta_stability, cointegration_diagnostic,
                                spread_stationarity_diagnostic)

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


def test_static_ols_and_exact_spread_do_not_mutate_input():
    index = pd.date_range("2024-01-01", periods=20)
    x = np.linspace(10.0, 30.0, len(index))
    data = pd.DataFrame(
        {"symbol_1": 2.0 + 1.5 * x, "symbol_2": x}, index=index
    )
    original = data.copy(deep=True)

    fitted = estimate_hedge_ratio(data)
    spread = calculate_spread(data, fitted["beta"], fitted["alpha"])

    assert fitted == pytest.approx({"alpha": 2.0, "beta": 1.5})
    assert spread["spread"].abs().max() < 1e-12
    pd.testing.assert_frame_equal(data, original)


def test_spread_uses_supplied_parameters_without_refitting():
    data = pd.DataFrame({"symbol_1": [10.0, 12.0], "symbol_2": [2.0, 3.0]})

    result = calculate_spread(data, beta=4.0, alpha=1.0)

    pd.testing.assert_series_equal(
        result["spread"], pd.Series([1.0, -1.0], name="spread")
    )


@pytest.mark.parametrize(
    "data, message",
    [
        (
            pd.DataFrame(
                {"symbol_1": [10.0, 11.0], "symbol_2": [5.0, 6.0]},
                index=[1, 0],
            ),
            "chronological",
        ),
        (
            pd.DataFrame(
                {"symbol_1": [10.0, 11.0], "symbol_2": [5.0, 6.0]},
                index=[0, 0],
            ),
            "duplicate",
        ),
        (
            pd.DataFrame(
                {"symbol_1": [10.0, np.nan], "symbol_2": [5.0, 6.0]}
            ),
            "finite",
        ),
    ],
)
def test_pair_validation_rejects_bad_alignment(data, message):
    with pytest.raises(ValueError, match=message):
        estimate_hedge_ratio(data)


def test_cointegration_diagnostic_structure():
    rng = np.random.default_rng(7)
    x = 100.0 + np.cumsum(rng.normal(size=300))
    data = pd.DataFrame(
        {"symbol_1": 5.0 + 1.2 * x + rng.normal(scale=0.3, size=300),
         "symbol_2": x}
    )

    result = cointegration_diagnostic(data)

    assert {"test_statistic", "p_value"}.issubset(result)
    assert "is_cointegrated" not in result
    assert np.isfinite(result["test_statistic"])
    assert 0.0 <= result["p_value"] <= 1.0


def test_cointegrated_pair_is_more_stationary_than_unrelated_walks():
    rng = np.random.default_rng(19)
    x = 100.0 + np.cumsum(rng.normal(size=600))
    stationary_noise = np.empty(600)
    stationary_noise[0] = 0.0
    shocks = rng.normal(scale=0.5, size=600)
    for position in range(1, 600):
        stationary_noise[position] = 0.4 * stationary_noise[position - 1] + shocks[position]
    cointegrated = pd.DataFrame(
        {"symbol_1": 50.0 + 1.3 * x + stationary_noise, "symbol_2": x}
    )
    unrelated = pd.DataFrame(
        {"symbol_1": 100.0 + np.cumsum(rng.normal(size=600)),
         "symbol_2": 100.0 + np.cumsum(rng.normal(size=600))}
    )

    cointegrated_p = cointegration_diagnostic(cointegrated)["p_value"]
    unrelated_p = cointegration_diagnostic(unrelated)["p_value"]

    assert cointegrated_p < unrelated_p
    stationary_result = spread_stationarity_diagnostic(
        calculate_spread(cointegrated, 1.3, 50.0)["spread"]
    )
    unrelated_result = spread_stationarity_diagnostic(
        calculate_spread(unrelated, 1.0, 0.0)["spread"]
    )
    assert "is_stationary" not in stationary_result
    assert stationary_result["p_value"] < unrelated_result["p_value"]


def test_half_life_for_mean_reverting_spread_is_positive_and_finite():
    rng = np.random.default_rng(11)
    spread = np.zeros(500)
    for position in range(1, len(spread)):
        spread[position] = 0.8 * spread[position - 1] + rng.normal(scale=0.2)

    result = estimate_spread_half_life(pd.Series(spread))

    assert result["lambda"] < 0
    assert np.isfinite(result["half_life"])
    assert result["half_life"] > 0


def test_half_life_is_nan_when_dynamics_are_not_mean_reverting():
    spread = pd.Series(np.square(np.arange(1.0, 101.0)))

    result = estimate_spread_half_life(spread)

    assert result["lambda"] >= 0
    assert np.isnan(result["half_life"])


def test_rolling_hedge_ratio_is_trailing_and_has_warmup_nans():
    index = pd.date_range("2024-01-01", periods=12)
    x = np.arange(10.0, 22.0)
    data = pd.DataFrame(
        {"symbol_1": 3.0 + 2.0 * x, "symbol_2": x}, index=index
    )

    result = rolling_hedge_ratio(data, window=5)

    assert result.iloc[:4].isna().all().all()
    assert result.iloc[4:]["alpha"].to_numpy() == pytest.approx(3.0)
    assert result.iloc[4:]["beta"].to_numpy() == pytest.approx(2.0)
    assert result.index.equals(index)


def test_future_changes_do_not_change_earlier_rolling_estimates():
    index = pd.date_range("2024-01-01", periods=15)
    x = np.arange(10.0, 25.0)
    data = pd.DataFrame(
        {"symbol_1": 4.0 + 0.75 * x, "symbol_2": x}, index=index
    )
    changed = data.copy()
    changed.iloc[10:, 0] *= 4.0

    original_result = rolling_hedge_ratio(data, window=5)
    changed_result = rolling_hedge_ratio(changed, window=5)

    pd.testing.assert_frame_equal(
        original_result.iloc[:10], changed_result.iloc[:10]
    )


def test_beta_stability_uses_only_valid_estimates():
    rolling = pd.DataFrame({"beta": [np.nan, 1.0, 2.0, np.inf, 3.0]})

    result = summarize_beta_stability(rolling)

    assert result == pytest.approx(
        {
            "mean_beta": 2.0,
            "median_beta": 2.0,
            "std_beta": 1.0,
            "min_beta": 1.0,
            "max_beta": 3.0,
            "valid_estimates": 3,
        }
    )




