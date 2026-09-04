from src.research import (split_data, calculate_zscore, generate_signal, signal_to_position,
                          calculate_strategy_returns, calculate_equity_curve, run_mean_reversion,
                          calculate_spread, estimate_beta, generate_pair_positions, calculate_pair_returns,
                          run_pairs_trading, check_spread_stationarity, estimate_hedge_ratio,
                          calculate_pca_residuals, calculate_residual_zscore, generate_residual_signals,
                          run_pca_stat_arb, fit_pca, calculate_residual_zscore_with_history)
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



def test_estimate_beta():

    data = pd.DataFrame({
        "symbol_2" : [100, 120, 150],
        "symbol_1" : [200, 240, 300]
    })

    result = estimate_beta(data)

    assert result == pytest.approx(2)

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


def test_calculate_pair_returns():

    data = pd.DataFrame({
        "symbol_1": [100, 110, 121],
        "symbol_2": [200, 180, 198]
    })

    positions = pd.DataFrame({
        "position_1": [1, -1, 0],
        "position_2": [-0.5, 0.5, 0]
    })

    result = calculate_pair_returns(data, positions)

    assert pd.isna(result["strategy_return"].iloc[0])
    assert result["strategy_return"].iloc[1] == pytest.approx(0.15)
    assert result["strategy_return"].iloc[2] == pytest.approx(-0.05)



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

def test_hedge_ratio():

    data = pd.DataFrame({
        "symbol_1": [210, 252, 288, 372, 408],
        "symbol_2": [100, 121, 139, 181, 199]
    })

    result = estimate_hedge_ratio(data)

    assert result["alpha"] == pytest.approx(10)
    assert result["beta"] == pytest.approx(2)

def test_calculate_spread():

    data = pd.DataFrame({
        "symbol_1": [210, 252, 288],
        "symbol_2": [100, 121, 139]
    })

    result = calculate_spread(data, 2.0, 10.0)

    assert result["spread"].iloc[0] == pytest.approx(0)
    assert result["spread"].iloc[1] == pytest.approx(0)
    assert result["spread"].iloc[2] == pytest.approx(0)


def test_calculate_pca_residuals_shape():

    train = pd.DataFrame({
        "AAPL": [0.01, 0.02, -0.01, 0.03],
        "MSFT": [0.02, 0.01, -0.02, 0.02],
        "GOOG": [0.01, -0.01, 0.02, 0.01]
    })

    test = pd.DataFrame({
        "AAPL": [-0.02, 0.01],
        "MSFT": [-0.01, 0.03],
        "GOOG": [-0.03, 0.02]
    })

    pca = fit_pca(train, 2)

    result = calculate_pca_residuals(test, pca)

    assert result.shape == test.shape
    assert result.index.equals(test.index)
    assert result.columns.equals(test.columns)

def test_calculate_residual_zscore():
    train = pd.DataFrame({
        "AAPL": [0.01, 0.02, -0.01, 0.03],
        "MSFT": [0.02, 0.01, -0.02, 0.02],
        "GOOG": [0.01, -0.01, 0.02, 0.01]
    })

    test = pd.DataFrame({
        "AAPL": [-0.02, 0.01],
        "MSFT": [-0.01, 0.03],
        "GOOG": [-0.03, 0.02]
    })

    pca = fit_pca(train, 2)

    residuals = calculate_pca_residuals(test, pca)

    result = calculate_residual_zscore(residuals, 3)

    assert result.shape == residuals.shape
    assert result.columns.equals(residuals.columns)
    assert result.index.equals(residuals.index)

    assert result.iloc[0].isna().all()
    assert result.iloc[1].isna().all()

def test_generate_signals():

    zscores = pd.DataFrame({
        "AAPL": [-2.0, 0.5, 1.5],
        "MSFT": [2.0, -0.2, -1.5]
    })

    result = generate_residual_signals(zscores, 1)

    expected = pd.DataFrame({
        "AAPL" : [1.0, 0, -1.0],
        "MSFT" : [-1.0, 0, +1.0]
    })

    pd.testing.assert_frame_equal(result, expected)

def test_run_pca_stat_arb():

    train = pd.DataFrame({
        "AAPL": [0.01, 0.02, -0.01, 0.03],
        "MSFT": [0.02, 0.01, -0.02, 0.02],
        "GOOG": [0.01, -0.01, 0.02, 0.01]
    })

    test = pd.DataFrame({
        "AAPL": [-0.02, 0.01, 0.03, -0.01],
        "MSFT": [-0.01, 0.03, 0.02, -0.02],
        "GOOG": [-0.03, 0.02, 0.01, 0.03]
    })

    pca = fit_pca(train, 2)

    result = run_pca_stat_arb(
        test,
        train,
        pca,
        window=3,
        threshold=1
    )

    assert set(result.keys()) == {
        "residuals",
        "zscores",
        "signals"
    }

    assert result["residuals"].shape == test.shape
    assert result["zscores"].shape == test.shape
    assert result["signals"].shape == test.shape

    assert result["zscores"].index.equals(test.index)
    assert result["signals"].columns.equals(test.columns)

    assert result["zscores"].iloc[0].notna().all()


def test_residual_zscore_with_history():

    train = pd.DataFrame({
        "AAPL": [0.01, 0.02, -0.01, 0.03],
        "MSFT": [0.02, 0.01, -0.02, 0.02],
        "GOOG": [0.01, -0.01, 0.02, 0.01]
    })

    test = pd.DataFrame({
        "AAPL": [-0.02, 0.01, 0.03, -0.01],
        "MSFT": [-0.01, 0.03, 0.02, -0.02],
        "GOOG": [-0.03, 0.02, 0.01, 0.03]
    })

    pca = fit_pca(train, 2)

    train_residual = calculate_pca_residuals(train, pca)

    test_residual = calculate_pca_residuals(test, pca)

    result = calculate_residual_zscore_with_history(train_residual, test_residual, 3)

    assert result.shape == test_residual.shape
    assert result.index.equals(test_residual.index)
    assert result.iloc[0].notna().all()












