from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import pytest

from src.research.pca import (calculate_pca_residuals, fit_pca, calculate_residual_zscore,
                              calculate_residual_zscore_with_history, generate_residual_signals,
                              calculate_portfolio_returns, run_pca_stat_arb, estimate_ar1_phi,
                              calculate_half_life, estimate_mean_reversion_horizon, 
                              summarize_mean_reversion_horizons)



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

def test_calculate_portfolio_returns():

    returns = pd.DataFrame({
        "AAPL": [0.00, 0.10],
        "MSFT": [0.00, -0.10]
    })

    positions = pd.DataFrame({
        "AAPL": [0.5, 0.0],
        "MSFT": [-0.5, 0.0]
    })

    result = calculate_portfolio_returns(returns,positions)

    assert result.iloc[1] == pytest.approx(0.10)


def test_run_pca_stat_arb_returns():

    historical_data = pd.DataFrame({
        "AAPL": [0.01, 0.02, -0.01, 0.03],
        "MSFT": [0.02, 0.01, -0.02, 0.02]
    })

    data = pd.DataFrame({
        "AAPL": [0.01, -0.02, 0.03],
        "MSFT": [-0.01, 0.02, -0.01]
    })

    pca = fit_pca(historical_data, n_components=1)

    result = run_pca_stat_arb(data, historical_data, pca, window=3, threshold=1)

    assert "strategy_return" in result
    assert "equity" in result
    assert "positions" in result

    assert len(result["strategy_return"]) == len(data)
    assert len(result["equity"]) == len(data)


def test_estimate_ar1_phi():

    series = pd.Series([1.0, 0.8, 0.64, 0.512])

    result = estimate_ar1_phi(series)

    assert result == pytest.approx(0.8)

def test_calculate_half_life():

    result = calculate_half_life(0.5)

    assert result == pytest.approx(1.0)


def test_calculate_half_life_phi_08():

    result = calculate_half_life(0.8)

    assert result == pytest.approx(3.1062837)

def test_calculate_half_life_invalid():

    with pytest.raises(ValueError):
        calculate_half_life(1.0)

    with pytest.raises(ValueError):
        calculate_half_life(0.0)

    with pytest.raises(ValueError):
        calculate_half_life(-0.5)

def test_estimate_mean_reversion_horizon():

    series = pd.Series([1.0, 0.5, 0.25, 0.125])

    result = estimate_mean_reversion_horizon(series)

    assert result["phi"] == pytest.approx(0.5)
    assert result["half_life"] == pytest.approx(1.0)

def test_summarize_mean_reversion_horizons():

    residuals = pd.DataFrame({
        "AAPL": [1.0, 0.5, 0.25, 0.125],
        "MSFT": [1.0, 1.1, 1.21, 1.331]
    })

    result = summarize_mean_reversion_horizons(residuals)

    assert result.loc["AAPL", "phi"] == pytest.approx(0.5)
    assert result.loc["AAPL", "half_life"] == pytest.approx(1.0)

    assert result.loc["MSFT", "phi"] == pytest.approx(1.1)
    assert pd.isna(result.loc["MSFT", "half_life"])


