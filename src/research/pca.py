from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from src.research.common import (normalize_positions, calculate_equity_curve)


def fit_pca(data, n_components):

    data_used = data.copy()

    pca = PCA(n_components = n_components)

    return pca.fit(data_used)

def calculate_pca_residuals(data, pca):

    data_used = data.copy()

    factors = pca.transform(data_used)

    reconstructed = pca.inverse_transform(factors)

    reconstructed = pd.DataFrame(reconstructed, index = data_used.index, columns = data_used.columns)

    residual = data_used - reconstructed

    return residual

def calculate_residual_zscore(residuals, window):

    rolling_mean = residuals.rolling(window).mean()
    rolling_std = residuals.rolling(window).std()

    zscore = (residuals - rolling_mean) / rolling_std

    return zscore

def calculate_residual_zscore_with_history(history_residuals, residuals, window):

    combined = pd.concat([history_residuals, residuals])

    combined_zscores = calculate_residual_zscore(combined, window)

    zscores = combined_zscores.iloc[-len(residuals):]

    return zscores

def generate_residual_signals(zscores, threshold):

    signals = pd.DataFrame(0.0, index = zscores.index, columns = zscores.columns)
    signals[zscores > threshold] = -1
    signals[zscores < -threshold] = +1

    return signals

def calculate_portfolio_returns(returns, positions):

    strategy_return = (positions.shift(1) * returns).sum(axis=1)

    strategy_return.iloc[0] = 0.0

    return strategy_return

def run_pca_stat_arb(data, historical_data, pca, window, threshold):

    historical_residual = calculate_pca_residuals(historical_data, pca)

    residuals = calculate_pca_residuals(data, pca)

    zscores = calculate_residual_zscore_with_history(historical_residual, residuals, window)

    signals = generate_residual_signals(zscores, threshold)

    positions = normalize_positions(signals)

    strategy_return = calculate_portfolio_returns(data, positions)

    equity = calculate_equity_curve(strategy_return)

    return {
        "residuals": residuals,
        "zscores": zscores,
        "signals": signals,
        "positions": positions,
        "strategy_return": strategy_return,
        "equity": equity
    }

def estimate_ar1_phi(series):

    x = series.shift(1)
    y = series

    phi = (x * y).sum() / (x ** 2).sum()

    return phi

def calculate_half_life(phi):

    if phi <= 0 or phi >= 1:
        raise ValueError("phi must be between 0 and 1 to be meaningful")

    half_life = -np.log(2) / np.log(phi)

    return half_life

def estimate_mean_reversion_horizon(series):

    phi = estimate_ar1_phi(series)

    half_life = calculate_half_life(phi)

    return {
        "phi": phi,
        "half_life": half_life
    }

def summarize_mean_reversion_horizons(residuals):

    results = {}

    for symbol in residuals.columns:

        series = residuals[symbol].dropna()

        phi = estimate_ar1_phi(series)

        try:
            half_life = calculate_half_life(phi)

        except ValueError:
            half_life = None

        results[symbol] = {
            "phi": phi,
            "half_life": half_life
        }

    return pd.DataFrame(results).T

