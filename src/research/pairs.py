import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations

from src.research.common import (normalize_positions, calculate_zscore, calculate_equity_curve, calculate_zscore_with_history)

def calculate_spread(data, beta, alpha = 0.0):

    data_used = data.copy()

    data_used["spread"] = data_used["symbol_1"] - beta * data_used["symbol_2"] - alpha

    return data_used

def estimate_beta(data):

    data_used = data.copy()
    data_used["product"] = data_used["symbol_1"] * data_used["symbol_2"]
    data_used["squared"] = data_used["symbol_2"] ** 2
    beta = data_used["product"].sum() / data_used["squared"].sum()

    return beta

def estimate_hedge_ratio(data):

    y = data["symbol_1"]
    x = data["symbol_2"]

    X = np.column_stack([np.ones(len(x)), x])

    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

    alpha = coefficients[0]
    beta = coefficients[1]

    return {
        "alpha": alpha,
        "beta": beta
    }

def generate_pair_positions(zscore, threshold, beta):

    position_1 = pd.Series(0.0, index = zscore.index)
    position_2 = pd.Series(0.0, index = zscore.index)

    position_1.loc[zscore > threshold] = -1
    position_2.loc[zscore > threshold] = +beta

    position_1.loc[zscore < -threshold] = 1
    position_2.loc[zscore < -threshold] = -beta

    result = pd.DataFrame({
        "position_1": position_1,
        "position_2": position_2
        })

    return result

def calculate_pair_weights(data, positions):

    notionals = pd.DataFrame(index=data.index)

    notionals["symbol_1"] = (
        positions["position_1"] * data["symbol_1"]
    )

    notionals["symbol_2"] = (
        positions["position_2"] * data["symbol_2"]
    )

    weights = normalize_positions(notionals)

    return weights

def calculate_pair_returns(data, weights):

    data_used = data.copy()

    data_used["return_1"] = data_used["symbol_1"].pct_change()
    data_used["return_2"] = data_used["symbol_2"].pct_change()

    weights_used = weights.shift(1)

    data_used["strategy_return"] = (
        weights_used["symbol_1"] * data_used["return_1"]
        + weights_used["symbol_2"] * data_used["return_2"]
    )

    data_used.loc[data_used.index[0], "strategy_return"] = 0.0

    return data_used

def check_spread_stationarity(spread):

    result = adfuller(spread, result_object=False)

    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05
    }

def check_cointegration(data):

    result = coint(data["symbol_1"], data["symbol_2"])

    return {
            "test_statistic": result[0],
            "p_value": result[1],
            "is_cointegrated": result[1] < 0.05
        }

def run_pairs_trading(data, alpha, beta, window, threshold):

    data_spread = calculate_spread(data, beta, alpha)

    zscore = calculate_zscore(data_spread, window, column="spread")

    data_spread["zscore"] = zscore

    positions = generate_pair_positions(zscore, threshold, beta)

    weights = calculate_pair_weights(data_spread, positions)

    data_spread["position_1"] = positions["position_1"]
    data_spread["position_2"] = positions["position_2"]

    data_spread["weight_1"] = weights["symbol_1"]
    data_spread["weight_2"] = weights["symbol_2"]

    result = calculate_pair_returns(data_spread, weights)

    equity = calculate_equity_curve(result["strategy_return"])

    result["equity"] = equity

    return result

def run_pairs_trading_with_history(data, historical_data, alpha, beta, window, threshold):

    historical_spread = calculate_spread(historical_data, beta, alpha)

    data_spread = calculate_spread(data, beta, alpha)

    zscore = calculate_zscore_with_history(historical_spread, data_spread, window, column="spread")

    data_spread["zscore"] = zscore

    positions = generate_pair_positions(zscore, threshold, beta)

    weights = calculate_pair_weights(data_spread, positions)

    data_spread["position_1"] = positions["position_1"]
    data_spread["position_2"] = positions["position_2"]

    data_spread["weight_1"] = weights["symbol_1"]
    data_spread["weight_2"] = weights["symbol_2"]

    result = calculate_pair_returns(data_spread, weights)

    result["equity"] = calculate_equity_curve(result["strategy_return"])

    return result

def generate_pairs(symbols):

    return list(combinations(symbols, 2))

def screen_pairs(train_prices, candidate_pairs):

    results = []

    for symbol_1, symbol_2 in candidate_pairs:

        pair_data = pd.DataFrame({
            "symbol_1": train_prices[symbol_1],
            "symbol_2": train_prices[symbol_2]
        })

        hedge_ratio = estimate_hedge_ratio(pair_data)

        alpha = hedge_ratio["alpha"]
        beta = hedge_ratio["beta"]

        spread_data = calculate_spread(
            pair_data,
            beta,
            alpha
        )

        cointegration = check_cointegration(pair_data)

        stationarity = check_spread_stationarity(
            spread_data["spread"]
        )

        results.append({
            "symbol_1": symbol_1,
            "symbol_2": symbol_2,
            "alpha": alpha,
            "beta": beta,
            "coint_pvalue": cointegration["p_value"],
            "adf_pvalue": stationarity["p_value"],
            "is_cointegrated": cointegration["is_cointegrated"],
            "is_stationary": stationarity["is_stationary"]
        })

    return pd.DataFrame(results)

def select_pairs(screening_results):

    selected = screening_results[
        screening_results["is_cointegrated"]
    ]

    selected = selected.sort_values(
        "coint_pvalue",
        ascending=True
    )

    return selected

def run_best_pair_validation(train_prices, validation_prices, selected_pairs, window, threshold):

    best_pair = selected_pairs.iloc[0]

    symbol_1 = best_pair["symbol_1"]
    symbol_2 = best_pair["symbol_2"]

    alpha = best_pair["alpha"]
    beta = best_pair["beta"]

    pair_train = pd.DataFrame({
        "symbol_1": train_prices[symbol_1],
        "symbol_2": train_prices[symbol_2]
    })

    pair_validation = pd.DataFrame({
        "symbol_1": validation_prices[symbol_1],
        "symbol_2": validation_prices[symbol_2]
    })

    return run_pairs_trading_with_history(pair_validation, pair_train, alpha, beta, window, threshold)

