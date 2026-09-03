import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint

def split_data(data, train_ratio, validation_ratio):

    if not 0 < train_ratio < 1:
        raise ValueError("train ratio must be between 0 and 1")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation ratio must be between 0 and 1")

    test_ratio = 1 - train_ratio - validation_ratio

    if not 0 <= test_ratio <= 1:
        raise ValueError("test ratio must be between 0 and 1")
    
    train_end = int(len(data) * train_ratio)
    validation_end = int(len(data) * (train_ratio + validation_ratio))

    train_data = data.iloc[:train_end]
    validation_data = data.iloc[train_end:validation_end]
    test_data = data.iloc[validation_end:]  

    return train_data, validation_data, test_data

def calculate_zscore(data, window, column = "Close"):

    data_used = data.copy()

    data_used["rolling_mean"] = data_used[column].rolling(window).mean()
    data_used["std"] = data_used[column].rolling(window).std()
    data_used["zscore"] = (data_used[column] - data_used["rolling_mean"]) / data_used["std"]

    return data_used["zscore"]

def generate_signal(zscore, threshold):

    signals = pd.Series("HOLD", index = zscore.index)

    signals.loc[zscore > threshold] = "SELL"
    signals.loc[zscore < -threshold] = "BUY"

    return signals

def signal_to_position(signals):

    return signals.map({
        "BUY" : 1,
        "HOLD" : 0,
        "SELL" : -1
    })

def calculate_strategy_returns(data, positions):

    data_used = data.copy()

    data_used["return"] = data_used["Close"].pct_change()
    data_used["strategy_return"] = data_used["return"] * positions.shift(1)

    return data_used

def calculate_equity_curve(strategy_returns):

    returns = strategy_returns.copy()

    returns.iloc[0] = 0

    equity_curve = (1 + returns).cumprod()

    return equity_curve

def run_mean_reversion(data, window, threshold):

    zscore = calculate_zscore(data, window)
    data["zscore"] = zscore

    signals = generate_signal(zscore, threshold)
    data["signal"] = signals

    positions = signal_to_position(signals)
    data["position"] = positions

    result = calculate_strategy_returns(data, positions)

    equity_curve = calculate_equity_curve(result["strategy_return"])
    result["equity"] = equity_curve

    return result


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

def calculate_pair_returns(data, positions):

    data_used = data.copy()

    data_used["return_1"] = data_used["symbol_1"].pct_change()
    data_used["return_2"] = data_used["symbol_2"].pct_change()

    position_1 = positions["position_1"].shift(1)
    position_2 = positions["position_2"].shift(1)

    data_used["strategy_return"] = position_1 * data_used["return_1"] + position_2 * data_used["return_2"]

    return data_used

def run_pairs_trading(data, alpha, beta, window, threshold):

    data_spread = calculate_spread(data, beta, alpha)

    zscore = calculate_zscore(data_spread, window, column="spread")
    data_spread["zscore"] = zscore

    position = generate_pair_positions(zscore, threshold, beta)

    data_spread["position_1"] = position["position_1"]
    data_spread["position_2"] = position["position_2"]

    result = calculate_pair_returns(data_spread, position)

    equity = calculate_equity_curve(result["strategy_return"])

    result["equity"] = equity

    return result


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






