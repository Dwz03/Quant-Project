import pandas as pd
from .common import (calculate_zscore, calculate_equity_curve, calculate_zscore_with_history)

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

    data_used["strategy_return"] = (
        data_used["return"] * positions.shift(1)
    )

    data_used.loc[data_used.index[0], "strategy_return"] = 0.0

    return data_used

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

def run_mean_reversion_with_history(data, historical_data, window, threshold):

    data_used = data.copy()

    zscore = calculate_zscore_with_history(historical_data,data_used, window, column="Close")

    data_used["zscore"] = zscore

    signals = generate_signal(zscore, threshold)

    data_used["signal"] = signals

    positions = signal_to_position(signals)

    data_used["position"] = positions

    result = calculate_strategy_returns(data_used, positions)

    result["equity"] = calculate_equity_curve(result["strategy_return"])

    return result