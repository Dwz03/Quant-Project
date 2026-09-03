import pandas as pd

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

def calculate_zscore(data, window):

    data["rolling_mean"] = data["Close"].rolling(window).mean()
    data["std"] = data["Close"].rolling(window).std()
    data["zscore"] = (data["Close"] - data["rolling_mean"]) / data["std"]

    return data["zscore"]

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

    data["return"] = data["Close"].pct_change()
    data["strategy_return"] = data["return"] * positions.shift(1)

    return data

def calculate_equity_curve(strategy_returns):

    returns = strategy_returns.copy()

    returns.iloc[0] = 0

    equity_curve = (1 + returns).cumprod()

    return equity_curve

def run_mean_reversion(data, window, threshold):

    zscore = calculate_zscore(data, window)

    signals = generate_signal(zscore, threshold)
    data["signal"] = signals

    positions = signal_to_position(signals)
    data["position"] = positions

    result = calculate_strategy_returns(data, positions)

    equity_curve = calculate_equity_curve(result["strategy_return"])
    data["equity"] = equity_curve

    return data