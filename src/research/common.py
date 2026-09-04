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

def calculate_zscore(data, window, column = "Close"):

    data_used = data.copy()

    data_used["rolling_mean"] = data_used[column].rolling(window).mean()
    data_used["std"] = data_used[column].rolling(window).std()
    data_used["zscore"] = (data_used[column] - data_used["rolling_mean"]) / data_used["std"]

    return data_used["zscore"]

def calculate_zscore_with_history(historical_data, data, window, column):

    combined = pd.concat([
        historical_data,
        data
    ])

    combined_zscore = calculate_zscore(
        combined,
        window,
        column=column
    )

    zscore = combined_zscore.iloc[-len(data):]

    return zscore

def calculate_equity_curve(strategy_returns):

    returns = strategy_returns.copy()

    returns.iloc[0] = 0

    equity_curve = (1 + returns).cumprod()

    return equity_curve

def normalize_positions(positions):

    gross = positions.abs().sum(axis=1)

    normalized = positions.div(gross, axis=0)

    normalized = normalized.fillna(0)

    return normalized

