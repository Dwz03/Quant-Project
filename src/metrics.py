import pandas as pd
import numpy as np

def performance_summary(returns):

    summary = {
        "Total Return": total_return(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns),
        "Max Drawdown": max_drawdown(returns)
    }

    return summary

def total_return(returns):

    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    return (1 + returns).prod() - 1

def annualized_volatility(returns):

    returns = returns.dropna()

    std = returns.std()

    annual_vol = std * np.sqrt(252)

    return annual_vol

def sharpe_ratio(returns):

    returns = returns.dropna()

    mean_return = returns.mean()
    std_returns = returns.std()

    if np.isclose(std_returns, 0):
        return np.nan

    return mean_return * np.sqrt(252) / std_returns

def max_drawdown(returns):

    returns = returns.dropna()

    equity = (1 + returns).cumprod()

    running_max = equity.cummax()

    drawdown = equity / running_max - 1

    return drawdown.min()