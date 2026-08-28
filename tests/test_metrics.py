import pandas as pd
import pytest
import numpy as np

from src.metrics import total_return, annualized_volatility, sharpe_ratio,max_drawdown, performance_summary

def test_total_return():

    returns = pd.Series([0.10, -0.10])

    assert total_return(returns) == pytest.approx(-0.01)

def test_annualized_volatility():

    returns = pd.Series([0.01, -0.01])

    std = np.sqrt(0.0002)

    assert annualized_volatility(returns) ==  pytest.approx(std * np.sqrt(252))

def test_sharpe_ratio():

    returns = pd.Series([0.01, 0.03])

    mean_return = 0.02
    std = np.sqrt(0.0002)

    expected = mean_return * np.sqrt(252) / std

    assert sharpe_ratio(returns) == pytest.approx(expected)

def test_sharpe_ratio_zero_volatility():

    returns = pd.Series([0.01, 0.01, 0.01])

    assert sharpe_ratio(returns) == 0

def test_max_drawdown():

    returns = pd.Series([0.10, -0.20])

    assert max_drawdown(returns) == pytest.approx(-0.20)

def test_performance_summary():

    returns = pd.Series([0.01, 0.03])

    summary = performance_summary(returns)

    assert summary["Total Return"] == pytest.approx(total_return(returns))
    assert summary["Annualized Volatility"] == pytest.approx(annualized_volatility(returns))
    assert summary["Sharpe Ratio"] == pytest.approx(sharpe_ratio(returns))
    assert summary["Max Drawdown"] == pytest.approx(max_drawdown(returns))



