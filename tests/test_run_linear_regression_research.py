from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.run_linear_regression_research as runner


def test_load_symbol_data_uses_loader_and_returns_ordered_clean_closes():
    calls = []

    def fake_loader(symbol, start, end):
        calls.append((symbol, start, end))
        return pd.DataFrame(
            {"Close": [102.0, 100.0, 101.0, 999.0], "Open": [0.0] * 4},
            index=pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
        )

    result = runner.load_symbol_data(
        symbol="TEST",
        start="2024-01-01",
        end="2024-02-01",
        loader=fake_loader,
    )

    assert calls == [("TEST", "2024-01-01", "2024-02-01")]
    assert result.columns.tolist() == ["Close"]
    assert result.index.tolist() == list(pd.date_range("2024-01-01", periods=3))
    assert result["Close"].tolist() == [100.0, 101.0, 102.0]


def test_main_forwards_cli_options_and_prints_metrics_and_coefficients(
    monkeypatch,
    capsys,
):
    loader_calls = []
    research_call = {}
    index = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])

    def fake_loader(symbol, start, end):
        loader_calls.append((symbol, start, end))
        return pd.DataFrame({"Close": [102.0, 100.0, 101.0]}, index=index)

    expected_result = {
        "model": SimpleNamespace(
            coef_=np.array([0.1, -0.2, 0.3, -0.4]),
            intercept_=0.005,
        ),
        "splits": {
            "X_train": pd.DataFrame(
                columns=["return_1", "return_2", "momentum", "volatility"]
            )
        },
        "validation_metrics": {
            "rmse": 0.0123456,
            "r2": 0.25,
            "directional_accuracy": 0.75,
        },
        "validation_baselines": {
            "zero_return": {"rmse": 0.02, "r2": -0.1},
            "train_mean": {"rmse": 0.019, "r2": -0.05},
            "always_positive": {"directional_accuracy": 0.55},
        },
        "test_metrics": {
            "rmse": 0.0234567,
            "r2": -0.5,
            "directional_accuracy": 0.6,
        },
        "test_baselines": {
            "zero_return": {"rmse": 0.03, "r2": -0.2},
            "train_mean": {"rmse": 0.029, "r2": -0.15},
            "always_positive": {"directional_accuracy": 0.45},
        },
    }

    def fake_research(**kwargs):
        research_call.update(kwargs)
        return expected_result

    monkeypatch.setattr(runner, "run_linear_regression_research", fake_research)

    result = runner.main(
        argv=[
            "--symbol",
            "TEST",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-01",
            "--momentum-window",
            "7",
            "--volatility-window",
            "11",
            "--train-ratio",
            "0.5",
            "--validation-ratio",
            "0.25",
            "--show-coefficients",
        ],
        loader=fake_loader,
    )

    assert result is expected_result
    assert loader_calls == [("TEST", "2024-01-01", "2024-02-01")]
    assert research_call["momentum_window"] == 7
    assert research_call["volatility_window"] == 11
    assert research_call["train_ratio"] == 0.5
    assert research_call["validation_ratio"] == 0.25
    assert research_call["data"].index.is_monotonic_increasing

    output = capsys.readouterr().out
    assert "Validation:\nLinear Regression\nRMSE: 0.012346\nR2: 0.250000" in output
    assert "Directional Accuracy: 75.00%" in output
    assert "Zero Return RMSE: 0.020000" in output
    assert "Zero Return R2: -0.100000" in output
    assert "Train Mean RMSE: 0.019000" in output
    assert "Train Mean R2: -0.050000" in output
    assert "Always Positive Directional Accuracy: 55.00%" in output
    assert "Test:\nLinear Regression\nRMSE: 0.023457\nR2: -0.500000" in output
    assert "Directional Accuracy: 60.00%" in output
    assert "Zero Return RMSE: 0.030000" in output
    assert "Zero Return R2: -0.200000" in output
    assert "Train Mean RMSE: 0.029000" in output
    assert "Train Mean R2: -0.150000" in output
    assert "Always Positive Directional Accuracy: 45.00%" in output
    assert "return_1: 0.100000" in output
    assert "volatility: -0.400000" in output
    assert "Intercept: 0.005000" in output


def test_main_walk_forward_path_prints_metrics_and_baselines(monkeypatch, capsys):
    calls = {}
    index = pd.date_range("2024-01-01", periods=3)

    def fake_loader(symbol, start, end):
        return pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=index)

    expected_result = {
        "metrics": {
            "rmse": 0.011,
            "r2": 0.12,
            "directional_accuracy": 0.625,
        },
        "baselines": {
            "zero_return": {"rmse": 0.013, "r2": -0.02},
            "expanding_train_mean": {"rmse": 0.012, "r2": -0.01},
            "always_positive": {"directional_accuracy": 0.5},
        },
        "coefficient_history": pd.DataFrame(),
    }

    def fake_walk_forward(**kwargs):
        calls.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        runner,
        "run_linear_regression_walk_forward",
        fake_walk_forward,
    )
    monkeypatch.setattr(
        runner,
        "run_linear_regression_research",
        lambda **kwargs: pytest.fail("fixed-split path should not run"),
    )

    result = runner.main(
        argv=[
            "--symbol", "TEST",
            "--walk-forward",
            "--momentum-window", "7",
            "--volatility-window", "11",
            "--initial-train-size", "50",
            "--test-size", "10",
        ],
        loader=fake_loader,
    )

    assert result is expected_result
    assert calls["momentum_window"] == 7
    assert calls["volatility_window"] == 11
    assert calls["initial_train_size"] == 50
    assert calls["test_size"] == 10

    output = capsys.readouterr().out
    assert "Walk-Forward Linear Regression:" in output
    assert "RMSE: 0.011000" in output
    assert "R2: 0.120000" in output
    assert "Directional Accuracy: 62.50%" in output
    assert "Walk-Forward Baselines:" in output
    assert "Zero Return RMSE: 0.013000" in output
    assert "Expanding Train Mean RMSE: 0.012000" in output
    assert "Always Positive Directional Accuracy: 50.00%" in output
