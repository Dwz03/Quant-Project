import pandas as pd
import numpy as np
import pytest

import src.research.linear_regression as linear_regression
from src.research.linear_regression import (build_regression_dataset, prepare_regression_data,
                                            fit_linear_regression, predict_linear_regression,
                                            evaluate_regression, evaluate_regression_baselines,
                                            run_linear_regression_research,
                                            run_linear_regression_walk_forward)


def test_build_regression_dataset_alignment():

    data = pd.DataFrame({
        "Close": [
            100,
            110,
            99,
            108.9,
            119.79,
            107.811,
            118.5921
        ]
    })

    result = build_regression_dataset(
        data,
        momentum_window=2,
        volatility_window=2
    )

    row = result.loc[3]

    assert row["return_1"] == pytest.approx(0.10)
    assert row["return_2"] == pytest.approx(-0.10)
    assert row["momentum"] == pytest.approx(-0.01)
    assert row["target"] == pytest.approx(0.10)
    assert not result.isna().any().any()

def test_prepare_regression_data():

    dataset = pd.DataFrame({
        "return_1": range(10),
        "return_2": range(10),
        "momentum": range(10),
        "volatility": range(10),
        "target": range(10)
    })

    result = prepare_regression_data(
        dataset,
        train_ratio=0.6,
        validation_ratio=0.2
    )

    assert len(result["X_train"]) == 6
    assert len(result["X_validation"]) == 2
    assert len(result["X_test"]) == 2

    assert result["X_train"].index[-1] == 5
    assert result["X_validation"].index[0] == 6
    assert result["X_test"].index[0] == 8

    assert "target" not in result["X_train"].columns

def test_fit_linear_regression():

    X = pd.DataFrame({
        "return_1": [1, 2, 3, 4],
        "return_2": [0, 0, 0, 0],
        "momentum": [0, 0, 0, 0],
        "volatility": [0, 0, 0, 0]
    })

    y = pd.Series([2, 4, 6, 8])

    model = fit_linear_regression(X, y)

    predictions = predict_linear_regression(
        model,
        X
    )

    assert predictions[0] == pytest.approx(2)
    assert predictions[-1] == pytest.approx(8)

def test_evaluate_regression():

    y_true = np.array([
        0.01,
        -0.02,
        0.03,
        -0.01
    ])

    y_pred = np.array([
        0.02,
        -0.01,
        0.01,
        0.02
    ])

    result = evaluate_regression(
        y_true,
        y_pred
    )

    assert "rmse" in result
    assert "r2" in result
    assert "directional_accuracy" in result

    assert result[
        "directional_accuracy"
    ] == pytest.approx(0.75)

def test_evaluate_regression_baselines():

    y_train = pd.Series([0.1, 0.3])
    y_evaluation = pd.Series([-0.2, 0.0, 0.4])

    result = evaluate_regression_baselines(
        y_train,
        y_evaluation
    )

    assert np.array_equal(
        result["zero_return"]["predictions"],
        np.zeros(len(y_evaluation))
    )
    assert np.array_equal(
        result["train_mean"]["predictions"],
        np.full(len(y_evaluation), 0.2)
    )
    assert result["always_positive"]["directional_accuracy"] == pytest.approx(
        1 / 3
    )
    assert "rmse" in result["zero_return"]
    assert "r2" in result["zero_return"]
    assert "rmse" in result["train_mean"]
    assert "r2" in result["train_mean"]

def test_train_mean_baseline_does_not_depend_on_evaluation_values():

    y_train = pd.Series([-0.03, 0.01, 0.08])

    first_result = evaluate_regression_baselines(
        y_train,
        pd.Series([-10.0, -5.0])
    )
    second_result = evaluate_regression_baselines(
        y_train,
        pd.Series([100.0, 200.0, 300.0])
    )

    expected_mean = y_train.mean()
    assert np.all(first_result["train_mean"]["predictions"] == expected_mean)
    assert np.all(second_result["train_mean"]["predictions"] == expected_mean)

def test_run_linear_regression_research():

    prices = pd.Series(
        [
            100, 101, 102, 101, 103,
            104, 105, 103, 106, 107,
            108, 107, 109, 110, 111,
            109, 112, 113, 114, 115,
            113, 116, 117, 118, 119,
            120, 118, 121, 122, 123
        ]
    )

    data = pd.DataFrame({
        "Close": prices
    })

    result = run_linear_regression_research(
        data,
        momentum_window=3,
        volatility_window=3,
        train_ratio=0.6,
        validation_ratio=0.2
    )

    assert "model" in result
    assert "validation_metrics" in result
    assert "test_metrics" in result
    assert "validation_baselines" in result
    assert "test_baselines" in result

    assert "rmse" in result["validation_metrics"]
    assert "r2" in result["validation_metrics"]
    assert "directional_accuracy" in result["validation_metrics"]

    train_mean = result["splits"]["y_train"].mean()
    validation_mean = result["splits"]["y_validation"].mean()
    test_mean = result["splits"]["y_test"].mean()

    validation_predictions = result["validation_baselines"]["train_mean"][
        "predictions"
    ]
    test_predictions = result["test_baselines"]["train_mean"]["predictions"]

    assert np.all(validation_predictions == train_mean)
    assert np.all(test_predictions == train_mean)
    assert train_mean != pytest.approx(validation_mean)
    assert train_mean != pytest.approx(test_mean)


def walk_forward_dataset():
    values = np.arange(10, dtype=float)
    return pd.DataFrame({
        "return_1": values,
        "return_2": values ** 2,
        "momentum": np.sin(values),
        "volatility": values + 1.0,
        "target": [-0.04, 0.01, 0.03, -0.02, 0.05,
                   -0.01, 0.02, 0.04, -0.03, 0.06]
    }, index=pd.date_range("2024-01-01", periods=10))


def test_linear_regression_walk_forward_expands_without_leakage(monkeypatch):
    dataset = walk_forward_dataset()
    monkeypatch.setattr(
        linear_regression,
        "build_regression_dataset",
        lambda data, momentum_window, volatility_window: dataset.copy()
    )

    result = run_linear_regression_walk_forward(
        pd.DataFrame(),
        momentum_window=5,
        volatility_window=5,
        initial_train_size=4,
        test_size=2
    )

    predictions = result["predictions"]
    expected_index = dataset.index[4:]
    assert predictions.index.equals(expected_index)
    assert predictions.index.is_monotonic_increasing
    assert predictions.index.is_unique
    assert predictions["actual"].equals(dataset.loc[expected_index, "target"])
    assert predictions["fold"].tolist() == [1, 1, 2, 2, 3, 3]

    folds = result["folds"]
    assert [len(fold["train_index"]) for fold in folds] == [4, 6, 8]
    assert [len(fold["prediction_index"]) for fold in folds] == [2, 2, 2]
    for fold in folds:
        assert fold["train_index"][-1] < fold["prediction_index"][0]
        assert not fold["train_index"].isin(fold["prediction_index"]).any()

        expected_mean = dataset.loc[fold["train_index"], "target"].mean()
        fold_predictions = predictions.loc[fold["prediction_index"]]
        assert np.all(
            fold_predictions["expanding_train_mean_prediction"] == expected_mean
        )

    assert folds[0]["prediction_index"].isin(folds[1]["train_index"]).all()
    assert folds[1]["prediction_index"].isin(folds[2]["train_index"]).all()

    baseline_predictions = result["baselines"]["expanding_train_mean"][
        "predictions"
    ]
    assert baseline_predictions.index.equals(predictions.index)
    assert result["baselines"]["zero_return"]["predictions"].index.equals(
        predictions.index
    )

    coefficient_history = result["coefficient_history"]
    assert len(coefficient_history) == len(folds)
    assert coefficient_history["train_size"].tolist() == [4, 6, 8]
    assert {
        "intercept",
        "coef_return_1",
        "coef_return_2",
        "coef_momentum",
        "coef_volatility",
    }.issubset(coefficient_history.columns)


def test_walk_forward_first_fold_fit_is_unchanged_by_future_targets(monkeypatch):
    datasets = {"current": walk_forward_dataset()}
    monkeypatch.setattr(
        linear_regression,
        "build_regression_dataset",
        lambda data, momentum_window, volatility_window: datasets["current"].copy()
    )

    original = run_linear_regression_walk_forward(
        pd.DataFrame(),
        initial_train_size=4,
        test_size=2
    )

    changed = walk_forward_dataset()
    changed.loc[changed.index[4:], "target"] = np.arange(100.0, 106.0)
    datasets["current"] = changed
    changed_result = run_linear_regression_walk_forward(
        pd.DataFrame(),
        initial_train_size=4,
        test_size=2
    )

    coefficient_columns = [
        "intercept",
        "coef_return_1",
        "coef_return_2",
        "coef_momentum",
        "coef_volatility",
    ]
    np.testing.assert_allclose(
        original["coefficient_history"].loc[0, coefficient_columns].to_numpy(
            dtype=float
        ),
        changed_result["coefficient_history"].loc[
            0, coefficient_columns
        ].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        original["predictions"].iloc[:2]["prediction"],
        changed_result["predictions"].iloc[:2]["prediction"],
    )
