import numpy as np
import pandas as pd
from .common import split_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from .walk_forward import generate_expanding_window_folds

def build_regression_dataset(data, momentum_window=5, volatility_window=5):

    return_1 = data["Close"].pct_change()

    momentum = data["Close"].pct_change(momentum_window)

    volatility = return_1.rolling(volatility_window).std()

    target = return_1.shift(-1)

    return_2 = return_1.shift(1)

    dataset = pd.DataFrame({
        "return_1" : return_1,
        "return_2" : return_2,
        "momentum" : momentum,
        "volatility" : volatility,
        "target" : target
    })

    dataset = dataset.dropna()

    return dataset

FEATURE_COLUMNS = ["return_1", "return_2", "momentum", "volatility"]

def prepare_regression_data(dataset, train_ratio=0.6, validation_ratio=0.2):

    train, validation, test = split_data(dataset, train_ratio, validation_ratio)

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["target"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test
    }

def fit_linear_regression(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model

def predict_linear_regression(model, X):

    return model.predict(X)

def evaluate_regression(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    directional_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

    return {
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": directional_accuracy
    }

def evaluate_regression_baselines(y_train, y_evaluation):

    zero_predictions = np.zeros(len(y_evaluation), dtype=float)

    train_mean = np.mean(y_train)
    train_mean_predictions = np.full(
        len(y_evaluation),
        train_mean,
        dtype=float
    )

    return {
        "zero_return": {
            "predictions": zero_predictions,
            "rmse": np.sqrt(mean_squared_error(y_evaluation, zero_predictions)),
            "r2": r2_score(y_evaluation, zero_predictions)
        },
        "train_mean": {
            "predictions": train_mean_predictions,
            "rmse": np.sqrt(mean_squared_error(y_evaluation, train_mean_predictions)),
            "r2": r2_score(y_evaluation, train_mean_predictions)
        },
        "always_positive": {
            "directional_accuracy": np.mean(np.asarray(y_evaluation) > 0)
        }
    }

def run_linear_regression_research(data, momentum_window=5, volatility_window=5, train_ratio=0.6,
                                    validation_ratio=0.2):

    dataset = build_regression_dataset(data, momentum_window, volatility_window)

    splits = prepare_regression_data(dataset, train_ratio, validation_ratio)

    model = fit_linear_regression(splits["X_train"], splits["y_train"])

    validation_predictions = predict_linear_regression(model, splits["X_validation"])

    validation_metrics = evaluate_regression(splits["y_validation"], validation_predictions)

    validation_baselines = evaluate_regression_baselines(
        splits["y_train"],
        splits["y_validation"]
    )

    test_predictions = predict_linear_regression(model, splits["X_test"])

    test_metrics = evaluate_regression(splits["y_test"], test_predictions)

    test_baselines = evaluate_regression_baselines(
        splits["y_train"],
        splits["y_test"]
    )

    return {
        "model": model,
        "dataset": dataset,
        "splits": splits,
        "validation_predictions": validation_predictions,
        "validation_metrics": validation_metrics,
        "validation_baselines": validation_baselines,
        "test_predictions": test_predictions,
        "test_metrics": test_metrics,
        "test_baselines": test_baselines
    }

def run_linear_regression_walk_forward(
    data,
    momentum_window=5,
    volatility_window=5,
    initial_train_size=252,
    test_size=21
):

    dataset = build_regression_dataset(
        data,
        momentum_window=momentum_window,
        volatility_window=volatility_window
    )
    folds = generate_expanding_window_folds(
        dataset,
        initial_train_size=initial_train_size,
        test_size=test_size
    )

    prediction_frames = []
    coefficient_rows = []
    fold_history = []

    for fold in folds:
        X_train = fold.train[FEATURE_COLUMNS]
        y_train = fold.train["target"]
        X_test = fold.test[FEATURE_COLUMNS]
        y_test = fold.test["target"]

        model = fit_linear_regression(X_train, y_train)
        predictions = predict_linear_regression(model, X_test)
        train_mean = float(np.mean(y_train))

        prediction_frames.append(pd.DataFrame({
            "actual": y_test,
            "prediction": predictions,
            "fold": fold.fold,
            "train_end": fold.train.index[-1],
            "zero_return_prediction": 0.0,
            "expanding_train_mean_prediction": train_mean
        }, index=fold.test.index))

        coefficient_row = {
            "fold": fold.fold,
            "train_start": fold.train.index[0],
            "train_end": fold.train.index[-1],
            "prediction_start": fold.test.index[0],
            "prediction_end": fold.test.index[-1],
            "train_size": len(fold.train),
            "prediction_size": len(fold.test),
            "intercept": float(model.intercept_)
        }
        coefficient_row.update({
            f"coef_{feature}": float(coefficient)
            for feature, coefficient in zip(FEATURE_COLUMNS, model.coef_)
        })
        coefficient_rows.append(coefficient_row)

        fold_history.append({
            "fold": fold.fold,
            "train_index": fold.train.index.copy(),
            "prediction_index": fold.test.index.copy(),
            "train_mean": train_mean
        })

    walk_forward_predictions = pd.concat(prediction_frames)

    if not walk_forward_predictions.index.is_monotonic_increasing:
        raise ValueError("walk-forward prediction index must be chronological")
    if not walk_forward_predictions.index.is_unique:
        raise ValueError("walk-forward prediction blocks must not overlap")

    actual = walk_forward_predictions["actual"]
    predictions = walk_forward_predictions["prediction"]
    zero_predictions = walk_forward_predictions["zero_return_prediction"]
    train_mean_predictions = walk_forward_predictions[
        "expanding_train_mean_prediction"
    ]

    metrics = evaluate_regression(actual, predictions)
    baselines = {
        "zero_return": {
            "predictions": zero_predictions.copy(),
            "rmse": np.sqrt(mean_squared_error(actual, zero_predictions)),
            "r2": r2_score(actual, zero_predictions)
        },
        "expanding_train_mean": {
            "predictions": train_mean_predictions.copy(),
            "rmse": np.sqrt(mean_squared_error(actual, train_mean_predictions)),
            "r2": r2_score(actual, train_mean_predictions)
        },
        "always_positive": {
            "directional_accuracy": np.mean(actual.to_numpy() > 0)
        }
    }

    return {
        "dataset": dataset,
        "predictions": walk_forward_predictions,
        "metrics": metrics,
        "baselines": baselines,
        "coefficient_history": pd.DataFrame(coefficient_rows),
        "folds": tuple(fold_history)
    }
