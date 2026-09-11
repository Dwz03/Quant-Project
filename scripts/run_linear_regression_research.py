"""Run the linear-regression predictive research pipeline for one symbol."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import clean_market_data, load_market_data
from src.research.linear_regression import (
    run_linear_regression_research,
    run_linear_regression_walk_forward,
)


DEFAULT_START = "2021-01-01"
DEFAULT_END = "2026-01-01"
DEFAULT_MOMENTUM_WINDOW = 5
DEFAULT_VOLATILITY_WINDOW = 5
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VALIDATION_RATIO = 0.2
DEFAULT_INITIAL_TRAIN_SIZE = 252
DEFAULT_TEST_SIZE = 21


def positive_integer(value: str) -> int:
    """Parse a strictly positive integer for a rolling-window argument."""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed_value


def load_symbol_data(
    symbol: str,
    start: str,
    end: str,
    loader: Callable[[str, str, str], pd.DataFrame] = load_market_data,
) -> pd.DataFrame:
    """Load and clean one symbol's adjusted daily close history."""
    raw_data = loader(symbol, start, end)

    try:
        data = clean_market_data(raw_data)
    except KeyError as exc:
        raise ValueError(f"market data for {symbol} must contain Close") from exc

    close_data = data[["Close"]].astype(float)
    if close_data.empty:
        raise ValueError(f"market data for {symbol} has no close observations")

    close_values = close_data["Close"].to_numpy(dtype=float)
    if not np.isfinite(close_values).all() or (close_values <= 0).any():
        raise ValueError("adjusted-close prices must be finite and positive")

    return close_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the linear-regression alpha model on held-out data.",
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--momentum-window",
        type=positive_integer,
        default=DEFAULT_MOMENTUM_WINDOW,
    )
    parser.add_argument(
        "--volatility-window",
        type=positive_integer,
        default=DEFAULT_VOLATILITY_WINDOW,
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
    )
    parser.add_argument(
        "--show-coefficients",
        action="store_true",
        help="print the fitted intercept and feature coefficients",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="run expanding-window out-of-sample evaluation",
    )
    parser.add_argument(
        "--initial-train-size",
        type=positive_integer,
        default=DEFAULT_INITIAL_TRAIN_SIZE,
    )
    parser.add_argument(
        "--test-size",
        type=positive_integer,
        default=DEFAULT_TEST_SIZE,
    )
    return parser


def print_metrics(metrics: dict[str, float]) -> None:
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")


def print_baselines(baselines: dict) -> None:
    zero_return = baselines["zero_return"]
    train_mean = baselines["train_mean"]
    always_positive = baselines["always_positive"]

    print("Baselines:")
    print(f"Zero Return RMSE: {zero_return['rmse']:.6f}")
    print(f"Zero Return R2: {zero_return['r2']:.6f}")
    print(f"Train Mean RMSE: {train_mean['rmse']:.6f}")
    print(f"Train Mean R2: {train_mean['r2']:.6f}")
    print(
        "Always Positive Directional Accuracy: "
        f"{always_positive['directional_accuracy']:.2%}"
    )


def print_split_results(label: str, metrics: dict, baselines: dict) -> None:
    print(f"{label}:")
    print("Linear Regression")
    print_metrics(metrics)
    print()
    print_baselines(baselines)


def display_results(result: dict, show_coefficients: bool = False) -> None:
    """Print validation/test metrics and, when requested, fitted parameters."""
    print_split_results(
        "Validation",
        result["validation_metrics"],
        result["validation_baselines"],
    )
    print()
    print_split_results(
        "Test",
        result["test_metrics"],
        result["test_baselines"],
    )

    if show_coefficients:
        model = result["model"]
        feature_names = result["splits"]["X_train"].columns

        print("\nCoefficients:")
        for feature_name, coefficient in zip(feature_names, model.coef_):
            print(f"{feature_name}: {coefficient:.6f}")
        print(f"Intercept: {model.intercept_:.6f}")


def display_walk_forward_results(
    result: dict,
    show_coefficients: bool = False,
) -> None:
    metrics = result["metrics"]
    baselines = result["baselines"]
    zero_return = baselines["zero_return"]
    train_mean = baselines["expanding_train_mean"]
    always_positive = baselines["always_positive"]

    print("Walk-Forward Linear Regression:")
    print_metrics(metrics)
    print("\nWalk-Forward Baselines:")
    print(f"Zero Return RMSE: {zero_return['rmse']:.6f}")
    print(f"Zero Return R2: {zero_return['r2']:.6f}")
    print(f"Expanding Train Mean RMSE: {train_mean['rmse']:.6f}")
    print(f"Expanding Train Mean R2: {train_mean['r2']:.6f}")
    print(
        "Always Positive Directional Accuracy: "
        f"{always_positive['directional_accuracy']:.2%}"
    )

    if show_coefficients:
        print("\nCoefficient History:")
        print(result["coefficient_history"].to_string(index=False))


def main(
    argv: Sequence[str] | None = None,
    loader: Callable[[str, str, str], pd.DataFrame] = load_market_data,
) -> dict:
    args = build_parser().parse_args(argv)

    if pd.Timestamp(args.start) >= pd.Timestamp(args.end):
        raise ValueError("start must be earlier than end")

    data = load_symbol_data(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        loader=loader,
    )

    if args.walk_forward:
        result = run_linear_regression_walk_forward(
            data=data,
            momentum_window=args.momentum_window,
            volatility_window=args.volatility_window,
            initial_train_size=args.initial_train_size,
            test_size=args.test_size,
        )
        display_walk_forward_results(
            result,
            show_coefficients=args.show_coefficients,
        )
        return result

    if not 0 < args.train_ratio < 1:
        raise ValueError("train ratio must be between 0 and 1")
    if not 0 < args.validation_ratio < 1:
        raise ValueError("validation ratio must be between 0 and 1")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("train ratio plus validation ratio must be less than 1")

    result = run_linear_regression_research(
        data=data,
        momentum_window=args.momentum_window,
        volatility_window=args.volatility_window,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )

    display_results(result, show_coefficients=args.show_coefficients)
    return result


if __name__ == "__main__":
    main()
