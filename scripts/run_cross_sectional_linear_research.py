"""Run frozen pooled cross-sectional OLS on Research and Validation only."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_feature_screen import load_symbol_data
from src.data_loader import load_market_data
from src.research.cross_sectional_linear import (
    PRIMARY_FEATURES_V1,
    run_cross_sectional_linear_research,
)
from src.research.features import build_multi_asset_feature_panel
from src.research.universes import UNIVERSES, UNIVERSE_METADATA


DEFAULT_UNIVERSE = "large"
LOAD_START = "2021-01-01"
LOAD_END = "2026-01-01"
DEFAULT_MIN_SYMBOLS = 50


def minimum_symbols(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen cross-sectional linear-alpha research.",
    )
    parser.add_argument(
        "--universe",
        choices=tuple(UNIVERSES),
        default=DEFAULT_UNIVERSE,
    )
    parser.add_argument(
        "--min-symbols",
        type=minimum_symbols,
        default=DEFAULT_MIN_SYMBOLS,
    )
    return parser


def _print_evaluation(label: str, evaluation: dict) -> None:
    numerical = evaluation["numerical_metrics"]
    rank_ic = evaluation["rank_ic_summary"]
    print(f"{label}:")
    print(f"RMSE: {numerical['rmse']:.6f}")
    print(f"R2: {numerical['r2']:.6f}")
    print(f"Mean Prediction Rank IC: {rank_ic['mean_prediction_ic']:.6f}")
    print(f"Median Prediction Rank IC: {rank_ic['median_prediction_ic']:.6f}")
    print(f"Prediction Rank IC Std: {rank_ic['std_prediction_ic']:.6f}")
    print(f"Prediction ICIR: {rank_ic['prediction_icir']:.6f}")
    print(
        "Fraction Positive Prediction IC: "
        f"{rank_ic['fraction_positive_prediction_ic']:.2%}"
    )
    print(f"Number of IC Dates: {rank_ic['number_of_dates']}")
    print(
        "Mean Symbols per IC Date: "
        f"{rank_ic['mean_number_of_symbols']:.2f}"
    )


def display_results(result: dict, universe_name: str) -> None:
    metadata = UNIVERSE_METADATA[universe_name]
    print("Cross-Sectional Linear Alpha — Research and Validation Only")
    print(f"Universe: {metadata['version']} ({metadata['number_of_symbols']} symbols)")
    print(f"Frozen features: {', '.join(PRIMARY_FEATURES_V1)}")

    for model_name, model_result in result["models"].items():
        print(f"\nModel: {model_name}")
        print(f"Intercept: {model_result['intercept']:.6f}")
        for feature, coefficient in model_result["coefficients"].items():
            print(f"Coefficient {feature}: {coefficient:.6f}")
        print()
        _print_evaluation("Research fitted values", model_result["research"])
        print()
        _print_evaluation("Validation predictions", model_result["validation"])

    baselines = result["validation_baselines"]
    print("\nValidation Baselines:")
    print(f"Zero Return RMSE: {baselines['zero_return']['rmse']:.6f}")
    print(f"Zero Return R2: {baselines['zero_return']['r2']:.6f}")
    print(
        "Research Mean Return: "
        f"{baselines['research_mean']['research_mean']:.6f}"
    )
    print(f"Research Mean RMSE: {baselines['research_mean']['rmse']:.6f}")
    print(f"Research Mean R2: {baselines['research_mean']['r2']:.6f}")
    print("Equal Score Rank IC: undefined")

    print("\nValidation Model Comparison:")
    print(result["validation_comparison"].to_string(index=False))


def main(
    argv: Sequence[str] | None = None,
    loader: Callable[[str, str, str], pd.DataFrame] = load_market_data,
) -> dict:
    args = build_parser().parse_args(argv)
    symbols = UNIVERSES[args.universe]
    symbol_data = load_symbol_data(
        symbols=symbols,
        start=LOAD_START,
        end=LOAD_END,
        loader=loader,
    )
    panel = build_multi_asset_feature_panel(symbol_data)
    result = run_cross_sectional_linear_research(
        panel,
        min_symbols=args.min_symbols,
    )
    display_results(result, universe_name=args.universe)
    return result


if __name__ == "__main__":
    main()
