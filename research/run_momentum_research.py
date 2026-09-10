"""Run the standardized momentum experiment on adjusted daily close prices."""

import argparse
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.data_loader import clean_market_data, load_market_data
from src.research import ExperimentConfig
from src.research.momentum_experiment import (
    MomentumExperimentResult,
    MomentumBenchmarkResult,
    MomentumWalkForwardResult,
    MOMENTUM_NAME,
    compare_momentum_with_benchmarks,
    run_momentum_experiment,
    run_momentum_walk_forward,
    summarize_momentum_experiment,
)
from src.research.walk_forward import WalkForwardConfig


DEFAULT_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT")
DEFAULT_START = "2021-01-01"
DEFAULT_END = "2026-01-01"
DEFAULT_LOOKBACKS = (5, 10, 20, 60)
DEFAULT_TARGET_WEIGHT = 0.25
DEFAULT_TRANSACTION_COST_RATE = 0.001
DEFAULT_OUTPUT_DIR = Path("research_results/momentum")
DEFAULT_INITIAL_TRAIN_SIZE = 252
DEFAULT_VALIDATION_SIZE = 63
DEFAULT_TEST_SIZE = 63


def load_adjusted_close_prices(
    symbols: Sequence[str],
    start: str,
    end: str,
    loader: Callable[[str, str, str], pd.DataFrame] = load_market_data,
) -> pd.DataFrame:
    """Load adjusted closes and align the universe on common trading dates."""
    symbols_used = list(symbols)

    if not symbols_used:
        raise ValueError("symbols must contain at least one symbol")

    if len(set(symbols_used)) != len(symbols_used):
        raise ValueError("symbols must be unique")

    closes = []
    for symbol in symbols_used:
        raw_data = loader(symbol, start, end)

        try:
            prepared_data = clean_market_data(raw_data)
        except KeyError as exc:
            raise ValueError(f"market data for {symbol} must contain Close") from exc

        closes.append(prepared_data["Close"].astype(float).rename(symbol))

    prices = pd.concat(closes, axis=1, join="inner").sort_index()

    if prices.empty:
        raise ValueError("symbols have no overlapping adjusted-close observations")

    price_values = prices.to_numpy(dtype=float)
    if not np.isfinite(price_values).all() or (price_values <= 0).any():
        raise ValueError("aligned adjusted-close prices must be finite and positive")

    return prices


def save_research_outputs(
    result: MomentumExperimentResult,
    summary: dict,
    benchmarks: MomentumBenchmarkResult,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Save candidate, summary, and benchmark comparison CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_results.csv"
    summary_path = output_dir / "summary.csv"
    benchmark_path = output_dir / "benchmark_comparison.csv"

    result.candidate_results.to_csv(candidate_path, index=False)

    summary_table = summary["performance"].reset_index()
    summary_table.insert(0, "experiment_name", summary["experiment_name"])
    for name, value in summary["selected_parameters"].items():
        summary_table[f"selected_{name}"] = value
    summary_table.to_csv(summary_path, index=False)
    benchmarks.comparison.reset_index().to_csv(benchmark_path, index=False)

    return candidate_path, summary_path, benchmark_path


def display_research_results(
    result: MomentumExperimentResult,
    summary: dict,
    benchmarks: MomentumBenchmarkResult,
    symbols: Sequence[str],
    configured_start: str,
    configured_end: str,
    transaction_cost_rate: float,
    output_paths: tuple[Path, Path, Path],
) -> None:
    """Print the configured experiment, candidates, selection, and split report."""
    prices = result.experiment.splits
    actual_start = prices.train.index.min()
    actual_end = prices.test.index.max()

    print(f"Experiment: {summary['experiment_name']}")
    print(f"Universe: {', '.join(symbols)}")
    print(f"Configured date range: {configured_start} to {configured_end}")
    print(f"Available aligned dates: {actual_start.date()} to {actual_end.date()}")
    print(f"Transaction cost rate: {transaction_cost_rate:.6f}")

    print("\nCANDIDATE PARAMETER RESULTS")
    print(result.candidate_results.to_string(index=False))

    print("\nSelected parameters")
    for name, value in summary["selected_parameters"].items():
        print(f"{name}: {value}")

    print("\nVALIDATION — model-selection performance")
    print(summary["performance"].loc["validation"].to_string())

    print("\nTEST — final held-out evaluation")
    print(summary["performance"].loc["test"].to_string())

    print("\nVALIDATION — model-selection comparison")
    print(benchmarks.comparison.xs("validation", level="split").to_string())

    print("\nTEST — final held-out comparison")
    print(benchmarks.comparison.xs("test", level="split").to_string())

    print("\nSaved generated outputs")
    for output_path in output_paths:
        print(output_path)


def save_walk_forward_outputs(
    result: MomentumWalkForwardResult,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Save fold, Momentum summary, and benchmark comparison CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = output_dir / "walk_forward_folds.csv"
    summary_path = output_dir / "walk_forward_summary.csv"
    benchmark_path = output_dir / "walk_forward_benchmark_comparison.csv"

    result.momentum.fold_table.to_csv(folds_path, index=False)

    summary = result.comparison.loc[[MOMENTUM_NAME]].reset_index(
        names="strategy"
    )
    summary.insert(0, "folds", len(result.momentum.folds))
    summary.to_csv(summary_path, index=False)
    result.comparison.reset_index(names="strategy").to_csv(
        benchmark_path,
        index=False,
    )

    return folds_path, summary_path, benchmark_path


def display_walk_forward_results(
    result: MomentumWalkForwardResult,
    symbols: Sequence[str],
    configured_start: str,
    configured_end: str,
    transaction_cost_rate: float,
    output_paths: tuple[Path, Path, Path],
) -> None:
    """Print walk-forward configuration, fold stability, and stitched results."""
    config = result.momentum.config

    print("WALK-FORWARD CONFIGURATION")
    print(f"Universe: {', '.join(symbols)}")
    print(f"Configured date range: {configured_start} to {configured_end}")
    print(f"Initial train observations: {config.initial_train_size}")
    print(f"Validation observations: {config.validation_size}")
    print(f"Test observations: {config.test_size}")
    print(f"Step observations: {config.step_size}")
    print(f"Transaction cost rate: {transaction_cost_rate:.6f}")

    print("\nFOLD RESULTS")
    print(result.momentum.fold_table.to_string(index=False))

    parameter_columns = [
        "fold",
        "train_end",
        "validation_end",
        "test_end",
        *[
            column
            for column in result.momentum.fold_table.columns
            if column.startswith("selected_") and column != "selected_parameters"
        ],
    ]
    print("\nPARAMETER STABILITY")
    print(result.momentum.fold_table[parameter_columns].to_string(index=False))

    print("\nSTITCHED OOS MOMENTUM PERFORMANCE")
    print(result.comparison.loc[MOMENTUM_NAME].to_string())

    print("\nSTITCHED OOS BENCHMARK COMPARISON")
    print(result.comparison.to_string())

    print("\nSaved generated outputs")
    for output_path in output_paths:
        print(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standardized momentum research experiment.",
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        default=list(DEFAULT_LOOKBACKS),
    )
    parser.add_argument("--target-weight", type=float, default=DEFAULT_TARGET_WEIGHT)
    parser.add_argument(
        "--transaction-cost-rate",
        type=float,
        default=DEFAULT_TRANSACTION_COST_RATE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="run expanding-window robustness instead of the fixed split",
    )
    parser.add_argument(
        "--initial-train-size",
        type=int,
        default=DEFAULT_INITIAL_TRAIN_SIZE,
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=DEFAULT_VALIDATION_SIZE,
    )
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--step-size", type=int)
    return parser


def main(
    argv: Sequence[str] | None = None,
    loader: Callable[[str, str, str], pd.DataFrame] = load_market_data,
) -> MomentumExperimentResult | MomentumWalkForwardResult:
    args = build_parser().parse_args(argv)

    if pd.Timestamp(args.start) >= pd.Timestamp(args.end):
        raise ValueError("start must be earlier than end")

    prices = load_adjusted_close_prices(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        loader=loader,
    )

    if args.walk_forward:
        walk_forward_config = WalkForwardConfig(
            initial_train_size=args.initial_train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            step_size=args.step_size,
        )
        walk_forward_result = run_momentum_walk_forward(
            data=prices,
            config=walk_forward_config,
            transaction_cost_rate=args.transaction_cost_rate,
            lookbacks=args.lookbacks,
            target_weight=args.target_weight,
            allow_short=False,
        )
        walk_forward_paths = save_walk_forward_outputs(
            walk_forward_result,
            args.output_dir,
        )
        display_walk_forward_results(
            result=walk_forward_result,
            symbols=args.symbols,
            configured_start=args.start,
            configured_end=args.end,
            transaction_cost_rate=args.transaction_cost_rate,
            output_paths=walk_forward_paths,
        )
        return walk_forward_result

    config = ExperimentConfig(
        name="momentum-adjusted-close-daily",
        train_ratio=0.60,
        validation_ratio=0.20,
        transaction_cost_rate=args.transaction_cost_rate,
    )
    result = run_momentum_experiment(
        data=prices,
        config=config,
        lookbacks=args.lookbacks,
        target_weights=[args.target_weight],
        allow_short=False,
    )
    summary = summarize_momentum_experiment(result)
    benchmarks = compare_momentum_with_benchmarks(result)
    output_paths = save_research_outputs(
        result,
        summary,
        benchmarks,
        args.output_dir,
    )
    display_research_results(
        result=result,
        summary=summary,
        benchmarks=benchmarks,
        symbols=args.symbols,
        configured_start=args.start,
        configured_end=args.end,
        transaction_cost_rate=args.transaction_cost_rate,
        output_paths=output_paths,
    )

    return result


if __name__ == "__main__":
    main()
