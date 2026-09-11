"""Run Validation economic backtest for the frozen volatility alpha."""

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
from src.research.universes import UNIVERSES, UNIVERSE_METADATA
from src.research.volatility_factor_backtest import (
    COST_BPS,
    FROZEN_ALPHA_V1,
    run_volatility_factor_backtest,
)


DEFAULT_UNIVERSE = "large"
# Fixed pre-Validation warm-up for the 20-return rolling feature.
LOAD_START = "2023-11-01"
LOAD_END = "2026-01-01"
# Fixed pre-Holdout warm-up for the unchanged 20-return rolling feature.
HOLDOUT_LOAD_START = "2025-11-01"


def holdout_load_end(as_of: pd.Timestamp | None = None) -> str:
    """Return the exclusive end date using the New York market calendar day."""
    if as_of is None:
        current_date = pd.Timestamp.now(tz="America/New_York").normalize()
    else:
        current_date = pd.Timestamp(as_of)
        if current_date.tzinfo is not None:
            current_date = current_date.tz_convert("America/New_York")
        current_date = current_date.normalize()

    return current_date.strftime("%Y-%m-%d")


def load_adjusted_ohlc_data(
    symbols: Sequence[str],
    start: str,
    end: str,
    loader: Callable[[str, str, str], pd.DataFrame] = load_market_data,
) -> dict[str, pd.DataFrame]:
    """Load adjusted Open and Close data using the project loader."""
    result = {}
    for symbol in symbols:
        cleaned = clean_market_data(loader(symbol, start, end))
        missing = {"Open", "Close"}.difference(cleaned.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"market data for {symbol} is missing: {names}")
        prices = cleaned[["Open", "Close"]].astype(float)
        if prices.empty:
            raise ValueError(f"market data for {symbol} cannot be empty")
        close = prices["Close"].to_numpy(dtype=float)
        available_open = prices["Open"].dropna().to_numpy(dtype=float)
        if not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError(f"Close prices for {symbol} must be finite and positive")
        if len(available_open) == 0:
            raise ValueError(f"market data for {symbol} has no valid Open prices")
        if (
            not np.isfinite(available_open).all()
            or (available_open <= 0).any()
        ):
            raise ValueError(
                f"available Open prices for {symbol} must be finite and positive"
            )
        result[symbol] = prices
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Validation or guarded one-time Holdout economic research."
        ),
    )
    parser.add_argument(
        "--universe",
        choices=tuple(UNIVERSES),
        default=DEFAULT_UNIVERSE,
    )
    parser.add_argument(
        "--period",
        choices=("validation", "holdout"),
        default="validation",
    )
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="explicitly authorize the one-time confirmatory Holdout evaluation",
    )
    return parser


def _print_performance(label: str, metrics: dict) -> None:
    print(f"{label}:")
    print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {metrics['maximum_drawdown']:.2%}")
    print(f"Average Daily Turnover: {metrics['average_daily_turnover']:.4f}")
    print(f"Total Traded Notional: {metrics['total_traded_notional']:.4f}")
    print(f"Trading Days: {metrics['number_of_trading_days']}")
    print(
        "Average Long Positions: "
        f"{metrics['average_number_of_long_positions']:.2f}"
    )
    print(
        "Average Short Positions: "
        f"{metrics['average_number_of_short_positions']:.2f}"
    )


def _print_excess(label: str, metrics: dict) -> None:
    print(f"{label}:")
    print(f"Annualized Excess Return: {metrics['annualized_excess_return']:.2%}")
    print(f"Tracking Error: {metrics['tracking_error']:.2%}")
    print(f"Information Ratio: {metrics['information_ratio']:.4f}")
    print(f"Comparison Days: {metrics['number_of_comparison_days']}")


def display_results(result: dict, universe_name: str) -> None:
    metadata = UNIVERSE_METADATA[universe_name]
    period = result.get("period", "validation")
    if period == "holdout":
        print("ONE-TIME CONFIRMATORY HOLDOUT EVALUATION")
        print("HOLDOUT RESULTS MUST NOT BE USED TO RETUNE V1.")
        print("Any changed strategy must be versioned as V2 and must not claim")
        print("this 2026 sample as untouched Holdout. No automatic tuning follows.")
        print()
    print(f"{period.title()} Economic Backtest")
    if period == "holdout":
        print(
            "Realized Holdout range: "
            f"{result['period_start'].date()} through "
            f"{result['evaluation_end'].date()}"
        )
    print(f"Universe: {metadata['version']} ({metadata['number_of_symbols']} symbols)")
    print(f"Frozen Alpha: {FROZEN_ALPHA_V1} (positive direction)")
    print(f"Transaction Cost: {COST_BPS} bps per dollar traded")
    print("Execution: signal after Close_t; hold Open_{t+1} to Open_{t+2}")

    long_only = result["long_only"]
    print("\nLong-Only Top Quintile:")
    _print_performance("Gross", long_only["performance"]["gross"])
    print()
    _print_performance("Net", long_only["performance"]["net"])

    benchmark = long_only["benchmark"]
    print("\nEqual-Weight Benchmark:")
    _print_performance("Gross", benchmark["performance"]["gross"])
    print()
    _print_performance("Net", benchmark["performance"]["net"])
    print()
    _print_excess("Gross Excess", long_only["excess_performance"]["gross"])
    print()
    _print_excess("Net Excess", long_only["excess_performance"]["net"])

    long_short = result["long_short"]
    print("\nLong-Short Top/Bottom Quintile:")
    _print_performance("Gross", long_short["performance"]["gross"])
    print()
    _print_performance("Net", long_short["performance"]["net"])

    invalid_counts = {
        "signal formation invalid count": len(result["invalid_signal_dates"]),
        "long-only invalid count": len(long_only["invalid_dates"]),
        "benchmark invalid count": len(benchmark["invalid_dates"]),
        "long-short invalid count": len(long_short["invalid_dates"]),
    }
    if any(invalid_counts.values()):
        if period == "holdout":
            print("\nWARNING: INVALID HOLDOUT DATES DETECTED")
            print("HOLDOUT ECONOMIC RESULTS MUST NOT BE INTERPRETED.")
        else:
            print("\nWARNING: INVALID BACKTEST DATES DETECTED")
            print("ECONOMIC RESULTS MUST NOT BE INTERPRETED.")
    print("\nInvalid Dates (fail-closed):")
    for name, count in invalid_counts.items():
        print(f"{name}: {count}")


def main(
    argv: Sequence[str] | None = None,
    loader: Callable[[str, str, str], pd.DataFrame] = load_market_data,
) -> dict:
    args = build_parser().parse_args(argv)
    if args.period == "holdout" and not args.allow_holdout:
        raise ValueError("Holdout requires explicit --allow-holdout authorization")
    if args.allow_holdout and args.period != "holdout":
        raise ValueError("--allow-holdout requires --period holdout")

    if args.period == "holdout":
        load_start = HOLDOUT_LOAD_START
        load_end = holdout_load_end()
    else:
        load_start = LOAD_START
        load_end = LOAD_END
    symbol_data = load_adjusted_ohlc_data(
        symbols=UNIVERSES[args.universe],
        start=load_start,
        end=load_end,
        loader=loader,
    )
    result = run_volatility_factor_backtest(
        symbol_data,
        period=args.period,
        allow_holdout=args.allow_holdout,
    )
    display_results(result, universe_name=args.universe)
    return result


if __name__ == "__main__":
    main()
