"""Guarded one-time Holdout evaluation for frozen Momentum 252 V1."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_volatility_factor_backtest import (
    _print_excess,
    _print_performance,
    holdout_load_end,
    load_adjusted_ohlc_data,
)
from src.data_loader import load_market_data
from src.research.momentum_factor_backtest import (
    FROZEN_MOMENTUM_V1,
    FROZEN_MOMENTUM_V1_HORIZON,
    FROZEN_MOMENTUM_V1_PRIMARY_PORTFOLIO,
    run_momentum_252_v1_backtest,
)
from src.research.universes import UNIVERSES, UNIVERSE_METADATA
from src.research.volatility_factor_backtest import COST_BPS


LOAD_START = "2020-01-01"
LOAD_END = "2026-01-01"
HOLDOUT_LOAD_START = "2024-12-01"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen Momentum 252 V1 with an explicit Holdout guard.",
    )
    parser.add_argument(
        "--period",
        choices=("research", "validation", "holdout"),
        default="validation",
    )
    parser.add_argument("--allow-holdout", action="store_true")
    return parser


def print_holdout_warning() -> None:
    print("ONE-TIME CONFIRMATORY HOLDOUT EVALUATION")
    print("HOLDOUT RESULTS MUST NOT BE USED TO RETUNE MOMENTUM 252 V1.")
    print("Any changed strategy must be versioned as V2 and must not claim")
    print("this 2026 sample as untouched Holdout.")
    print()


def primary_holdout_gate(result: dict) -> dict[str, bool | float | int]:
    long_only = result["long_only"]
    benchmark = long_only["benchmark"]
    long_short = result["long_short"]
    invalid_count = sum([
        len(result["invalid_signal_dates"]),
        len(long_only["invalid_dates"]),
        len(benchmark["invalid_dates"]),
        len(long_short["invalid_dates"]),
    ])
    net_excess = long_only["excess_performance"]["net"]
    annualized_excess_return = net_excess["annualized_excess_return"]
    information_ratio = net_excess["information_ratio"]
    return {
        "invalid_date_count": invalid_count,
        "invalid_dates_clear": invalid_count == 0,
        "net_annualized_excess_return": annualized_excess_return,
        "net_excess_return_positive": annualized_excess_return > 0,
        "net_information_ratio": information_ratio,
        "net_information_ratio_positive": information_ratio > 0,
        "passed": (
            invalid_count == 0
            and annualized_excess_return > 0
            and information_ratio > 0
        ),
    }


def display_results(result: dict) -> None:
    metadata = UNIVERSE_METADATA["large"]
    print(f"{result['period'].title()} Momentum 252 V1 Economic Backtest")
    print(f"Universe: {metadata['version']} ({metadata['number_of_symbols']} symbols)")
    print(f"Frozen Feature: {FROZEN_MOMENTUM_V1}")
    print(f"Frozen Horizon: {FROZEN_MOMENTUM_V1_HORIZON}")
    print(f"Primary Portfolio: {FROZEN_MOMENTUM_V1_PRIMARY_PORTFOLIO}")
    print(f"Transaction Cost: {COST_BPS} bps per dollar traded")
    print("Execution: signal after Close_t; hold Open_{t+1} to Open_{t+2}")

    long_only = result["long_only"]
    print("\nLong-Only Top Quintile:")
    _print_performance("Gross", long_only["performance"]["gross"])
    print()
    _print_performance("Net", long_only["performance"]["net"])

    benchmark = long_only["benchmark"]
    print("\nEqual-Weight Eligible-Universe Benchmark:")
    _print_performance("Gross", benchmark["performance"]["gross"])
    print()
    _print_performance("Net", benchmark["performance"]["net"])
    print()
    _print_excess("Gross Excess", long_only["excess_performance"]["gross"])
    print()
    _print_excess("Net Excess", long_only["excess_performance"]["net"])

    long_short = result["long_short"]
    print("\nLong-Short Secondary Diagnostic:")
    _print_performance("Gross", long_short["performance"]["gross"])
    print()
    _print_performance("Net", long_short["performance"]["net"])

    invalid_counts = {
        "signal formation invalid count": len(result["invalid_signal_dates"]),
        "long-only invalid count": len(long_only["invalid_dates"]),
        "benchmark invalid count": len(benchmark["invalid_dates"]),
        "long-short invalid count": len(long_short["invalid_dates"]),
    }
    print("\nInvalid Dates (fail-closed):")
    for label, count in invalid_counts.items():
        print(f"{label}: {count}")

    if result["period"] == "holdout":
        gate = primary_holdout_gate(result)
        print("\nPRIMARY HOLDOUT GATE")
        print(f"All invalid-date counts zero: {gate['invalid_dates_clear']}")
        print(
            "Long-only net annualized excess return > 0: "
            f"{gate['net_excess_return_positive']} "
            f"({gate['net_annualized_excess_return']:.2%})"
        )
        print(
            "Long-only net Information Ratio > 0: "
            f"{gate['net_information_ratio_positive']} "
            f"({gate['net_information_ratio']:.4f})"
        )
        print(f"Gate Result: {'PASS' if gate['passed'] else 'FAIL'}")


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
        print_holdout_warning()
        load_start = HOLDOUT_LOAD_START
        load_end = holdout_load_end()
    else:
        load_start = LOAD_START
        load_end = LOAD_END

    symbol_data = load_adjusted_ohlc_data(
        symbols=UNIVERSES["large"],
        start=load_start,
        end=load_end,
        loader=loader,
    )
    result = run_momentum_252_v1_backtest(
        symbol_data,
        period=args.period,
        allow_holdout=args.allow_holdout,
    )
    display_results(result)
    return result


if __name__ == "__main__":
    main()
