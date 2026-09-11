"""Run descriptive V1 price-feature screening across a symbol universe."""

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
from src.research.features import (
    build_multi_asset_feature_panel,
    calculate_cross_sectional_ic,
    calculate_feature_redundancy,
    run_multi_asset_feature_screen,
    run_cross_sectional_ic_by_period,
    summarize_cross_sectional_ic,
    summarize_feature_stability,
    summarize_feature_screen,
)
from src.research.periods import get_research_period
from src.research.universes import UNIVERSES, UNIVERSE_METADATA


DEFAULT_SYMBOLS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "AMZN",
    "META",
    "JPM",
    "XOM",
    "JNJ",
    "KO",
)
DEFAULT_START = "2021-01-01"
DEFAULT_END = "2026-01-01"


def minimum_symbols(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed_value


def load_symbol_data(
    symbols: Sequence[str],
    start: str,
    end: str,
    loader: Callable[[str, str, str], pd.DataFrame] = load_market_data,
) -> dict[str, pd.DataFrame]:
    """Load and clean adjusted daily closes using project conventions."""
    symbols_used = list(symbols)
    if not symbols_used:
        raise ValueError("symbols must contain at least one symbol")
    if len(set(symbols_used)) != len(symbols_used):
        raise ValueError("symbols must be unique")

    symbol_data = {}
    for symbol in symbols_used:
        raw_data = loader(symbol, start, end)
        try:
            cleaned_data = clean_market_data(raw_data)
        except KeyError as exc:
            raise ValueError(
                f"market data for {symbol} must contain Close"
            ) from exc

        close_data = cleaned_data[["Close"]].astype(float)
        close_values = close_data["Close"].to_numpy(dtype=float)
        if close_data.empty:
            raise ValueError(f"market data for {symbol} has no close observations")
        if not np.isfinite(close_values).all() or (close_values <= 0).any():
            raise ValueError("adjusted-close prices must be finite and positive")

        symbol_data[symbol] = close_data

    return symbol_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen fixed price features across a symbol universe.",
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--universe", choices=tuple(UNIVERSES))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--cross-sectional",
        action="store_true",
        help="run same-date cross-sectional IC screening",
    )
    parser.add_argument(
        "--method",
        choices=("pearson", "spearman"),
        default="spearman",
    )
    parser.add_argument("--min-symbols", type=minimum_symbols, default=5)
    parser.add_argument(
        "--period",
        action="append",
        choices=("research", "validation", "holdout"),
        help="run a fixed named period; repeat to compare periods",
    )
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="explicitly authorize display of holdout diagnostics",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    loader: Callable[[str, str, str], pd.DataFrame] = load_market_data,
) -> dict:
    args = build_parser().parse_args(argv)
    if pd.Timestamp(args.start) >= pd.Timestamp(args.end):
        raise ValueError("start must be earlier than end")
    if args.period and not args.cross_sectional:
        raise ValueError("--period requires --cross-sectional")
    for period_name in args.period or []:
        get_research_period(
            period_name,
            allow_holdout=args.allow_holdout,
        )

    symbols = UNIVERSES[args.universe] if args.universe else args.symbols

    symbol_data = load_symbol_data(
        symbols=symbols,
        start=args.start,
        end=args.end,
        loader=loader,
    )

    if args.cross_sectional:
        panel = build_multi_asset_feature_panel(symbol_data)

        if args.period:
            period_results = run_cross_sectional_ic_by_period(
                panel,
                period_names=args.period,
                min_symbols=args.min_symbols,
                method=args.method,
                allow_holdout=args.allow_holdout,
            )
            for period_name, period_result in period_results.items():
                print(
                    f"{period_name.title()} Period Cross-Sectional IC Summary "
                    f"({args.method}):"
                )
                print(period_result["summary"].to_string(index=False))

            stability = None
            if {"research", "validation"}.issubset(period_results):
                stability = summarize_feature_stability(
                    period_results["research"]["summary"],
                    period_results["validation"]["summary"],
                )
                print("\nResearch vs Validation Feature Stability:")
                print(stability.to_string(index=False))

            research_redundancy = None
            if "research" in period_results:
                research_redundancy = calculate_feature_redundancy(
                    period_results["research"]["panel"],
                    min_symbols=args.min_symbols,
                    method=args.method,
                )
                print("\nResearch-Period Feature Redundancy:")
                print(research_redundancy.to_string())

            if args.universe:
                metadata = UNIVERSE_METADATA[args.universe]
                print(
                    f"\nUniverse: {metadata['version']} "
                    f"({len(symbols)} symbols; not point-in-time)"
                )

            return {
                "panel": panel,
                "period_results": period_results,
                "stability": stability,
                "research_redundancy": research_redundancy,
            }

        ic_results = calculate_cross_sectional_ic(
            panel,
            min_symbols=args.min_symbols,
            method=args.method,
        )
        summary = summarize_cross_sectional_ic(ic_results)

        print(
            f"Exploratory Cross-Sectional IC Summary ({args.method}):"
        )
        print(f"Research period: {args.start} to {args.end}")
        print(f"Minimum symbols per date: {args.min_symbols}")
        print(summary.to_string(index=False))

        return {
            "panel": panel,
            "ic_results": ic_results,
            "summary": summary,
        }

    screen_results = run_multi_asset_feature_screen(symbol_data)
    summary = summarize_feature_screen(screen_results)

    print("Cross-Asset Feature Summary:")
    print(summary.to_string(index=False))

    return {
        "screen_results": screen_results,
        "summary": summary,
    }


if __name__ == "__main__":
    main()
