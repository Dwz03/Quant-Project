"""Screen all large-universe pairs using cached Research-period data only."""

from collections.abc import Callable
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import load_local_market_data
from src.research.pairs import (
    PAIR_DIAGNOSTIC_ROLLING_WINDOW,
    screen_research_pairs,
)
from src.research.universes import UNIVERSES, UNIVERSE_METADATA


UNIVERSE_NAME = "large"
CACHE_START = "2020-01-01"
CACHE_END = "2026-01-01"
FDR_LEVEL = 0.05
OUTPUT_DIR = Path("reports/pairs_screen")


def _distribution_rows(category, values):
    finite = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    statistics = {
        "count": len(finite),
        "min": finite.min() if len(finite) else np.nan,
        "q25": finite.quantile(0.25) if len(finite) else np.nan,
        "median": finite.median() if len(finite) else np.nan,
        "q75": finite.quantile(0.75) if len(finite) else np.nan,
        "max": finite.max() if len(finite) else np.nan,
    }
    return [
        {"category": category, "metric": metric, "value": value}
        for metric, value in statistics.items()
    ]


def build_screening_summary(result, symbols_loaded):
    diagnostics = result["diagnostics"]
    skipped = result["skipped"]
    prices = result["research_prices"]
    realized_dates = prices.index[prices.notna().any(axis=1)]

    rows = [
        {"category": "run", "metric": "universe", "value": "large-liquid-us-equities-v1"},
        {"category": "run", "metric": "research_start", "value": realized_dates.min().date()},
        {"category": "run", "metric": "research_end", "value": realized_dates.max().date()},
        {"category": "run", "metric": "symbols_loaded", "value": symbols_loaded},
        {"category": "run", "metric": "total_possible_pairs", "value": result["total_pairs"]},
        {"category": "run", "metric": "evaluated_pairs", "value": len(diagnostics)},
        {"category": "run", "metric": "skipped_pairs", "value": len(skipped)},
        {"category": "run", "metric": "rolling_beta_window", "value": result["rolling_window"]},
        {"category": "run", "metric": "fdr_level", "value": result["fdr_level"]},
        {
            "category": "run",
            "metric": "raw_cointegration_p_below_05",
            "value": int((diagnostics["cointegration_p_value"] < 0.05).sum()),
        },
        {
            "category": "run",
            "metric": "bh_q_below_05",
            "value": int((diagnostics["cointegration_q_value"] < 0.05).sum()),
        },
        {
            "category": "run",
            "metric": "finite_positive_half_life",
            "value": int(
                (np.isfinite(diagnostics["half_life"]) & (diagnostics["half_life"] > 0)).sum()
            ),
        },
    ]
    rows.extend(_distribution_rows("aligned_observations", diagnostics["aligned_observations"]))
    rows.extend(_distribution_rows("cointegration_p_value", diagnostics["cointegration_p_value"]))
    rows.extend(_distribution_rows("cointegration_q_value", diagnostics["cointegration_q_value"]))
    rows.extend(_distribution_rows("half_life", diagnostics["half_life"]))
    for column in (
        "mean_beta",
        "median_beta",
        "std_beta",
        "min_beta",
        "max_beta",
        "valid_estimates",
    ):
        rows.extend(_distribution_rows(f"rolling_{column}", diagnostics[column]))
    return pd.DataFrame(rows)


def main(
    loader: Callable[[str, str, str], pd.DataFrame] = load_local_market_data,
    output_dir: Path = OUTPUT_DIR,
):
    symbols = UNIVERSES[UNIVERSE_NAME]
    metadata = UNIVERSE_METADATA[UNIVERSE_NAME]
    if metadata["version"] != "large-liquid-us-equities-v1" or len(symbols) != 88:
        raise ValueError("unexpected large-liquid-us-equities-v1 configuration")

    symbol_data = {
        symbol: loader(symbol, CACHE_START, CACHE_END)
        for symbol in symbols
    }
    result = screen_research_pairs(
        symbol_data,
        rolling_window=PAIR_DIAGNOSTIC_ROLLING_WINDOW,
        min_observations=PAIR_DIAGNOSTIC_ROLLING_WINDOW,
        fdr_level=FDR_LEVEL,
    )
    summary = build_screening_summary(result, symbols_loaded=len(symbol_data))

    output_dir.mkdir(parents=True, exist_ok=True)
    result["diagnostics"].to_csv(output_dir / "pair_diagnostics.csv", index=False)
    summary.to_csv(output_dir / "screening_summary.csv", index=False)
    result["skipped"].to_csv(output_dir / "skipped_pairs.csv", index=False)

    print("RESEARCH PAIR SCREENING")
    print(f"Universe: {metadata['version']}")
    print(f"Orientation: lexicographic symbol_1 (Y), symbol_2 (X)")
    print(f"Rolling-beta window: {PAIR_DIAGNOSTIC_ROLLING_WINDOW}")
    print(summary.loc[summary["category"] == "run"].to_string(index=False))
    print("\nDIAGNOSTIC TABLE SAMPLE (deterministic pair order)")
    print(result["diagnostics"].head(10).to_string(index=False))
    print(f"\nOutputs: {output_dir}")

    result["summary"] = summary
    result["universe_metadata"] = metadata
    return result


if __name__ == "__main__":
    main()
