"""Run the frozen momentum screen on Research and Validation market data."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_feature_screen import load_symbol_data
from src.data_loader import load_market_data
from src.research.features import (
    MOMENTUM_FEATURE_COLUMNS,
    build_multi_asset_momentum_panel,
    run_momentum_feature_screen,
)
from src.research.periods import HOLDOUT_PERIOD
from src.research.universes import UNIVERSES, UNIVERSE_METADATA


START = "2020-01-01"
END = "2026-01-01"
MIN_SYMBOLS = 5
OUTPUT_DIR = Path("reports/momentum_feature_screen")


def main() -> dict[str, object]:
    symbols = UNIVERSES["large"]
    metadata = UNIVERSE_METADATA["large"]
    if metadata["version"] != "large-liquid-us-equities-v1":
        raise ValueError("unexpected large-universe version")

    symbol_data = load_symbol_data(
        symbols=symbols,
        start=START,
        end=END,
        loader=load_market_data,
    )
    if tuple(symbol_data) != tuple(symbols):
        raise ValueError("loaded symbols do not match the fixed large universe")
    for symbol, data in symbol_data.items():
        dates = pd.DatetimeIndex(data.index)
        if (dates >= HOLDOUT_PERIOD.start).any():
            raise ValueError(f"holdout observation returned for {symbol}")

    panel = build_multi_asset_momentum_panel(symbol_data)
    if not panel.index.is_unique:
        raise ValueError("momentum panel index must be unique")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("momentum panel must be chronological")
    if panel.index.get_level_values("date").max() >= HOLDOUT_PERIOD.start:
        raise ValueError("momentum panel contains holdout observations")
    if not set(MOMENTUM_FEATURE_COLUMNS).issubset(panel.columns):
        raise ValueError("momentum panel is missing a frozen horizon")

    result = run_momentum_feature_screen(panel, min_symbols=MIN_SYMBOLS)
    period_results = result["period_results"]
    research_panel = period_results["research"]["panel"]
    validation_panel = period_results["validation"]["panel"]
    research_dates = research_panel.index.get_level_values("date")
    validation_dates = validation_panel.index.get_level_values("date")
    if research_dates.max() >= validation_dates.min():
        raise ValueError("Research and Validation periods overlap")

    sample_rows = []
    for period_name, period_panel in (
        ("research", research_panel),
        ("validation", validation_panel),
    ):
        dates = period_panel.index.get_level_values("date")
        sample_rows.append({
            "period": period_name,
            "start_date": dates.min(),
            "end_date": dates.max(),
            "unique_symbols": period_panel.index.get_level_values(
                "symbol"
            ).nunique(),
            "panel_rows": len(period_panel),
        })
    sample_information = pd.DataFrame(sample_rows)

    coverage_rows = []
    combined_panel = pd.concat([research_panel, validation_panel]).sort_index()
    for feature in MOMENTUM_FEATURE_COLUMNS:
        valid = combined_panel[[feature, "target"]].notna().all(axis=1)
        valid_dates = combined_panel.index.get_level_values("date")[valid]
        row = {
            "feature": feature,
            "first_valid_date": valid_dates.min(),
        }
        for period_name, period_panel in (
            ("research", research_panel),
            ("validation", validation_panel),
        ):
            row[f"{period_name}_valid_pairs"] = int(
                period_panel[[feature, "target"]].notna().all(axis=1).sum()
            )
        coverage_rows.append(row)
    coverage = pd.DataFrame(coverage_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result["comparison"].to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    result["stability"].to_csv(OUTPUT_DIR / "stability.csv", index=False)
    sample_information.to_csv(
        OUTPUT_DIR / "sample_information.csv", index=False
    )
    coverage.to_csv(OUTPUT_DIR / "feature_coverage.csv", index=False)

    print(f"Universe: {metadata['version']}")
    print(f"Configured symbols: {len(symbols)}")
    print(f"Source range: {START} to {END} (exclusive)")
    print("\nSAMPLE_INFORMATION")
    print(sample_information.to_string(index=False))
    print("\nFEATURE_COVERAGE")
    print(coverage.to_string(index=False))
    print("\nRESEARCH_VS_VALIDATION")
    print(result["comparison"].to_string(index=False))
    print("\nSTABILITY")
    print(result["stability"].to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")

    return {
        **result,
        "panel": panel,
        "sample_information": sample_information,
        "coverage": coverage,
        "universe_metadata": metadata,
    }


if __name__ == "__main__":
    main()
