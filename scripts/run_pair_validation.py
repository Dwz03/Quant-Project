"""Replicate the frozen Research pair cohort on cached Validation data."""

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
    RESEARCH_CANDIDATE_P_THRESHOLD,
    VALIDATION_FDR_LEVEL,
    validate_frozen_pair_candidates,
)
from src.research.universes import UNIVERSES, UNIVERSE_METADATA


UNIVERSE_NAME = "large"
CACHE_START = "2020-01-01"
CACHE_END = "2026-01-01"
RESEARCH_DIAGNOSTICS_PATH = Path("reports/pairs_screen/pair_diagnostics.csv")
OUTPUT_DIR = Path("reports/pairs_validation")
EXPECTED_FROZEN_CANDIDATES = 246


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


def build_validation_summary(result):
    diagnostics = result["diagnostics"]
    prices = result["validation_prices"]
    realized_dates = prices.index[prices.notna().any(axis=1)]
    raw_replicated = int(
        (diagnostics["validation_cointegration_p_value"] < 0.05).sum()
    )
    frozen_count = len(result["frozen_candidates"])
    rows = [
        {"category": "run", "metric": "frozen_research_candidates", "value": frozen_count},
        {"category": "run", "metric": "research_raw_p_threshold", "value": RESEARCH_CANDIDATE_P_THRESHOLD},
        {"category": "run", "metric": "validation_start", "value": realized_dates.min().date()},
        {"category": "run", "metric": "validation_end", "value": realized_dates.max().date()},
        {"category": "run", "metric": "evaluated_candidates", "value": len(diagnostics)},
        {"category": "run", "metric": "skipped_candidates", "value": len(result["skipped"])},
        {"category": "run", "metric": "validation_raw_p_below_05", "value": raw_replicated},
        {"category": "run", "metric": "validation_bh_q_below_05", "value": len(result["advancing_pairs"])},
        {"category": "run", "metric": "raw_replication_fraction", "value": raw_replicated / frozen_count if frozen_count else np.nan},
        {"category": "run", "metric": "rolling_beta_window", "value": result["rolling_window"]},
        {"category": "run", "metric": "validation_fdr_level", "value": result["fdr_level"]},
    ]
    rows.extend(_distribution_rows(
        "validation_aligned_observations",
        diagnostics["validation_aligned_observations"],
    ))
    rows.extend(_distribution_rows(
        "validation_cointegration_p_value",
        diagnostics["validation_cointegration_p_value"],
    ))
    rows.extend(_distribution_rows(
        "validation_cointegration_q_value",
        diagnostics["validation_cointegration_q_value"],
    ))
    rows.extend(_distribution_rows(
        "validation_half_life", diagnostics["validation_half_life"]
    ))
    for column in (
        "validation_mean_beta",
        "validation_median_beta",
        "validation_std_beta",
        "validation_min_beta",
        "validation_max_beta",
        "validation_valid_estimates",
    ):
        rows.extend(_distribution_rows(column, diagnostics[column]))
    return pd.DataFrame(rows)


def main(
    loader: Callable[[str, str, str], pd.DataFrame] = load_local_market_data,
    research_diagnostics_path: Path = RESEARCH_DIAGNOSTICS_PATH,
    output_dir: Path = OUTPUT_DIR,
):
    metadata = UNIVERSE_METADATA[UNIVERSE_NAME]
    symbols = UNIVERSES[UNIVERSE_NAME]
    if metadata["version"] != "large-liquid-us-equities-v1" or len(symbols) != 88:
        raise ValueError("unexpected large-liquid-us-equities-v1 configuration")

    research_diagnostics = pd.read_csv(research_diagnostics_path)
    frozen_count = int(
        (research_diagnostics["cointegration_p_value"] < RESEARCH_CANDIDATE_P_THRESHOLD).sum()
    )
    if frozen_count != EXPECTED_FROZEN_CANDIDATES:
        raise ValueError(
            f"expected {EXPECTED_FROZEN_CANDIDATES} frozen Research candidates, "
            f"found {frozen_count}"
        )

    symbol_data = {
        symbol: loader(symbol, CACHE_START, CACHE_END)
        for symbol in symbols
    }
    result = validate_frozen_pair_candidates(
        symbol_data,
        research_diagnostics,
        rolling_window=PAIR_DIAGNOSTIC_ROLLING_WINDOW,
        min_observations=PAIR_DIAGNOSTIC_ROLLING_WINDOW,
    )
    summary = build_validation_summary(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    result["diagnostics"].to_csv(
        output_dir / "validation_diagnostics.csv", index=False
    )
    summary.to_csv(output_dir / "validation_summary.csv", index=False)
    result["advancing_pairs"].to_csv(
        output_dir / "advancing_pairs.csv", index=False
    )
    result["skipped"].to_csv(
        output_dir / "skipped_candidates.csv", index=False
    )

    print("FROZEN RESEARCH COHORT — VALIDATION REPLICATION")
    print(f"Universe: {metadata['version']}")
    print(f"Research cohort rule: raw p < {RESEARCH_CANDIDATE_P_THRESHOLD}")
    print(f"Validation advancement rule: BH-FDR q < {VALIDATION_FDR_LEVEL}")
    print(summary.loc[summary["category"] == "run"].to_string(index=False))
    if result["advancing_pairs"].empty:
        print(
            "\nNo statistically replicated pair passed the frozen Validation "
            "BH-FDR gate. Pairs V2 should not advance to economic backtesting."
        )
    else:
        print("\nADVANCING PAIRS (not ranked)")
        print(
            result["advancing_pairs"][[
                "symbol_1",
                "symbol_2",
                "validation_cointegration_p_value",
                "validation_cointegration_q_value",
            ]].to_string(index=False)
        )
    print(f"\nOutputs: {output_dir}")

    result["summary"] = summary
    result["universe_metadata"] = metadata
    return result


if __name__ == "__main__":
    main()
