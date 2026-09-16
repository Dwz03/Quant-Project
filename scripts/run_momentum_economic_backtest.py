"""Run fixed-rule momentum economic backtests from the approved local cache."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_volatility_factor_backtest import load_adjusted_ohlc_data
from src.data_loader import load_local_market_data
from src.research.momentum_factor_backtest import (
    ALLOWED_PERIODS,
    FROZEN_MOMENTUM_FEATURES,
    run_all_momentum_factor_backtests,
)
from src.research.periods import HOLDOUT_PERIOD
from src.research.universes import UNIVERSES, UNIVERSE_METADATA


START = "2020-01-01"
END = "2026-01-01"
OUTPUT_DIR = Path("reports/momentum_economic_backtest")


def _add_metrics(row: dict, prefix: str, metrics: dict) -> None:
    for name, value in metrics.items():
        row[f"{prefix}_{name}"] = value


def _daily_output(result: dict) -> pd.DataFrame:
    long_only = result["long_only"]["daily"]
    benchmark = result["long_only"]["benchmark"]["daily"]
    long_short = result["long_short"]["daily"]
    return pd.DataFrame({
        "signal_date": long_only["signal_date"],
        "execution_date": long_only["execution_date"],
        "next_execution_date": long_only["next_execution_date"],
        "long_only_gross_return": long_only["gross_return"],
        "long_only_net_return": long_only["net_return"],
        "long_only_turnover": long_only["turnover"],
        "long_only_is_valid": long_only["is_valid"],
        "benchmark_gross_return": benchmark["gross_return"],
        "benchmark_net_return": benchmark["net_return"],
        "benchmark_turnover": benchmark["turnover"],
        "benchmark_is_valid": benchmark["is_valid"],
        "long_short_gross_return": long_short["gross_return"],
        "long_short_net_return": long_short["net_return"],
        "long_short_turnover": long_short["turnover"],
        "long_short_is_valid": long_short["is_valid"],
    }).reset_index(drop=True)


def main() -> dict[str, object]:
    metadata = UNIVERSE_METADATA["large"]
    if metadata["version"] != "large-liquid-us-equities-v1":
        raise ValueError("unexpected large-universe version")
    symbols = UNIVERSES["large"]
    symbol_data = load_adjusted_ohlc_data(
        symbols=symbols,
        start=START,
        end=END,
        loader=load_local_market_data,
    )
    for symbol, data in symbol_data.items():
        if (data.index >= HOLDOUT_PERIOD.start).any():
            raise ValueError(f"holdout observation found in cache for {symbol}")

    results = run_all_momentum_factor_backtests(symbol_data)
    metric_rows = []
    invalid_rows = []
    sample_rows = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for period in ALLOWED_PERIODS:
        for feature in FROZEN_MOMENTUM_FEATURES:
            result = results[period][feature]
            long_only = result["long_only"]
            benchmark = long_only["benchmark"]
            long_short = result["long_short"]
            daily = long_only["daily"]
            if daily["next_execution_date"].max() >= HOLDOUT_PERIOD.start:
                raise ValueError("execution entered the Holdout period")

            row = {"period": period, "feature": feature}
            for name, portfolio in (
                ("long_only", long_only),
                ("benchmark", benchmark),
                ("long_short", long_short),
            ):
                for return_type in ("gross", "net"):
                    _add_metrics(
                        row,
                        f"{name}_{return_type}",
                        portfolio["performance"][return_type],
                    )
            for return_type in ("gross", "net"):
                _add_metrics(
                    row,
                    f"long_only_{return_type}_excess",
                    long_only["excess_performance"][return_type],
                )
            metric_rows.append(row)

            counts = {
                "invalid_signal_dates": len(result["invalid_signal_dates"]),
                "long_only_invalid_dates": len(long_only["invalid_dates"]),
                "benchmark_invalid_dates": len(benchmark["invalid_dates"]),
                "long_short_invalid_dates": len(long_short["invalid_dates"]),
            }
            invalid_rows.append({
                "period": period,
                "feature": feature,
                **counts,
                "results_interpretable": not any(counts.values()),
            })
            sample_rows.append({
                "period": period,
                "feature": feature,
                "first_signal_date": daily["signal_date"].min(),
                "last_signal_date": daily["signal_date"].max(),
                "first_execution_date": daily["execution_date"].min(),
                "last_execution_date": daily["next_execution_date"].max(),
                "scheduled_days": len(daily),
            })

            detail_dir = OUTPUT_DIR / period / feature
            detail_dir.mkdir(parents=True, exist_ok=True)
            _daily_output(result).to_csv(detail_dir / "daily.csv", index=False)
            result["invalid_signal_dates"].to_csv(
                detail_dir / "invalid_signal_dates.csv", index=False
            )
            long_only["invalid_dates"].to_csv(
                detail_dir / "long_only_invalid_dates.csv", index=False
            )
            benchmark["invalid_dates"].to_csv(
                detail_dir / "benchmark_invalid_dates.csv", index=False
            )
            long_short["invalid_dates"].to_csv(
                detail_dir / "long_short_invalid_dates.csv", index=False
            )

    detailed_metrics = pd.DataFrame(metric_rows)
    invalid_diagnostics = pd.DataFrame(invalid_rows)
    sample_information = pd.DataFrame(sample_rows)
    validation = detailed_metrics.loc[
        detailed_metrics["period"] == "validation"
    ].set_index("feature")
    validation_comparison = pd.DataFrame({
        "feature": FROZEN_MOMENTUM_FEATURES,
        "long_only_net_ann_return": validation.loc[
            list(FROZEN_MOMENTUM_FEATURES),
            "long_only_net_annualized_return",
        ].to_numpy(),
        "long_only_net_sharpe": validation.loc[
            list(FROZEN_MOMENTUM_FEATURES),
            "long_only_net_sharpe_ratio",
        ].to_numpy(),
        "long_only_net_excess_return": validation.loc[
            list(FROZEN_MOMENTUM_FEATURES),
            "long_only_net_excess_annualized_excess_return",
        ].to_numpy(),
        "long_only_net_information_ratio": validation.loc[
            list(FROZEN_MOMENTUM_FEATURES),
            "long_only_net_excess_information_ratio",
        ].to_numpy(),
        "long_short_net_ann_return": validation.loc[
            list(FROZEN_MOMENTUM_FEATURES),
            "long_short_net_annualized_return",
        ].to_numpy(),
        "long_short_net_sharpe": validation.loc[
            list(FROZEN_MOMENTUM_FEATURES),
            "long_short_net_sharpe_ratio",
        ].to_numpy(),
        "long_only_turnover": validation.loc[
            list(FROZEN_MOMENTUM_FEATURES),
            "long_only_net_average_daily_turnover",
        ].to_numpy(),
        "long_short_turnover": validation.loc[
            list(FROZEN_MOMENTUM_FEATURES),
            "long_short_net_average_daily_turnover",
        ].to_numpy(),
    })

    detailed_metrics.to_csv(OUTPUT_DIR / "detailed_metrics.csv", index=False)
    invalid_diagnostics.to_csv(
        OUTPUT_DIR / "invalid_diagnostics.csv", index=False
    )
    sample_information.to_csv(
        OUTPUT_DIR / "sample_information.csv", index=False
    )
    validation_comparison.to_csv(
        OUTPUT_DIR / "validation_comparison.csv", index=False
    )

    print(f"Universe: {metadata['version']} ({len(symbols)} symbols)")
    print(f"Cache range: {START} to {END} (exclusive)")
    print("\nSAMPLE_INFORMATION")
    print(sample_information.to_string(index=False))
    print("\nVALIDATION_COMPARISON")
    print(validation_comparison.to_string(index=False))
    print("\nINVALID_DIAGNOSTICS")
    print(invalid_diagnostics.to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")

    return {
        "results": results,
        "detailed_metrics": detailed_metrics,
        "invalid_diagnostics": invalid_diagnostics,
        "sample_information": sample_information,
        "validation_comparison": validation_comparison,
    }


if __name__ == "__main__":
    main()
