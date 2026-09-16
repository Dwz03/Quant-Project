"""Economic backtests for the five frozen cross-sectional momentum features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import MOMENTUM_FEATURE_COLUMNS, MOMENTUM_HORIZONS
from .periods import get_research_period
from .volatility_factor_backtest import (
    COST_BPS,
    QUINTILE_FRACTION,
    _build_long_only_comparison,
    _build_schedules,
    _evaluate_schedules,
    _validate_symbol_data,
)


FROZEN_MOMENTUM_FEATURES = tuple(MOMENTUM_FEATURE_COLUMNS)
ALLOWED_PERIODS = ("research", "validation")
FROZEN_MOMENTUM_V1 = "momentum_252"
FROZEN_MOMENTUM_V1_HORIZON = 252
FROZEN_MOMENTUM_V1_PRIMARY_PORTFOLIO = "long_only_top_quintile"


def build_momentum_scores(
    symbol_data: dict[str, pd.DataFrame],
    feature: str,
    allow_holdout: bool = False,
) -> pd.Series:
    """Build one frozen momentum score without using future prices."""
    if feature not in FROZEN_MOMENTUM_FEATURES:
        valid = ", ".join(FROZEN_MOMENTUM_FEATURES)
        raise ValueError(f"feature must be one of: {valid}")
    horizon = MOMENTUM_HORIZONS[FROZEN_MOMENTUM_FEATURES.index(feature)]
    validated = _validate_symbol_data(
        symbol_data,
        allow_holdout=allow_holdout,
    )

    scores = {
        symbol: data["Close"] / data["Close"].shift(horizon) - 1.0
        for symbol, data in validated.items()
    }
    panel = (
        pd.concat(scores, names=["symbol", "signal_date"])
        .reorder_levels(["signal_date", "symbol"])
        .sort_index()
    )
    panel.name = "score"
    return panel


def run_momentum_factor_backtest(
    symbol_data: dict[str, pd.DataFrame],
    feature: str,
    period: str,
    allow_holdout: bool = False,
) -> dict:
    """Run one frozen momentum feature using the V1 economic mechanics."""
    supported_periods = (*ALLOWED_PERIODS, "holdout")
    if period not in supported_periods:
        valid = ", ".join(supported_periods)
        raise ValueError(f"period must be one of: {valid}")
    if period == "holdout" and not allow_holdout:
        raise ValueError("holdout requires explicit allow_holdout=True")
    if allow_holdout and period != "holdout":
        raise ValueError("allow_holdout may only be used with period='holdout'")
    period_specification = get_research_period(
        period,
        allow_holdout=allow_holdout,
    )
    validated = _validate_symbol_data(
        symbol_data,
        allow_holdout=allow_holdout,
    )
    scores = build_momentum_scores(
        validated,
        feature,
        allow_holdout=allow_holdout,
    )
    calendar = pd.DatetimeIndex(sorted({
        date
        for data in validated.values()
        for date in data.index
    }))

    long_only_schedules, invalid_signal_dates = _build_schedules(
        scores,
        calendar,
        "long_only_top_quintile",
        period_specification.start,
        period_specification.end,
    )
    long_short_schedules, _ = _build_schedules(
        scores,
        calendar,
        "long_short_top_bottom_quintile",
        period_specification.start,
        period_specification.end,
    )
    benchmark_schedules, _ = _build_schedules(
        scores,
        calendar,
        "equal_weight_benchmark",
        period_specification.start,
        period_specification.end,
    )
    if not long_only_schedules:
        raise ValueError(
            f"no {period.title()} signal dates have sufficient observations"
        )

    long_only = _evaluate_schedules(long_only_schedules, validated)
    long_short = _evaluate_schedules(long_short_schedules, validated)
    benchmark = _evaluate_schedules(benchmark_schedules, validated)
    comparison_daily, excess_performance = _build_long_only_comparison(
        long_only,
        benchmark,
    )
    long_only["daily"]["benchmark_return"] = comparison_daily[
        "benchmark_gross_return"
    ]
    long_only["daily"]["net_benchmark_return"] = comparison_daily[
        "benchmark_net_return"
    ]
    long_only["daily"]["excess_return"] = comparison_daily[
        "gross_excess_return"
    ]
    long_only["daily"]["net_excess_return"] = comparison_daily[
        "net_excess_return"
    ]
    long_only["daily"]["comparison_is_valid"] = comparison_daily["is_valid"]
    long_only["benchmark"] = benchmark
    long_only["comparison_daily"] = comparison_daily
    long_only["excess_performance"] = excess_performance

    period_dates = scores.index.get_level_values("signal_date")
    period_mask = period_dates >= period_specification.start
    if period_specification.end is not None:
        period_mask &= period_dates <= period_specification.end

    return {
        "feature": feature,
        "horizon": MOMENTUM_HORIZONS[FROZEN_MOMENTUM_FEATURES.index(feature)],
        "direction": "positive",
        "quintile_fraction": QUINTILE_FRACTION,
        "cost_bps": COST_BPS,
        "period": period,
        "period_start": period_specification.start,
        "period_end": period_specification.end,
        "evaluation_end": long_only["daily"]["next_execution_date"].max(),
        "scores": scores.loc[period_mask],
        "invalid_signal_dates": invalid_signal_dates,
        "long_only": long_only,
        "long_short": long_short,
    }


def run_momentum_252_v1_backtest(
    symbol_data: dict[str, pd.DataFrame],
    period: str = "validation",
    allow_holdout: bool = False,
) -> dict:
    """Run the immutable Momentum 252 V1 specification."""
    result = run_momentum_factor_backtest(
        symbol_data,
        feature=FROZEN_MOMENTUM_V1,
        period=period,
        allow_holdout=allow_holdout,
    )
    result["frozen_alpha"] = FROZEN_MOMENTUM_V1
    result["primary_portfolio"] = FROZEN_MOMENTUM_V1_PRIMARY_PORTFOLIO
    return result


def run_all_momentum_factor_backtests(
    symbol_data: dict[str, pd.DataFrame],
) -> dict[str, dict[str, dict]]:
    """Run every frozen feature in fixed order for both allowed periods."""
    return {
        period: {
            feature: run_momentum_factor_backtest(
                symbol_data,
                feature=feature,
                period=period,
            )
            for feature in FROZEN_MOMENTUM_FEATURES
        }
        for period in ALLOWED_PERIODS
    }
