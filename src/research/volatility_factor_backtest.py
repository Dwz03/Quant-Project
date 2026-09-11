"""Economic Validation backtest for the frozen cross-sectional volatility alpha.

The predictive target used elsewhere remains unchanged. This module forms a
signal after ``Close_t`` and evaluates it only over ``Open_{t+1}`` to
``Open_{t+2}`` during the Validation period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .periods import HOLDOUT_PERIOD, VALIDATION_PERIOD, get_research_period


FROZEN_ALPHA_V1 = "volatility_20"
FEATURE_WINDOW = 20
QUINTILE_FRACTION = 0.20
COST_BPS = 10
COST_RATE = COST_BPS / 10_000
TRADING_DAYS_PER_YEAR = 252
MINIMUM_ELIGIBLE_SYMBOLS = 5

PORTFOLIO_VARIANTS = (
    "long_only_top_quintile",
    "long_short_top_bottom_quintile",
)


def _validate_symbol_data(
    symbol_data: dict[str, pd.DataFrame],
    allow_holdout: bool = False,
) -> dict[str, pd.DataFrame]:
    if not symbol_data:
        raise ValueError("symbol_data must contain at least one symbol")

    validated = {}
    for symbol, data in symbol_data.items():
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(f"market data index for {symbol} must be a DatetimeIndex")
        if not data.index.is_unique:
            raise ValueError(f"market data index for {symbol} must be unique")
        if not data.index.is_monotonic_increasing:
            raise ValueError(f"market data index for {symbol} must be chronological")

        missing_columns = {"Open", "Close"}.difference(data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"market data for {symbol} is missing: {missing}")
        if data.empty:
            raise ValueError(f"market data for {symbol} cannot be empty")
        if not allow_holdout and (data.index >= HOLDOUT_PERIOD.start).any():
            raise ValueError("dates on or after 2026-01-01 are not allowed")

        used = data[["Open", "Close"]].astype(float).copy()
        close = used["Close"].to_numpy(dtype=float)
        if not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError(f"Close prices for {symbol} must be finite and positive")
        available_open = used["Open"].dropna().to_numpy(dtype=float)
        if len(available_open) == 0:
            raise ValueError(f"market data for {symbol} has no valid Open prices")
        if (
            not np.isfinite(available_open).all()
            or (available_open <= 0).any()
        ):
            raise ValueError(
                f"available Open prices for {symbol} must be finite and positive"
            )
        validated[symbol] = used

    return validated


def build_volatility_20_scores(
    symbol_data: dict[str, pd.DataFrame],
    allow_holdout: bool = False,
) -> pd.Series:
    """Calculate the frozen feature without constructing a future target."""
    validated = _validate_symbol_data(
        symbol_data,
        allow_holdout=allow_holdout,
    )
    scores = {}
    for symbol, data in validated.items():
        returns = data["Close"].pct_change(fill_method=None)
        scores[symbol] = returns.rolling(FEATURE_WINDOW).std()

    panel = (
        pd.concat(scores, names=["symbol", "signal_date"])
        .reorder_levels(["signal_date", "symbol"])
        .sort_index()
    )
    panel.name = "score"
    return panel


def form_portfolio_weights(
    same_date_scores: pd.Series,
    variant: str,
) -> pd.DataFrame:
    """Rank one contemporaneous cross-section and form a frozen portfolio."""
    if variant not in PORTFOLIO_VARIANTS and variant != "equal_weight_benchmark":
        valid = ", ".join((*PORTFOLIO_VARIANTS, "equal_weight_benchmark"))
        raise ValueError(f"variant must be one of: {valid}")

    scores = same_date_scores.astype(float)
    scores = scores.loc[scores.notna() & np.isfinite(scores)]
    scores.index = scores.index.astype(str)
    if not scores.index.is_unique:
        raise ValueError("same_date_scores symbol index must be unique")
    if len(scores) < MINIMUM_ELIGIBLE_SYMBOLS:
        raise ValueError(
            f"at least {MINIMUM_ELIGIBLE_SYMBOLS} eligible symbols are required"
        )

    ranked = (
        scores.rename("score")
        .reset_index()
        .rename(columns={scores.index.name or "index": "symbol"})
        .sort_values(["score", "symbol"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["portfolio_weight"] = 0.0

    if variant == "equal_weight_benchmark":
        ranked["portfolio_weight"] = 1.0 / len(ranked)
    else:
        quintile_size = int(np.floor(len(ranked) * QUINTILE_FRACTION))
        top_rows = ranked.index[:quintile_size]
        if variant == "long_only_top_quintile":
            ranked.loc[top_rows, "portfolio_weight"] = 1.0 / quintile_size
        else:
            bottom_rows = ranked.index[-quintile_size:]
            ranked.loc[top_rows, "portfolio_weight"] = 0.5 / quintile_size
            ranked.loc[bottom_rows, "portfolio_weight"] = -0.5 / quintile_size

    return ranked.set_index("symbol")[["score", "rank", "portfolio_weight"]]


def calculate_turnover(
    current_weights: pd.Series,
    previous_weights: pd.Series | None = None,
) -> float:
    """Calculate dollar traded notional, including initial entry from cash."""
    current = current_weights.astype(float)
    previous = (
        pd.Series(dtype=float)
        if previous_weights is None
        else previous_weights.astype(float)
    )
    symbols = current.index.union(previous.index)
    return float(
        (current.reindex(symbols, fill_value=0.0)
         - previous.reindex(symbols, fill_value=0.0)).abs().sum()
    )


def calculate_drifted_weights(
    previous_weights: pd.Series,
    asset_returns: pd.Series,
) -> pd.Series:
    """Move signed holdings through one return period without renormalizing.

    Cash is implicit. For a long-only portfolio the weights sum to one; for the
    frozen long-short portfolio they sum to zero. In both cases portfolio
    equity after the holding period is ``1 + sum(weight * asset_return)``.
    """
    weights = previous_weights.astype(float)
    if not weights.index.is_unique:
        raise ValueError("previous_weights index must be unique")
    if not asset_returns.index.is_unique:
        raise ValueError("asset_returns index must be unique")

    missing_returns = weights.index.difference(asset_returns.index)
    if not missing_returns.empty:
        missing = ", ".join(map(str, missing_returns))
        raise ValueError(f"asset_returns is missing held symbols: {missing}")
    returns = asset_returns.reindex(weights.index).astype(float)
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("previous_weights must be finite")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("asset_returns must be finite")

    portfolio_return = float((weights * returns).sum())
    portfolio_equity = 1.0 + portfolio_return
    if not np.isfinite(portfolio_equity) or portfolio_equity <= 0:
        raise ValueError("portfolio equity must remain positive after returns")

    dollar_positions = weights * (1.0 + returns)
    return dollar_positions / portfolio_equity


def _performance_metrics(
    returns: pd.Series,
    daily: pd.DataFrame,
) -> dict[str, float | int]:
    values = returns.dropna().astype(float)
    number_of_days = len(values)
    if number_of_days == 0:
        cumulative_return = annualized_return = np.nan
        annualized_volatility = sharpe_ratio = maximum_drawdown = np.nan
    else:
        wealth = float((1.0 + values).prod())
        cumulative_return = wealth - 1.0
        annualized_return = (
            wealth ** (TRADING_DAYS_PER_YEAR / number_of_days) - 1.0
            if wealth > 0
            else np.nan
        )
        daily_std = float(values.std(ddof=1)) if number_of_days > 1 else np.nan
        annualized_volatility = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe_ratio = (
            float(values.mean()) / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
            if pd.notna(daily_std) and not np.isclose(daily_std, 0.0)
            else np.nan
        )
        equity = pd.concat([
            pd.Series([1.0]),
            (1.0 + values).cumprod().reset_index(drop=True),
        ], ignore_index=True)
        maximum_drawdown = float((equity / equity.cummax() - 1.0).min())

    valid_daily = daily.loc[daily["is_valid"]]
    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "maximum_drawdown": maximum_drawdown,
        "average_daily_turnover": float(valid_daily["turnover"].mean()),
        "total_traded_notional": float(valid_daily["turnover"].sum()),
        "number_of_trading_days": number_of_days,
        "average_number_of_long_positions": float(
            valid_daily["number_of_long_positions"].mean()
        ),
        "average_number_of_short_positions": float(
            valid_daily["number_of_short_positions"].mean()
        ),
    }


def _excess_metrics(excess_returns: pd.Series) -> dict[str, float | int]:
    values = excess_returns.dropna().astype(float)
    number_of_days = len(values)
    if number_of_days == 0:
        annualized_excess_return = tracking_error = information_ratio = np.nan
    else:
        annualized_excess_return = float(values.mean()) * TRADING_DAYS_PER_YEAR
        daily_std = float(values.std(ddof=1)) if number_of_days > 1 else np.nan
        tracking_error = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        information_ratio = (
            annualized_excess_return / tracking_error
            if pd.notna(tracking_error) and not np.isclose(tracking_error, 0.0)
            else np.nan
        )
    return {
        "annualized_excess_return": annualized_excess_return,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "number_of_comparison_days": number_of_days,
    }


def _open_return(
    data: pd.DataFrame,
    execution_date: pd.Timestamp,
    next_execution_date: pd.Timestamp,
) -> float | None:
    if execution_date not in data.index or next_execution_date not in data.index:
        return None
    first_open = data.at[execution_date, "Open"]
    second_open = data.at[next_execution_date, "Open"]
    if pd.isna(first_open) or pd.isna(second_open):
        return None
    return float(second_open / first_open - 1.0)


def _evaluate_schedules(
    schedules: list[dict],
    symbol_data: dict[str, pd.DataFrame],
) -> dict:
    position_rows = []
    daily_rows = []
    previous_pretrade_weights = None

    for schedule in schedules:
        weights = schedule["weights"]
        for symbol, row in weights.iterrows():
            position_rows.append({
                "signal_date": schedule["signal_date"],
                "execution_date": schedule["execution_date"],
                "next_execution_date": schedule["next_execution_date"],
                "symbol": symbol,
                "score": row["score"],
                "rank": int(row["rank"]),
                "portfolio_weight": row["portfolio_weight"],
            })

        active_weights = weights.loc[
            ~np.isclose(weights["portfolio_weight"], 0.0),
            "portfolio_weight",
        ]
        asset_returns = {}
        missing_symbols = []
        for symbol in active_weights.index:
            asset_return = _open_return(
                symbol_data[symbol],
                schedule["execution_date"],
                schedule["next_execution_date"],
            )
            if asset_return is None:
                missing_symbols.append(symbol)
            else:
                asset_returns[symbol] = asset_return

        is_valid = not missing_symbols
        if is_valid:
            turnover = calculate_turnover(
                active_weights,
                previous_pretrade_weights,
            )
            transaction_cost = turnover * COST_RATE
            gross_return = float(sum(
                active_weights[symbol] * asset_returns[symbol]
                for symbol in active_weights.index
            ))
            net_return = gross_return - transaction_cost
            previous_pretrade_weights = calculate_drifted_weights(
                active_weights,
                pd.Series(asset_returns, dtype=float),
            )
        else:
            turnover = transaction_cost = gross_return = net_return = np.nan

        daily_rows.append({
            "signal_date": schedule["signal_date"],
            "execution_date": schedule["execution_date"],
            "next_execution_date": schedule["next_execution_date"],
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "number_of_long_positions": int((active_weights > 0).sum()),
            "number_of_short_positions": int((active_weights < 0).sum()),
            "is_valid": is_valid,
            "invalid_reason": (
                "missing required Open prices: " + ", ".join(missing_symbols)
                if missing_symbols
                else None
            ),
        })

    position_columns = [
        "signal_date", "execution_date", "next_execution_date", "symbol",
        "score", "rank", "portfolio_weight",
    ]
    daily_columns = [
        "signal_date", "execution_date", "next_execution_date", "gross_return",
        "turnover", "transaction_cost", "net_return",
        "number_of_long_positions", "number_of_short_positions", "is_valid",
        "invalid_reason",
    ]
    positions = pd.DataFrame(position_rows, columns=position_columns)
    daily = pd.DataFrame(daily_rows, columns=daily_columns)
    if not daily.empty:
        daily = daily.set_index("signal_date", drop=False)

    performance = {
        "gross": _performance_metrics(daily["gross_return"], daily),
        "net": _performance_metrics(daily["net_return"], daily),
    }
    invalid_dates = daily.loc[~daily["is_valid"], [
        "signal_date", "execution_date", "next_execution_date", "invalid_reason",
    ]].reset_index(drop=True)
    return {
        "positions": positions,
        "daily": daily,
        "performance": performance,
        "invalid_dates": invalid_dates,
    }


def _build_schedules(
    scores: pd.Series,
    calendar: pd.DatetimeIndex,
    variant: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp | None,
) -> tuple[list[dict], pd.DataFrame]:
    schedules = []
    invalid_signal_rows = []
    for calendar_position, signal_date in enumerate(calendar):
        if signal_date < period_start:
            continue
        if period_end is not None and signal_date > period_end:
            continue
        if calendar_position + 2 >= len(calendar):
            continue
        execution_date = calendar[calendar_position + 1]
        next_execution_date = calendar[calendar_position + 2]
        if period_end is not None and next_execution_date > period_end:
            continue

        try:
            same_date_scores = scores.xs(signal_date, level="signal_date")
        except KeyError:
            same_date_scores = pd.Series(dtype=float)
        eligible_scores = same_date_scores.dropna()
        if len(eligible_scores) < MINIMUM_ELIGIBLE_SYMBOLS:
            invalid_signal_rows.append({
                "signal_date": signal_date,
                "reason": (
                    "fewer than "
                    f"{MINIMUM_ELIGIBLE_SYMBOLS} contemporaneously eligible symbols"
                ),
            })
            continue

        schedules.append({
            "signal_date": signal_date,
            "execution_date": execution_date,
            "next_execution_date": next_execution_date,
            "weights": form_portfolio_weights(eligible_scores, variant),
        })

    return schedules, pd.DataFrame(
        invalid_signal_rows,
        columns=["signal_date", "reason"],
    )


def _build_long_only_comparison(
    strategy: dict,
    benchmark: dict,
) -> tuple[pd.DataFrame, dict]:
    strategy_daily = strategy["daily"]
    benchmark_daily = benchmark["daily"]
    comparison = pd.DataFrame(index=strategy_daily.index)
    comparison["signal_date"] = strategy_daily["signal_date"]
    comparison["execution_date"] = strategy_daily["execution_date"]
    comparison["next_execution_date"] = strategy_daily["next_execution_date"]
    comparison["strategy_gross_return"] = strategy_daily["gross_return"]
    comparison["strategy_net_return"] = strategy_daily["net_return"]
    comparison["benchmark_gross_return"] = benchmark_daily["gross_return"]
    comparison["benchmark_net_return"] = benchmark_daily["net_return"]
    comparison["gross_excess_return"] = (
        comparison["strategy_gross_return"]
        - comparison["benchmark_gross_return"]
    )
    comparison["net_excess_return"] = (
        comparison["strategy_net_return"]
        - comparison["benchmark_net_return"]
    )
    comparison["is_valid"] = (
        strategy_daily["is_valid"] & benchmark_daily["is_valid"]
    )
    return comparison, {
        "gross": _excess_metrics(comparison["gross_excess_return"]),
        "net": _excess_metrics(comparison["net_excess_return"]),
    }


def run_volatility_factor_backtest(
    symbol_data: dict[str, pd.DataFrame],
    period: str = "validation",
    allow_holdout: bool = False,
) -> dict:
    """Run the two frozen portfolios in Validation or explicitly in Holdout.

    A desired weight schedule is always formed without checking future Opens.
    If any selected position lacks either required Open, that portfolio-date is
    marked invalid, no trade is recorded, and the other names are not reweighted.
    The next valid turnover is measured from the last valid drifted portfolio
    state. Results must not be interpreted if any invalid dates are reported,
    because the fail-closed path deliberately does not model holdings through
    the skipped execution interval.
    """
    if period not in {"validation", "holdout"}:
        raise ValueError("period must be validation or holdout")
    if allow_holdout and period != "holdout":
        raise ValueError("allow_holdout may only be used with period='holdout'")
    period_specification = get_research_period(
        period,
        allow_holdout=allow_holdout,
    )
    holdout_authorized = period == "holdout" and allow_holdout
    validated = _validate_symbol_data(
        symbol_data,
        allow_holdout=holdout_authorized,
    )
    scores = build_volatility_20_scores(
        validated,
        allow_holdout=holdout_authorized,
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
    period_score_mask = period_dates >= period_specification.start
    if period_specification.end is not None:
        period_score_mask &= period_dates <= period_specification.end

    result = {
        "frozen_alpha": FROZEN_ALPHA_V1,
        "direction": "positive",
        "feature_window": FEATURE_WINDOW,
        "quintile_fraction": QUINTILE_FRACTION,
        "cost_bps": COST_BPS,
        "period": period,
        "period_start": period_specification.start,
        "period_end": period_specification.end,
        "evaluation_end": long_only["daily"]["next_execution_date"].max(),
        "scores": scores.loc[period_score_mask],
        "invalid_signal_dates": invalid_signal_dates,
        "long_only": long_only,
        "long_short": long_short,
    }
    if period == "validation":
        result["validation_start"] = VALIDATION_PERIOD.start
        result["validation_end"] = VALIDATION_PERIOD.end
    return result
