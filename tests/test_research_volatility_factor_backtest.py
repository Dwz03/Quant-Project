import numpy as np
import pandas as pd
import pytest

from src.research.volatility_factor_backtest import (
    COST_BPS,
    COST_RATE,
    FROZEN_ALPHA_V1,
    calculate_drifted_weights,
    calculate_turnover,
    form_portfolio_weights,
    run_volatility_factor_backtest,
)


SYMBOLS = [f"S{i:02d}" for i in range(10)]


def synthetic_symbol_data() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2023-11-20", periods=55)
    steps = np.arange(len(dates), dtype=float)
    result = {}
    for symbol_number, symbol in enumerate(SYMBOLS, start=1):
        amplitude = symbol_number * 0.0004
        close_returns = amplitude * np.where(steps % 2 == 0, 1.0, -1.0)
        close = 100.0 * np.cumprod(1.0 + close_returns)
        open_price = (80.0 + symbol_number) * (
            1.0 + symbol_number * 0.0002
        ) ** steps
        result[symbol] = pd.DataFrame(
            {"Open": open_price, "Close": close},
            index=dates,
        )
    return result


def synthetic_holdout_data() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2025-11-03", periods=80)
    steps = np.arange(len(dates), dtype=float)
    result = {}
    for symbol_number, symbol in enumerate(SYMBOLS, start=1):
        amplitude = symbol_number * 0.0004
        close_returns = amplitude * np.where(steps % 2 == 0, 1.0, -1.0)
        result[symbol] = pd.DataFrame({
            "Open": (80.0 + symbol_number)
            * (1.0 + symbol_number * 0.0002) ** steps,
            "Close": 100.0 * np.cumprod(1.0 + close_returns),
        }, index=dates)
    return result


def ranked_scores() -> pd.Series:
    return pd.Series(
        np.arange(1.0, 11.0),
        index=pd.Index(SYMBOLS, name="symbol"),
    )


def test_frozen_constants_and_exact_quintile_portfolios():
    assert FROZEN_ALPHA_V1 == "volatility_20"
    assert COST_BPS == 10
    assert COST_RATE == pytest.approx(0.001)

    long_only = form_portfolio_weights(
        ranked_scores(),
        "long_only_top_quintile",
    )
    long_short = form_portfolio_weights(
        ranked_scores(),
        "long_short_top_bottom_quintile",
    )

    assert long_only.index[:2].tolist() == ["S09", "S08"]
    assert long_only.loc[long_only["portfolio_weight"] > 0].index.tolist() == [
        "S09", "S08",
    ]
    assert long_only["portfolio_weight"].sum() == pytest.approx(1.0)

    long_weights = long_short.loc[long_short["portfolio_weight"] > 0]
    short_weights = long_short.loc[long_short["portfolio_weight"] < 0]
    assert long_weights.index.tolist() == ["S09", "S08"]
    assert short_weights.index.tolist() == ["S01", "S00"]
    assert long_weights["portfolio_weight"].sum() == pytest.approx(0.5)
    assert short_weights["portfolio_weight"].sum() == pytest.approx(-0.5)
    assert long_short["portfolio_weight"].sum() == pytest.approx(0.0)
    assert long_short["portfolio_weight"].abs().sum() == pytest.approx(1.0)


def test_signal_executes_next_open_and_earns_following_open_return():
    data = synthetic_symbol_data()
    result = run_volatility_factor_backtest(data)
    daily = result["long_only"]["daily"]
    first = daily.iloc[0]
    positions = result["long_only"]["positions"]
    first_positions = positions.loc[
        positions["signal_date"] == first["signal_date"]
    ]
    selected = first_positions.loc[first_positions["portfolio_weight"] > 0]

    assert first["signal_date"] < first["execution_date"]
    assert first["execution_date"] < first["next_execution_date"]
    expected_return = sum(
        row["portfolio_weight"]
        * (
            data[row["symbol"]].at[first["next_execution_date"], "Open"]
            / data[row["symbol"]].at[first["execution_date"], "Open"]
            - 1.0
        )
        for _, row in selected.iterrows()
    )
    assert first["gross_return"] == pytest.approx(expected_return)


def test_future_open_and_target_mutations_cannot_change_formed_weights():
    original_data = synthetic_symbol_data()
    original = run_volatility_factor_backtest(original_data)
    first_daily = original["long_only"]["daily"].iloc[0]
    signal_date = first_daily["signal_date"]
    execution_date = first_daily["execution_date"]

    changed_data = {symbol: data.copy() for symbol, data in original_data.items()}
    for data in changed_data.values():
        data["target"] = np.linspace(-10.0, 10.0, len(data))
        data.loc[data.index >= execution_date, "Open"] *= 7.0
        data.loc[data.index > signal_date, "target"] *= -100.0

    changed = run_volatility_factor_backtest(changed_data)
    columns = ["signal_date", "symbol", "score", "rank", "portfolio_weight"]
    pd.testing.assert_frame_equal(
        original["long_only"]["positions"][columns],
        changed["long_only"]["positions"][columns],
    )
    pd.testing.assert_frame_equal(
        original["long_short"]["positions"][columns],
        changed["long_short"]["positions"][columns],
    )


def test_missing_future_open_is_not_used_for_eligibility_or_reweighting():
    data = synthetic_symbol_data()
    original = run_volatility_factor_backtest(data)
    first = original["long_only"]["daily"].iloc[0]
    signal_date = first["signal_date"]
    execution_date = first["execution_date"]
    original_positions = original["long_only"]["positions"]
    selected = original_positions.loc[
        (original_positions["signal_date"] == signal_date)
        & (original_positions["portfolio_weight"] > 0)
    ]
    missing_symbol = selected.iloc[0]["symbol"]

    changed_data = {symbol: frame.copy() for symbol, frame in data.items()}
    changed_data[missing_symbol].at[execution_date, "Open"] = np.nan
    changed = run_volatility_factor_backtest(changed_data)
    changed_positions = changed["long_only"]["positions"]
    changed_selected = changed_positions.loc[
        (changed_positions["signal_date"] == signal_date)
        & (changed_positions["portfolio_weight"] > 0)
    ]
    changed_daily = changed["long_only"]["daily"].loc[signal_date]

    pd.testing.assert_frame_equal(
        selected.reset_index(drop=True),
        changed_selected.reset_index(drop=True),
    )
    assert changed_selected["portfolio_weight"].sum() == pytest.approx(1.0)
    assert not changed_daily["is_valid"]
    assert missing_symbol in changed_daily["invalid_reason"]
    assert np.isnan(changed_daily["gross_return"])
    assert np.isnan(changed_daily["turnover"])


def test_ranking_uses_only_the_same_date_cross_section():
    first_scores = ranked_scores()
    future_scores = ranked_scores() * -1_000.0
    first = form_portfolio_weights(first_scores, "long_only_top_quintile")
    _ = form_portfolio_weights(future_scores, "long_only_top_quintile")
    repeated = form_portfolio_weights(first_scores, "long_only_top_quintile")

    pd.testing.assert_frame_equal(first, repeated)


def test_turnover_and_initial_cost_are_exact():
    initial = pd.Series({"AAA": 0.6, "BBB": 0.4})
    next_weights = pd.Series({"AAA": 0.2, "CCC": 0.8})

    assert calculate_turnover(initial) == pytest.approx(1.0)
    assert calculate_turnover(next_weights, initial) == pytest.approx(1.6)

    result = run_volatility_factor_backtest(synthetic_symbol_data())
    for portfolio in (
        result["long_only"],
        result["long_only"]["benchmark"],
        result["long_short"],
    ):
        first = portfolio["daily"].loc[portfolio["daily"]["is_valid"]].iloc[0]
        assert first["turnover"] == pytest.approx(1.0)
        assert first["transaction_cost"] == pytest.approx(0.001)
        assert first["net_return"] == pytest.approx(
            first["gross_return"] - 0.001
        )


def test_long_only_weights_drift_before_the_next_rebalance():
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
    returns = pd.Series({"AAA": 0.10, "BBB": -0.05})

    drifted = calculate_drifted_weights(weights, returns)
    expected_equity = 1.0 + 0.5 * 0.10 + 0.5 * -0.05

    assert drifted["AAA"] == pytest.approx(0.5 * 1.10 / expected_equity)
    assert drifted["BBB"] == pytest.approx(0.5 * 0.95 / expected_equity)
    assert drifted.sum() == pytest.approx(1.0)
    assert calculate_turnover(weights, drifted) > 0.0


def test_long_short_weights_drift_without_gross_or_net_renormalization():
    weights = pd.Series({
        "LONG_A": 0.25,
        "LONG_B": 0.25,
        "SHORT_A": -0.25,
        "SHORT_B": -0.25,
    })
    returns = pd.Series({
        "LONG_A": 0.10,
        "LONG_B": -0.02,
        "SHORT_A": 0.04,
        "SHORT_B": -0.08,
    })

    drifted = calculate_drifted_weights(weights, returns)
    portfolio_return = float((weights * returns).sum())
    expected = weights * (1.0 + returns) / (1.0 + portfolio_return)

    pd.testing.assert_series_equal(drifted, expected)
    assert not np.isclose(drifted.abs().sum(), 1.0)
    assert not np.isclose(drifted.sum(), 0.0)


def test_second_rebalance_turnover_uses_prior_drifted_weights():
    data = synthetic_symbol_data()
    result = run_volatility_factor_backtest(data)
    portfolio = result["long_only"]
    first_daily = portfolio["daily"].iloc[0]
    second_daily = portfolio["daily"].iloc[1]

    def active_weights(signal_date):
        rows = portfolio["positions"].loc[
            portfolio["positions"]["signal_date"] == signal_date
        ]
        return rows.loc[rows["portfolio_weight"] != 0].set_index(
            "symbol"
        )["portfolio_weight"]

    first_weights = active_weights(first_daily["signal_date"])
    second_weights = active_weights(second_daily["signal_date"])
    first_returns = pd.Series({
        symbol: (
            data[symbol].at[first_daily["next_execution_date"], "Open"]
            / data[symbol].at[first_daily["execution_date"], "Open"]
            - 1.0
        )
        for symbol in first_weights.index
    })
    drifted = calculate_drifted_weights(first_weights, first_returns)
    expected_turnover = calculate_turnover(second_weights, drifted)

    pd.testing.assert_series_equal(
        first_weights.sort_index(),
        second_weights.sort_index(),
    )
    assert expected_turnover > 0.0
    assert second_daily["turnover"] == pytest.approx(expected_turnover)
    assert second_daily["transaction_cost"] == pytest.approx(
        expected_turnover * COST_RATE
    )


def test_validation_only_and_holdout_is_rejected():
    result = run_volatility_factor_backtest(synthetic_symbol_data())
    for name in ("long_only", "long_short"):
        daily = result[name]["daily"]
        assert daily["signal_date"].min() >= pd.Timestamp("2024-01-01")
        assert daily["next_execution_date"].max() <= pd.Timestamp("2025-12-31")

    data = synthetic_symbol_data()
    holdout_row = data["S00"].iloc[[-1]].copy()
    holdout_row.index = pd.DatetimeIndex(["2026-01-02"])
    data["S00"] = pd.concat([data["S00"], holdout_row])
    with pytest.raises(ValueError, match="on or after 2026-01-01"):
        run_volatility_factor_backtest(data)


def test_holdout_requires_explicit_authorization():
    with pytest.raises(ValueError, match="explicit allow_holdout=True"):
        run_volatility_factor_backtest(
            synthetic_holdout_data(),
            period="holdout",
        )


def test_holdout_uses_warmup_only_for_features_and_reports_realizable_periods():
    data = synthetic_holdout_data()
    result = run_volatility_factor_backtest(
        data,
        period="holdout",
        allow_holdout=True,
    )
    daily = result["long_only"]["daily"]
    supplied_dates = data[SYMBOLS[0]].index

    assert result["period"] == "holdout"
    assert result["period_start"] == pd.Timestamp("2026-01-01")
    assert result["period_end"] is None
    assert daily["signal_date"].min() >= pd.Timestamp("2026-01-01")
    assert daily["execution_date"].min() >= pd.Timestamp("2026-01-01")
    assert daily["next_execution_date"].max() == supplied_dates[-1]
    assert daily.iloc[-1]["signal_date"] == supplied_dates[-3]
    assert not daily["signal_date"].isin(supplied_dates[-2:]).any()
    first = daily.iloc[0]
    first_location = supplied_dates.get_loc(first["signal_date"])
    assert first["execution_date"] == supplied_dates[first_location + 1]
    assert first["next_execution_date"] == supplied_dates[first_location + 2]
    assert result["scores"].index.get_level_values(
        "signal_date"
    ).min() >= pd.Timestamp("2026-01-01")

    changed = {symbol: frame.copy() for symbol, frame in data.items()}
    for frame in changed.values():
        pre_holdout = frame.index < pd.Timestamp("2026-01-01")
        frame.loc[pre_holdout, "Open"] *= 100.0
    changed_result = run_volatility_factor_backtest(
        changed,
        period="holdout",
        allow_holdout=True,
    )
    pd.testing.assert_frame_equal(
        daily,
        changed_result["long_only"]["daily"],
    )


def test_holdout_ranking_ignores_future_opens_and_targets(monkeypatch):
    from sklearn.linear_model import LinearRegression

    data = synthetic_holdout_data()
    original = run_volatility_factor_backtest(
        data,
        period="holdout",
        allow_holdout=True,
    )
    first_daily = original["long_only"]["daily"].iloc[0]
    signal_date = first_daily["signal_date"]
    execution_date = first_daily["execution_date"]

    changed = {symbol: frame.copy() for symbol, frame in data.items()}
    for frame in changed.values():
        frame["target"] = np.linspace(-1_000.0, 1_000.0, len(frame))
        frame.loc[frame.index >= execution_date, "Open"] *= 5.0
        frame.loc[frame.index > signal_date, "target"] *= -10.0

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Holdout economic evaluation must not fit a regression")

    monkeypatch.setattr(LinearRegression, "fit", forbidden_fit)
    changed_result = run_volatility_factor_backtest(
        changed,
        period="holdout",
        allow_holdout=True,
    )
    columns = ["signal_date", "symbol", "score", "rank", "portfolio_weight"]
    pd.testing.assert_frame_equal(
        original["long_only"]["positions"][columns],
        changed_result["long_only"]["positions"][columns],
    )
    pd.testing.assert_frame_equal(
        original["long_short"]["positions"][columns],
        changed_result["long_short"]["positions"][columns],
    )


def test_holdout_turnover_remains_drift_adjusted():
    data = synthetic_holdout_data()
    portfolio = run_volatility_factor_backtest(
        data,
        period="holdout",
        allow_holdout=True,
    )["long_only"]
    first_daily = portfolio["daily"].iloc[0]
    second_daily = portfolio["daily"].iloc[1]

    def active_weights(signal_date):
        rows = portfolio["positions"].loc[
            portfolio["positions"]["signal_date"] == signal_date
        ]
        return rows.loc[rows["portfolio_weight"] != 0].set_index(
            "symbol"
        )["portfolio_weight"]

    first_weights = active_weights(first_daily["signal_date"])
    second_weights = active_weights(second_daily["signal_date"])
    holding_returns = pd.Series({
        symbol: (
            data[symbol].at[first_daily["next_execution_date"], "Open"]
            / data[symbol].at[first_daily["execution_date"], "Open"]
            - 1.0
        )
        for symbol in first_weights.index
    })
    expected_turnover = calculate_turnover(
        second_weights,
        calculate_drifted_weights(first_weights, holding_returns),
    )

    assert second_daily["turnover"] == pytest.approx(expected_turnover)
    assert second_daily["transaction_cost"] == pytest.approx(
        expected_turnover * 0.001
    )


def test_holdout_and_validation_use_identical_performance_metric_schema():
    validation = run_volatility_factor_backtest(synthetic_symbol_data())
    holdout = run_volatility_factor_backtest(
        synthetic_holdout_data(),
        period="holdout",
        allow_holdout=True,
    )

    for portfolio_name in ("long_only", "long_short"):
        for return_type in ("gross", "net"):
            assert set(
                validation[portfolio_name]["performance"][return_type]
            ) == set(holdout[portfolio_name]["performance"][return_type])


def test_long_only_benchmark_and_daily_detail_are_aligned():
    result = run_volatility_factor_backtest(synthetic_symbol_data())
    long_only = result["long_only"]
    comparison = long_only["comparison_daily"]

    assert comparison["gross_excess_return"].equals(
        comparison["strategy_gross_return"]
        - comparison["benchmark_gross_return"]
    )
    assert comparison["net_excess_return"].equals(
        comparison["strategy_net_return"]
        - comparison["benchmark_net_return"]
    )
    assert {
        "signal_date", "execution_date", "next_execution_date", "symbol",
        "score", "rank", "portfolio_weight",
    }.issubset(long_only["positions"].columns)
    assert {
        "gross_return", "turnover", "transaction_cost", "net_return",
        "benchmark_return", "net_benchmark_return", "excess_return",
        "net_excess_return",
    }.issubset(long_only["daily"].columns)


def test_open_column_is_required_and_never_replaced_with_close():
    data = synthetic_symbol_data()
    data["S00"] = data["S00"].drop(columns="Open")
    with pytest.raises(ValueError, match="missing: Open"):
        run_volatility_factor_backtest(data)
