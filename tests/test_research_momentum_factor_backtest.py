import numpy as np
import pandas as pd
import pytest

import src.research.momentum_factor_backtest as momentum_backtest
from src.research.features import MOMENTUM_FEATURE_COLUMNS
from src.research.momentum_factor_backtest import (
    build_momentum_scores,
    run_all_momentum_factor_backtests,
    run_momentum_factor_backtest,
)
from src.research.volatility_factor_backtest import form_portfolio_weights


SYMBOLS = [f"S{number:02d}" for number in range(10)]


def synthetic_symbol_data() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2020-12-01", "2021-03-31").append(
        pd.bdate_range("2023-12-01", "2024-03-31")
    )
    steps = np.arange(len(dates), dtype=float)
    result = {}
    for number, symbol in enumerate(SYMBOLS, start=1):
        result[symbol] = pd.DataFrame({
            "Open": (80.0 + number) * (1.0 + number * 0.00003) ** steps,
            "Close": (100.0 + number) * (1.0 + number * 0.00010) ** steps,
        }, index=dates)
    return result


def test_same_date_ranking_exact_quintiles_and_frozen_exposures():
    data = synthetic_symbol_data()
    signal_date = data[SYMBOLS[0]].index[10]
    scores = build_momentum_scores(data, "momentum_5").xs(
        signal_date,
        level="signal_date",
    )

    long_only = form_portfolio_weights(scores, "long_only_top_quintile")
    long_short = form_portfolio_weights(
        scores,
        "long_short_top_bottom_quintile",
    )

    assert long_only.loc[long_only["portfolio_weight"] > 0].index.tolist() == [
        "S09", "S08"
    ]
    assert long_only["portfolio_weight"].sum() == pytest.approx(1.0)
    long_weights = long_short.loc[long_short["portfolio_weight"] > 0]
    short_weights = long_short.loc[long_short["portfolio_weight"] < 0]
    assert long_weights.index.tolist() == ["S09", "S08"]
    assert short_weights.index.tolist() == ["S01", "S00"]
    assert long_weights["portfolio_weight"].sum() == pytest.approx(0.5)
    assert short_weights["portfolio_weight"].sum() == pytest.approx(-0.5)


def test_future_prices_do_not_change_previous_signals_or_trades():
    data = synthetic_symbol_data()
    original = run_momentum_factor_backtest(data, "momentum_5", "research")
    cutoff = pd.Timestamp("2021-02-15")
    changed = {symbol: frame.copy() for symbol, frame in data.items()}
    for number, frame in enumerate(changed.values(), start=2):
        frame.loc[frame.index >= cutoff, ["Open", "Close"]] *= number

    altered = run_momentum_factor_backtest(changed, "momentum_5", "research")
    position_columns = [
        "signal_date", "symbol", "score", "rank", "portfolio_weight"
    ]
    for portfolio_name in ("long_only", "long_short"):
        original_positions = original[portfolio_name]["positions"]
        altered_positions = altered[portfolio_name]["positions"]
        pd.testing.assert_frame_equal(
            original_positions.loc[
                original_positions["signal_date"] < cutoff,
                position_columns,
            ].reset_index(drop=True),
            altered_positions.loc[
                altered_positions["signal_date"] < cutoff,
                position_columns,
            ].reset_index(drop=True),
        )
        original_daily = original[portfolio_name]["daily"]
        altered_daily = altered[portfolio_name]["daily"]
        safe = original_daily["next_execution_date"] < cutoff
        changed_safe = altered_daily["next_execution_date"] < cutoff
        pd.testing.assert_frame_equal(
            original_daily.loc[safe].reset_index(drop=True),
            altered_daily.loc[changed_safe].reset_index(drop=True),
        )


def test_research_validation_are_disjoint_and_holdout_fails_closed():
    data = synthetic_symbol_data()
    research = run_momentum_factor_backtest(data, "momentum_5", "research")
    validation = run_momentum_factor_backtest(
        data,
        "momentum_5",
        "validation",
    )
    research_daily = research["long_only"]["daily"]
    validation_daily = validation["long_only"]["daily"]

    assert research_daily["next_execution_date"].max() < validation_daily[
        "signal_date"
    ].min()
    assert validation_daily["next_execution_date"].max() < pd.Timestamp(
        "2026-01-01"
    )

    holdout_row = data["S00"].iloc[[-1]].copy()
    holdout_row.index = pd.DatetimeIndex(["2026-01-02"])
    data["S00"] = pd.concat([data["S00"], holdout_row])
    with pytest.raises(ValueError, match="on or after 2026-01-01"):
        run_momentum_factor_backtest(data, "momentum_5", "validation")


def test_costs_turnover_and_invalid_dates_reuse_frozen_mechanics():
    data = synthetic_symbol_data()
    result = run_momentum_factor_backtest(data, "momentum_5", "research")
    for portfolio in (
        result["long_only"],
        result["long_only"]["benchmark"],
        result["long_short"],
    ):
        valid = portfolio["daily"].loc[portfolio["daily"]["is_valid"]]
        assert (valid["turnover"] >= 0).all()
        assert (valid["net_return"] <= valid["gross_return"]).all()
        assert portfolio["performance"]["net"]["cumulative_return"] <= (
            portfolio["performance"]["gross"]["cumulative_return"]
        )

    first = result["long_only"]["daily"].iloc[0]
    selected = result["long_only"]["positions"].loc[
        (result["long_only"]["positions"]["signal_date"] == first["signal_date"])
        & (result["long_only"]["positions"]["portfolio_weight"] > 0)
    ]
    changed = {symbol: frame.copy() for symbol, frame in data.items()}
    missing_symbol = selected.iloc[0]["symbol"]
    changed[missing_symbol].at[first["execution_date"], "Open"] = np.nan
    invalid = run_momentum_factor_backtest(
        changed,
        "momentum_5",
        "research",
    )["long_only"]
    assert len(invalid["invalid_dates"]) == 1
    assert not invalid["daily"].loc[first["signal_date"], "is_valid"]


def test_all_five_frozen_horizons_are_evaluated_without_reordering(monkeypatch):
    calls = []

    def fake_run(symbol_data, feature, period):
        calls.append((period, feature))
        return {"period": period, "feature": feature}

    monkeypatch.setattr(
        momentum_backtest,
        "run_momentum_factor_backtest",
        fake_run,
    )
    result = run_all_momentum_factor_backtests({"unused": pd.DataFrame()})

    assert list(result) == ["research", "validation"]
    assert list(result["research"]) == MOMENTUM_FEATURE_COLUMNS
    assert list(result["validation"]) == MOMENTUM_FEATURE_COLUMNS
    assert calls == [
        (period, feature)
        for period in ("research", "validation")
        for feature in MOMENTUM_FEATURE_COLUMNS
    ]
