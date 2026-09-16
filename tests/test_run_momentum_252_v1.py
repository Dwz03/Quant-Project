import numpy as np
import pandas as pd
import pytest

import scripts.run_momentum_252_v1 as runner
from src.research.momentum_factor_backtest import (
    FROZEN_MOMENTUM_V1,
    FROZEN_MOMENTUM_V1_HORIZON,
    FROZEN_MOMENTUM_V1_PRIMARY_PORTFOLIO,
    run_momentum_252_v1_backtest,
    run_momentum_factor_backtest,
)
from src.research.volatility_factor_backtest import COST_BPS


SYMBOLS = [f"S{number:02d}" for number in range(10)]


def synthetic_data(start: str, end: str) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(start, end)
    steps = np.arange(len(dates), dtype=float)
    return {
        symbol: pd.DataFrame({
            "Open": (80.0 + number) * (1.0 + number * 0.00003) ** steps,
            "Close": (100.0 + number) * (1.0 + number * 0.00010) ** steps,
        }, index=dates)
        for number, symbol in enumerate(SYMBOLS, start=1)
    }


def test_holdout_requires_explicit_allow_flag_before_loading():
    with pytest.raises(ValueError, match="explicit --allow-holdout"):
        runner.main(
            argv=["--period", "holdout"],
            loader=lambda *args: pytest.fail("loader must not run"),
        )
    with pytest.raises(ValueError, match="explicit allow_holdout=True"):
        run_momentum_252_v1_backtest({}, period="holdout")


def test_frozen_v1_constants_portfolio_cost_and_execution_are_unchanged():
    assert FROZEN_MOMENTUM_V1 == "momentum_252"
    assert FROZEN_MOMENTUM_V1_HORIZON == 252
    assert FROZEN_MOMENTUM_V1_PRIMARY_PORTFOLIO == "long_only_top_quintile"
    assert COST_BPS == 10

    data = synthetic_data("2023-01-02", "2024-03-31")
    result = run_momentum_252_v1_backtest(data, period="validation")
    first = result["long_only"]["daily"].iloc[0]
    positions = result["long_only"]["positions"]
    selected = positions.loc[
        (positions["signal_date"] == first["signal_date"])
        & (positions["portfolio_weight"] > 0)
    ]

    assert result["horizon"] == 252
    assert result["cost_bps"] == 10
    assert selected["symbol"].tolist() == ["S09", "S08"]
    assert selected["portfolio_weight"].sum() == pytest.approx(1.0)
    assert first["signal_date"] < first["execution_date"]
    assert first["execution_date"] < first["next_execution_date"]


@pytest.mark.parametrize(
    ("period", "start", "end"),
    [
        ("research", "2020-01-02", "2021-03-31"),
        ("validation", "2023-01-02", "2024-03-31"),
    ],
)
def test_research_validation_behavior_matches_existing_momentum_engine(
    period,
    start,
    end,
):
    data = synthetic_data(start, end)
    frozen = run_momentum_252_v1_backtest(data, period=period)
    existing = run_momentum_factor_backtest(
        data,
        feature="momentum_252",
        period=period,
    )

    for portfolio_name in ("long_only", "long_short"):
        pd.testing.assert_frame_equal(
            frozen[portfolio_name]["daily"],
            existing[portfolio_name]["daily"],
        )
        assert frozen[portfolio_name]["performance"] == existing[
            portfolio_name
        ]["performance"]


def test_explicit_guard_allows_only_synthetic_holdout_evaluation():
    data = synthetic_data("2024-12-02", "2026-02-27")
    result = run_momentum_252_v1_backtest(
        data,
        period="holdout",
        allow_holdout=True,
    )
    daily = result["long_only"]["daily"]

    assert daily["signal_date"].min() >= pd.Timestamp("2026-01-01")
    assert daily.iloc[0]["signal_date"] < daily.iloc[0]["execution_date"]
    assert daily.iloc[0]["execution_date"] < daily.iloc[0]["next_execution_date"]


def test_warning_is_printed_before_holdout_loading(monkeypatch, capsys):
    class LoadingStopped(Exception):
        pass

    monkeypatch.setattr(runner, "holdout_load_end", lambda: "2026-03-01")

    def stop_after_warning(**kwargs):
        output = capsys.readouterr().out
        assert "ONE-TIME CONFIRMATORY HOLDOUT EVALUATION" in output
        assert "HOLDOUT RESULTS MUST NOT BE USED TO RETUNE MOMENTUM 252 V1." in output
        assert "must be versioned as V2" in output
        assert "this 2026 sample as untouched Holdout" in output
        raise LoadingStopped

    monkeypatch.setattr(runner, "load_adjusted_ohlc_data", stop_after_warning)
    with pytest.raises(LoadingStopped):
        runner.main(argv=["--period", "holdout", "--allow-holdout"])


def test_cli_exposes_no_tuning_or_follow_on_selection_arguments():
    option_strings = {
        option
        for action in runner.build_parser()._actions
        for option in action.option_strings
    }
    assert option_strings == {"-h", "--help", "--period", "--allow-holdout"}
    for forbidden in (
        "--horizon", "--feature", "--quantile", "--cost", "--direction",
        "--universe", "--weighting", "--rebalance", "--select", "--tune",
    ):
        assert forbidden not in option_strings


def test_primary_gate_uses_only_the_three_frozen_conditions():
    empty = pd.DataFrame()
    result = {
        "invalid_signal_dates": empty,
        "long_only": {
            "invalid_dates": empty,
            "benchmark": {"invalid_dates": empty},
            "excess_performance": {
                "net": {
                    "annualized_excess_return": 0.01,
                    "information_ratio": 0.10,
                },
            },
        },
        "long_short": {"invalid_dates": empty},
    }

    gate = runner.primary_holdout_gate(result)

    assert gate["passed"]
    assert gate["invalid_dates_clear"]
    assert gate["net_excess_return_positive"]
    assert gate["net_information_ratio_positive"]
