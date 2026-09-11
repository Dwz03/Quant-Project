import numpy as np
import pandas as pd
import pytest

import scripts.run_volatility_factor_backtest as runner
from src.research.universes import LARGE_LIQUID_US_EQUITIES_V1


def test_holdout_load_end_converts_asia_timestamp_to_new_york_date():
    as_of = pd.Timestamp("2026-09-12 01:00", tz="Asia/Singapore")

    assert runner.holdout_load_end(as_of) == "2026-09-11"


def test_holdout_load_end_converts_utc_timestamp_to_new_york_date():
    as_of = pd.Timestamp("2026-09-12 02:00", tz="UTC")

    assert runner.holdout_load_end(as_of) == "2026-09-11"


def test_holdout_load_end_preserves_naive_control_calendar_date():
    as_of = pd.Timestamp("2026-09-12 23:45:00")

    assert runner.holdout_load_end(as_of) == "2026-09-12"


def test_cli_uses_frozen_arguments_pre_holdout_data_and_prints_results(capsys):
    calls = []
    dates = pd.bdate_range("2023-11-20", periods=55)
    steps = np.arange(len(dates), dtype=float)

    def fake_loader(symbol, start, end):
        calls.append((symbol, start, end))
        symbol_number = LARGE_LIQUID_US_EQUITIES_V1.index(symbol) + 1
        amplitude = symbol_number * 0.00002
        close_returns = amplitude * np.where(steps % 2 == 0, 1.0, -1.0)
        frame = pd.DataFrame({
            "Open": (80.0 + symbol_number) * (
                1.0 + symbol_number * 0.00001
            ) ** steps,
            "Close": 100.0 * np.cumprod(1.0 + close_returns),
        }, index=dates[::-1])
        if symbol == LARGE_LIQUID_US_EQUITIES_V1[-1]:
            validation_dates = dates[dates >= pd.Timestamp("2024-01-01")]
            frame.at[validation_dates[1], "Open"] = np.nan
        return frame

    result = runner.main(argv=["--universe", "large"], loader=fake_loader)

    assert len(calls) == len(LARGE_LIQUID_US_EQUITIES_V1)
    assert all(start == "2023-11-01" for _, start, _ in calls)
    assert all(end == "2026-01-01" for _, _, end in calls)
    assert result["cost_bps"] == 10
    assert result["frozen_alpha"] == "volatility_20"
    assert result["scores"].index.get_level_values("signal_date").max() < pd.Timestamp(
        "2026-01-01"
    )
    assert result["scores"].index.get_level_values("signal_date").min() >= pd.Timestamp(
        "2024-01-01"
    )

    output = capsys.readouterr().out
    assert "Validation Economic Backtest" in output
    assert "Long-Only Top Quintile:" in output
    assert "Equal-Weight Benchmark:" in output
    assert "Gross Excess:" in output
    assert "Net Excess:" in output
    assert "Long-Short Top/Bottom Quintile:" in output
    assert "Transaction Cost: 10 bps" in output
    assert "Open_{t+1} to Open_{t+2}" in output
    assert "WARNING: INVALID BACKTEST DATES DETECTED" in output
    assert "ECONOMIC RESULTS MUST NOT BE INTERPRETED" in output


def test_cli_exposes_no_tuning_arguments():
    option_strings = {
        option
        for action in runner.build_parser()._actions
        for option in action.option_strings
    }
    assert "--quantile" not in option_strings
    assert "--cost" not in option_strings
    assert "--feature-window" not in option_strings
    assert "--direction" not in option_strings


def test_holdout_cli_requires_explicit_authorization_before_loading():
    def forbidden_loader(*args):
        pytest.fail("loader must not run without explicit Holdout authorization")

    with pytest.raises(ValueError, match="explicit --allow-holdout"):
        runner.main(
            argv=["--universe", "large", "--period", "holdout"],
            loader=forbidden_loader,
        )


def test_explicit_holdout_cli_uses_warmup_and_prints_confirmation_warning(
    monkeypatch,
    capsys,
):
    calls = []
    dates = pd.bdate_range("2025-11-03", periods=80)
    steps = np.arange(len(dates), dtype=float)

    monkeypatch.setattr(runner, "holdout_load_end", lambda: "2026-03-01")

    def fake_loader(symbol, start, end):
        calls.append((symbol, start, end))
        symbol_number = LARGE_LIQUID_US_EQUITIES_V1.index(symbol) + 1
        amplitude = symbol_number * 0.00002
        close_returns = amplitude * np.where(steps % 2 == 0, 1.0, -1.0)
        return pd.DataFrame({
            "Open": (80.0 + symbol_number)
            * (1.0 + symbol_number * 0.00001) ** steps,
            "Close": 100.0 * np.cumprod(1.0 + close_returns),
        }, index=dates)

    result = runner.main(
        argv=[
            "--universe", "large",
            "--period", "holdout",
            "--allow-holdout",
        ],
        loader=fake_loader,
    )

    assert len(calls) == len(LARGE_LIQUID_US_EQUITIES_V1)
    assert all(start == "2025-11-01" for _, start, _ in calls)
    assert all(end == "2026-03-01" for _, _, end in calls)
    assert result["period"] == "holdout"
    assert result["long_only"]["daily"]["signal_date"].min() >= pd.Timestamp(
        "2026-01-01"
    )

    output = capsys.readouterr().out
    assert "ONE-TIME CONFIRMATORY HOLDOUT EVALUATION" in output
    assert "HOLDOUT RESULTS MUST NOT BE USED TO RETUNE V1" in output
    assert "must be versioned as V2" in output
    assert "Holdout Economic Backtest" in output


def test_holdout_invalid_dates_print_exact_noninterpretation_warning(
    monkeypatch,
    capsys,
):
    dates = pd.bdate_range("2025-11-03", periods=80)
    steps = np.arange(len(dates), dtype=float)
    holdout_dates = dates[dates >= pd.Timestamp("2026-01-01")]
    monkeypatch.setattr(runner, "holdout_load_end", lambda: "2026-03-01")

    def fake_loader(symbol, start, end):
        symbol_number = LARGE_LIQUID_US_EQUITIES_V1.index(symbol) + 1
        amplitude = symbol_number * 0.00002
        close_returns = amplitude * np.where(steps % 2 == 0, 1.0, -1.0)
        frame = pd.DataFrame({
            "Open": (80.0 + symbol_number)
            * (1.0 + symbol_number * 0.00001) ** steps,
            "Close": 100.0 * np.cumprod(1.0 + close_returns),
        }, index=dates)
        if symbol == LARGE_LIQUID_US_EQUITIES_V1[-1]:
            frame.at[holdout_dates[1], "Open"] = np.nan
        return frame

    runner.main(
        argv=["--period", "holdout", "--allow-holdout"],
        loader=fake_loader,
    )

    output = capsys.readouterr().out
    assert "WARNING: INVALID HOLDOUT DATES DETECTED" in output
    assert "HOLDOUT ECONOMIC RESULTS MUST NOT BE INTERPRETED." in output
    assert "signal formation invalid count:" in output
    assert "long-only invalid count:" in output
    assert "benchmark invalid count:" in output
    assert "long-short invalid count:" in output
