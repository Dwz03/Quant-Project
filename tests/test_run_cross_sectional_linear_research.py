from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_cross_sectional_linear_research as runner
from src.research.cross_sectional_linear import PRIMARY_FEATURES_V1
from src.research.universes import LARGE_LIQUID_US_EQUITIES_V1


def evaluation(rmse, r2, mean_ic):
    return {
        "numerical_metrics": {"rmse": rmse, "r2": r2},
        "rank_ic_summary": {
            "mean_prediction_ic": mean_ic,
            "median_prediction_ic": mean_ic,
            "std_prediction_ic": 0.1,
            "prediction_icir": mean_ic / 0.1,
            "fraction_positive_prediction_ic": 0.6,
            "number_of_dates": 10,
            "mean_number_of_symbols": 80.0,
        },
    }


def test_cli_loads_only_pre_holdout_data_and_prints_results(monkeypatch, capsys):
    calls = {}
    panel = pd.DataFrame({"target": [0.01]})
    expected_result = {
        "models": {
            "volatility_20_only": {
                "features": ("volatility_20",),
                "model": SimpleNamespace(),
                "intercept": 0.001,
                "coefficients": {"volatility_20": -0.002},
                "research": evaluation(0.02, 0.01, 0.03),
                "validation": evaluation(0.021, -0.01, 0.02),
            }
        },
        "validation_baselines": {
            "zero_return": {"rmse": 0.022, "r2": -0.02},
            "research_mean": {
                "research_mean": 0.0005,
                "rmse": 0.0215,
                "r2": -0.015,
            },
            "equal_score": {"rank_ic": None},
        },
        "validation_comparison": pd.DataFrame({
            "model": ["volatility_20_only"],
            "validation_rmse": [0.021],
            "validation_r2": [-0.01],
            "validation_mean_rank_ic": [0.02],
            "validation_icir": [0.2],
            "validation_fraction_positive_ic": [0.6],
        }),
    }

    def fake_load(symbols, start, end, loader):
        calls["symbols"] = tuple(symbols)
        calls["start"] = start
        calls["end"] = end
        return {"AAA": pd.DataFrame({"Close": [100.0]})}

    def fake_research(received_panel, min_symbols):
        calls["panel"] = received_panel
        calls["min_symbols"] = min_symbols
        return expected_result

    monkeypatch.setattr(runner, "load_symbol_data", fake_load)
    monkeypatch.setattr(
        runner,
        "build_multi_asset_feature_panel",
        lambda symbol_data: panel,
    )
    monkeypatch.setattr(
        runner,
        "run_cross_sectional_linear_research",
        fake_research,
    )

    result = runner.main(
        argv=["--universe", "large", "--min-symbols", "50"],
        loader=lambda symbol, start, end: pytest.fail("loader should be wrapped"),
    )

    assert result is expected_result
    assert calls["symbols"] == LARGE_LIQUID_US_EQUITIES_V1
    assert calls["start"] == "2021-01-01"
    assert calls["end"] == "2026-01-01"
    assert calls["min_symbols"] == 50
    assert calls["panel"] is panel

    output = capsys.readouterr().out
    assert "Research and Validation Only" in output
    assert f"Frozen features: {', '.join(PRIMARY_FEATURES_V1)}" in output
    assert "Model: volatility_20_only" in output
    assert "Validation predictions:" in output
    assert "Validation Baselines:" in output
    assert "Equal Score Rank IC: undefined" in output
    assert "Validation Model Comparison:" in output
