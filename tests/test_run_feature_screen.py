import numpy as np
import pandas as pd
import pytest

import scripts.run_feature_screen as runner
from scripts.run_feature_screen import (
    DEFAULT_SYMBOLS,
    build_parser,
    main,
)
from src.research.features import FEATURE_COLUMNS_V1
from src.research.universes import LARGE_LIQUID_US_EQUITIES_V1


def test_parser_uses_requested_default_universe():
    args = build_parser().parse_args([])

    assert args.symbols == list(DEFAULT_SYMBOLS)
    assert args.method == "spearman"
    assert args.symbols == [
        "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
        "META", "JPM", "XOM", "JNJ", "KO",
    ]


def test_main_forwards_symbols_and_dates_with_offline_loader(capsys):
    calls = []
    index = pd.date_range("2024-01-01", periods=75)
    steps = np.arange(len(index), dtype=float)

    def fake_loader(symbol, start, end):
        calls.append((symbol, start, end))
        offset = 5.0 * ["AAA", "BBB"].index(symbol)
        return pd.DataFrame({
            "Close": 100.0 + offset + 0.1 * steps + np.sin(steps / 3.0)
        }, index=index[::-1])

    result = main(
        argv=[
            "--symbols", "AAA", "BBB",
            "--start", "2024-01-01",
            "--end", "2024-04-01",
        ],
        loader=fake_loader,
    )

    assert calls == [
        ("AAA", "2024-01-01", "2024-04-01"),
        ("BBB", "2024-01-01", "2024-04-01"),
    ]
    assert result["screen_results"]["symbol"].drop_duplicates().tolist() == [
        "AAA", "BBB"
    ]
    assert result["screen_results"].groupby("symbol").size().tolist() == [
        len(FEATURE_COLUMNS_V1),
        len(FEATURE_COLUMNS_V1),
    ]
    assert result["summary"]["feature"].tolist() == FEATURE_COLUMNS_V1

    output = capsys.readouterr().out
    assert "Cross-Asset Feature Summary:" in output
    assert "mean_correlation" in output
    assert "fraction_positive" in output


def test_main_routes_cross_sectional_arguments_without_time_series_screen(
    monkeypatch,
    capsys,
):
    calls = {}
    index = pd.date_range("2024-01-01", periods=75)
    steps = np.arange(len(index), dtype=float)

    def fake_loader(symbol, start, end):
        offset = 5.0 * ["AAA", "BBB", "CCC"].index(symbol)
        return pd.DataFrame({
            "Close": 100.0 + offset + 0.1 * steps + np.sin(steps / 3.0)
        }, index=index)

    expected_ic = pd.DataFrame({
        "date": [pd.Timestamp("2024-03-01")],
        "feature": ["return_1"],
        "ic": [0.25],
        "number_of_symbols": [3],
    })
    expected_summary = pd.DataFrame({
        "feature": ["return_1"],
        "mean_ic": [0.25],
    })

    def fake_ic(panel, min_symbols, method):
        calls["panel"] = panel
        calls["min_symbols"] = min_symbols
        calls["method"] = method
        return expected_ic

    monkeypatch.setattr(runner, "calculate_cross_sectional_ic", fake_ic)
    monkeypatch.setattr(
        runner,
        "summarize_cross_sectional_ic",
        lambda ic_results: expected_summary,
    )
    monkeypatch.setattr(
        runner,
        "run_multi_asset_feature_screen",
        lambda symbol_data: pytest.fail("time-series screen should not run"),
    )

    result = main(
        argv=[
            "--symbols", "AAA", "BBB", "CCC",
            "--start", "2024-01-01",
            "--end", "2024-04-01",
            "--cross-sectional",
            "--method", "spearman",
            "--min-symbols", "3",
        ],
        loader=fake_loader,
    )

    assert result["ic_results"] is expected_ic
    assert result["summary"] is expected_summary
    assert calls["min_symbols"] == 3
    assert calls["method"] == "spearman"
    assert calls["panel"].index.names == ["date", "symbol"]

    output = capsys.readouterr().out
    assert "Exploratory Cross-Sectional IC Summary (spearman):" in output
    assert "Research period: 2024-01-01 to 2024-04-01" in output
    assert "Minimum symbols per date: 3" in output


def test_main_routes_large_research_period_without_manual_symbols(
    monkeypatch,
    capsys,
):
    calls = {}
    panel = pd.DataFrame(
        {**{feature: [1.0] for feature in FEATURE_COLUMNS_V1}, "target": [0.1]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2022-01-03"), "AAA")],
            names=["date", "symbol"],
        ),
    )
    period_summary = pd.DataFrame({
        "feature": ["return_1"],
        "mean_ic": [0.1],
        "icir": [0.5],
    })
    period_results = {
        "research": {
            "panel": panel,
            "ic_results": pd.DataFrame(),
            "summary": period_summary,
        }
    }
    redundancy = pd.DataFrame(
        np.eye(len(FEATURE_COLUMNS_V1)),
        index=FEATURE_COLUMNS_V1,
        columns=FEATURE_COLUMNS_V1,
    )

    def fake_load(symbols, start, end, loader):
        calls["symbols"] = tuple(symbols)
        calls["dates"] = (start, end)
        return {"AAA": pd.DataFrame({"Close": [100.0]})}

    def fake_period_run(
        panel,
        period_names,
        min_symbols,
        method,
        allow_holdout,
    ):
        calls["period_names"] = period_names
        calls["min_symbols"] = min_symbols
        calls["method"] = method
        calls["allow_holdout"] = allow_holdout
        return period_results

    monkeypatch.setattr(runner, "load_symbol_data", fake_load)
    monkeypatch.setattr(
        runner,
        "build_multi_asset_feature_panel",
        lambda symbol_data: panel,
    )
    monkeypatch.setattr(
        runner,
        "run_cross_sectional_ic_by_period",
        fake_period_run,
    )
    monkeypatch.setattr(
        runner,
        "calculate_feature_redundancy",
        lambda research_panel, min_symbols, method: redundancy,
    )

    result = main(
        argv=[
            "--universe", "large",
            "--period", "research",
            "--cross-sectional",
            "--method", "spearman",
            "--min-symbols", "50",
        ],
        loader=lambda symbol, start, end: pytest.fail("loader should be wrapped"),
    )

    assert calls["symbols"] == LARGE_LIQUID_US_EQUITIES_V1
    assert calls["period_names"] == ["research"]
    assert calls["min_symbols"] == 50
    assert calls["method"] == "spearman"
    assert calls["allow_holdout"] is False
    assert result["period_results"] is period_results
    assert result["stability"] is None
    assert result["research_redundancy"] is redundancy

    output = capsys.readouterr().out
    assert "Research Period Cross-Sectional IC Summary (spearman):" in output
    assert "Research-Period Feature Redundancy:" in output
    assert "88 symbols; not point-in-time" in output


def test_holdout_period_requires_explicit_cli_authorization():
    with pytest.raises(ValueError, match="explicit"):
        main(
            argv=["--cross-sectional", "--period", "holdout"],
            loader=lambda symbol, start, end: pytest.fail(
                "holdout guard must run before loading"
            ),
        )
