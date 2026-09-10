from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.run_momentum_research import (
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_SYMBOLS,
    load_adjusted_close_prices,
    main,
)
from src.research.momentum_experiment import MomentumWalkForwardResult


def test_load_adjusted_close_prices_uses_loader_and_common_dates():
    calls = []
    index_by_symbol = {
        "AAA": pd.date_range("2021-01-01", periods=3, freq="D"),
        "BBB": pd.date_range("2021-01-02", periods=3, freq="D"),
    }

    def fake_loader(symbol, start, end):
        calls.append((symbol, start, end))
        index = index_by_symbol[symbol]
        return pd.DataFrame({
            "Open": [90.0, 91.0, 92.0],
            "Close": [100.0, 101.0, 102.0],
        }, index=index)

    result = load_adjusted_close_prices(
        symbols=["AAA", "BBB"],
        start="2021-01-01",
        end="2021-02-01",
        loader=fake_loader,
    )

    assert calls == [
        ("AAA", "2021-01-01", "2021-02-01"),
        ("BBB", "2021-01-01", "2021-02-01"),
    ]
    assert result.columns.tolist() == ["AAA", "BBB"]
    assert result.index.tolist() == list(pd.date_range("2021-01-02", periods=2))
    assert result["AAA"].tolist() == [101.0, 102.0]
    assert result["BBB"].tolist() == [100.0, 101.0]


def test_load_adjusted_close_prices_requires_close_column():
    def fake_loader(symbol, start, end):
        return pd.DataFrame({"Open": [100.0]}, index=[pd.Timestamp("2021-01-01")])

    with pytest.raises(ValueError, match="must contain Close"):
        load_adjusted_close_prices(
            symbols=["AAA"],
            start="2021-01-01",
            end="2021-02-01",
            loader=fake_loader,
        )


def test_main_runs_with_synthetic_loader_and_writes_generated_outputs(
    tmp_path,
    capsys,
):
    calls = []
    index = pd.bdate_range("2021-01-01", periods=260)
    time = np.arange(len(index))
    base_prices = 100.0 + 0.12 * time + 2.0 * np.sin(time / 4.0)

    def fake_loader(symbol, start, end):
        calls.append((symbol, start, end))
        symbol_offset = DEFAULT_SYMBOLS.index(symbol) * 10.0
        return pd.DataFrame(
            {"Close": base_prices + symbol_offset},
            index=index,
        )

    result = main(
        argv=["--output-dir", str(tmp_path)],
        loader=fake_loader,
    )

    assert calls == [
        (symbol, DEFAULT_START, DEFAULT_END)
        for symbol in DEFAULT_SYMBOLS
    ]
    assert result.candidate_results["lookback"].tolist() == [5, 10, 20, 60]
    assert result.candidate_results["target_weight"].tolist() == [0.25] * 4
    assert result.experiment.parameters["allow_short"] is False
    assert result.experiment.config.train_ratio == pytest.approx(0.60)
    assert result.experiment.config.validation_ratio == pytest.approx(0.20)
    assert result.experiment.config.transaction_cost_rate == pytest.approx(0.001)

    candidate_path = tmp_path / "candidate_results.csv"
    summary_path = tmp_path / "summary.csv"
    benchmark_path = tmp_path / "benchmark_comparison.csv"
    assert candidate_path.exists()
    assert summary_path.exists()
    assert benchmark_path.exists()

    candidates = pd.read_csv(candidate_path)
    summary = pd.read_csv(summary_path)
    benchmarks = pd.read_csv(benchmark_path)
    assert len(candidates) == 4
    assert summary["split"].tolist() == ["validation", "test"]
    assert set(benchmarks["strategy"]) == {
        "Momentum",
        "Equal-Weight Constant Allocation",
        "SPY Buy & Hold",
    }

    output = capsys.readouterr().out
    assert "Experiment: momentum-adjusted-close-daily" in output
    assert "Universe: SPY, QQQ, AAPL, MSFT" in output
    assert "CANDIDATE PARAMETER RESULTS" in output
    assert "VALIDATION — model-selection performance" in output
    assert "TEST — final held-out evaluation" in output
    assert "VALIDATION — model-selection comparison" in output
    assert "TEST — final held-out comparison" in output
    assert str(Path(tmp_path) / "candidate_results.csv") in output


def test_main_walk_forward_mode_uses_synthetic_loader_and_writes_outputs(
    tmp_path,
    capsys,
):
    calls = []
    index = pd.bdate_range("2021-01-01", periods=180)
    time = np.arange(len(index))
    base_prices = 100.0 + 0.12 * time + 2.0 * np.sin(time / 4.0)

    def fake_loader(symbol, start, end):
        calls.append((symbol, start, end))
        symbol_offset = DEFAULT_SYMBOLS.index(symbol) * 10.0
        return pd.DataFrame(
            {"Close": base_prices + symbol_offset},
            index=index,
        )

    result = main(
        argv=[
            "--walk-forward",
            "--initial-train-size", "60",
            "--validation-size", "20",
            "--test-size", "20",
            "--lookbacks", "5", "10",
            "--output-dir", str(tmp_path),
        ],
        loader=fake_loader,
    )

    assert isinstance(result, MomentumWalkForwardResult)
    assert calls == [
        (symbol, DEFAULT_START, DEFAULT_END)
        for symbol in DEFAULT_SYMBOLS
    ]
    assert result.momentum.net_returns.index.is_unique

    folds_path = tmp_path / "walk_forward_folds.csv"
    summary_path = tmp_path / "walk_forward_summary.csv"
    benchmark_path = tmp_path / "walk_forward_benchmark_comparison.csv"
    assert folds_path.exists()
    assert summary_path.exists()
    assert benchmark_path.exists()

    folds = pd.read_csv(folds_path)
    benchmarks = pd.read_csv(benchmark_path)
    assert "selected_lookback" in folds.columns
    assert set(benchmarks["strategy"]) == {
        "Momentum",
        "Equal-Weight Constant Allocation",
        "SPY Buy & Hold",
    }

    output = capsys.readouterr().out
    assert "WALK-FORWARD CONFIGURATION" in output
    assert "FOLD RESULTS" in output
    assert "PARAMETER STABILITY" in output
    assert "STITCHED OOS MOMENTUM PERFORMANCE" in output
    assert "STITCHED OOS BENCHMARK COMPARISON" in output
