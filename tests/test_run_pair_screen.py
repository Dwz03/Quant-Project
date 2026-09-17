import numpy as np
import pandas as pd

from scripts import run_pair_screen as runner


def test_runner_uses_cached_pre_holdout_range_and_writes_reports(
    monkeypatch
):
    calls = []
    written_paths = []
    dates = pd.date_range("2021-01-01", periods=90, freq="B")

    class OutputSink:
        def mkdir(self, **kwargs):
            return None

        def __truediv__(self, name):
            return name

        def __str__(self):
            return "synthetic-output"

    def synthetic_loader(symbol, start, end):
        calls.append((symbol, start, end))
        symbol_number = runner.UNIVERSES[runner.UNIVERSE_NAME].index(symbol) + 1
        return pd.DataFrame(
            {"Close": 100.0 + symbol_number + np.arange(len(dates)) * (1 + symbol_number / 100)},
            index=dates,
        )

    def synthetic_screen(symbol_data, **kwargs):
        prices = pd.concat(
            {symbol: frame["Close"] for symbol, frame in symbol_data.items()},
            axis=1,
        )
        diagnostics = pd.DataFrame({
            "symbol_1": ["AAPL"],
            "symbol_2": ["MSFT"],
            "aligned_observations": [90],
            "alpha": [1.0],
            "beta": [1.0],
            "cointegration_test_statistic": [-3.0],
            "cointegration_p_value": [0.02],
            "adf_statistic": [-3.1],
            "adf_p_value": [0.01],
            "half_life": [4.0],
            "mean_beta": [1.0],
            "median_beta": [1.0],
            "std_beta": [0.1],
            "min_beta": [0.8],
            "max_beta": [1.2],
            "valid_estimates": [31],
            "cointegration_q_value": [0.04],
            "fdr_significant_05": [True],
        })
        return {
            "diagnostics": diagnostics,
            "skipped": pd.DataFrame(
                columns=["symbol_1", "symbol_2", "aligned_observations", "reason"]
            ),
            "total_pairs": 88 * 87 // 2,
            "rolling_window": kwargs["rolling_window"],
            "fdr_level": kwargs["fdr_level"],
            "research_prices": prices,
        }

    monkeypatch.setattr(runner, "screen_research_pairs", synthetic_screen)
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda self, path, **kwargs: written_paths.append(path),
    )

    result = runner.main(loader=synthetic_loader, output_dir=OutputSink())

    assert len(calls) == 88
    assert {(start, end) for _, start, end in calls} == {
        (runner.CACHE_START, runner.CACHE_END)
    }
    assert result["total_pairs"] == 88 * 87 // 2
    assert result["research_prices"].index.max() <= pd.Timestamp("2023-12-31")
    assert written_paths == [
        "pair_diagnostics.csv",
        "screening_summary.csv",
        "skipped_pairs.csv",
    ]
