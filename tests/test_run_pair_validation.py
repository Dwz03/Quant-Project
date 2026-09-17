import numpy as np
import pandas as pd

from scripts import run_pair_validation as runner


def test_runner_loads_frozen_cohort_and_writes_validation_reports(monkeypatch):
    calls = []
    written_paths = []
    dates = pd.date_range("2024-01-02", periods=90, freq="B")
    all_pairs = [
        (left, right)
        for position, left in enumerate(runner.UNIVERSES[runner.UNIVERSE_NAME])
        for right in runner.UNIVERSES[runner.UNIVERSE_NAME][position + 1:]
    ]
    frozen_pairs = all_pairs[:runner.EXPECTED_FROZEN_CANDIDATES]
    research = pd.DataFrame({
        "symbol_1": [pair[0] for pair in frozen_pairs],
        "symbol_2": [pair[1] for pair in frozen_pairs],
        "cointegration_p_value": [0.01] * len(frozen_pairs),
        "cointegration_q_value": [0.50] * len(frozen_pairs),
    })

    class OutputSink:
        def mkdir(self, **kwargs):
            return None

        def __truediv__(self, name):
            return name

        def __str__(self):
            return "synthetic-output"

    def synthetic_loader(symbol, start, end):
        calls.append((symbol, start, end))
        return pd.DataFrame(
            {"Close": 100.0 + np.arange(len(dates))}, index=dates
        )

    def synthetic_validation(symbol_data, research_diagnostics, **kwargs):
        assert research_diagnostics is research
        diagnostics = pd.DataFrame({
            "symbol_1": ["AAPL"],
            "symbol_2": ["MSFT"],
            "research_cointegration_p_value": [0.01],
            "research_cointegration_q_value": [0.50],
            "validation_aligned_observations": [90],
            "validation_alpha": [1.0],
            "validation_beta": [1.0],
            "validation_cointegration_test_statistic": [-3.0],
            "validation_cointegration_p_value": [0.10],
            "validation_adf_statistic": [-2.5],
            "validation_adf_p_value": [0.10],
            "validation_half_life": [5.0],
            "validation_mean_beta": [1.0],
            "validation_median_beta": [1.0],
            "validation_std_beta": [0.1],
            "validation_min_beta": [0.8],
            "validation_max_beta": [1.2],
            "validation_valid_estimates": [31],
            "validation_cointegration_q_value": [0.10],
            "validation_fdr_significant_05": [False],
        })
        return {
            "frozen_candidates": research.rename(columns={
                "cointegration_p_value": "research_cointegration_p_value",
                "cointegration_q_value": "research_cointegration_q_value",
            }),
            "diagnostics": diagnostics,
            "advancing_pairs": diagnostics.iloc[:0].copy(),
            "skipped": pd.DataFrame(
                columns=["symbol_1", "symbol_2", "aligned_observations", "reason"]
            ),
            "validation_prices": pd.DataFrame(
                {"AAPL": np.arange(len(dates), dtype=float)}, index=dates
            ),
            "rolling_window": kwargs["rolling_window"],
            "fdr_level": runner.VALIDATION_FDR_LEVEL,
        }

    monkeypatch.setattr(pd, "read_csv", lambda path: research)
    monkeypatch.setattr(runner, "validate_frozen_pair_candidates", synthetic_validation)
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda self, path, **kwargs: written_paths.append(path),
    )

    result = runner.main(
        loader=synthetic_loader,
        research_diagnostics_path="synthetic.csv",
        output_dir=OutputSink(),
    )

    assert len(calls) == 88
    assert {(start, end) for _, start, end in calls} == {
        (runner.CACHE_START, runner.CACHE_END)
    }
    assert result["advancing_pairs"].empty
    assert written_paths == [
        "validation_diagnostics.csv",
        "validation_summary.csv",
        "advancing_pairs.csv",
        "skipped_candidates.csv",
    ]
