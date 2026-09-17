import numpy as np
import pandas as pd
import pytest

import src.research.pairs as pairs_module
from src.research.pairs import (
    benjamini_hochberg,
    build_validation_price_matrix,
    freeze_research_candidates,
    select_validation_advancing_pairs,
    validate_frozen_pair_candidates,
)


def _research_diagnostics():
    return pd.DataFrame({
        "symbol_1": ["AAA", "AAA", "BBB"],
        "symbol_2": ["BBB", "CCC", "CCC"],
        "cointegration_p_value": [0.01, 0.049, 0.20],
        "cointegration_q_value": [0.03, 0.08, 0.30],
    })


def _symbol_data(include_research=True):
    rng = np.random.default_rng(91)
    validation_dates = pd.date_range("2024-01-02", periods=140, freq="B")
    base = 100.0 + np.cumsum(rng.normal(scale=0.5, size=len(validation_dates)))
    data = {
        "AAA": pd.DataFrame({"Close": base}, index=validation_dates),
        "BBB": pd.DataFrame(
            {"Close": 20.0 + 1.2 * base + rng.normal(scale=0.4, size=len(base))},
            index=validation_dates,
        ),
        "CCC": pd.DataFrame(
            {"Close": 80.0 + np.cumsum(rng.normal(scale=0.7, size=len(base)))},
            index=validation_dates,
        ),
    }
    if include_research:
        research_dates = pd.date_range("2021-01-04", periods=20, freq="B")
        for symbol, frame in data.items():
            research = pd.DataFrame(
                {"Close": 1000.0 + np.arange(len(research_dates))},
                index=research_dates,
            )
            data[symbol] = pd.concat([research, frame])
    return data


def test_frozen_cohort_uses_only_research_raw_p_below_05():
    frozen = freeze_research_candidates(_research_diagnostics())

    assert list(frozen[["symbol_1", "symbol_2"]].itertuples(index=False, name=None)) == [
        ("AAA", "BBB"),
        ("AAA", "CCC"),
    ]
    assert frozen["research_cointegration_p_value"].tolist() == [0.01, 0.049]


def test_validation_values_cannot_change_frozen_research_cohort():
    research = _research_diagnostics()
    first = freeze_research_candidates(research)
    validation_values = pd.Series([0.99, 0.001, 0.02])
    research_with_validation = research.assign(validation_p_value=validation_values)

    second = freeze_research_candidates(research_with_validation)
    pd.testing.assert_frame_equal(first, second)


def test_validation_matrix_excludes_research_and_rejects_holdout():
    data = _symbol_data()
    validation = build_validation_price_matrix(data)

    assert validation.index.min() >= pd.Timestamp("2024-01-01")
    assert validation.index.max() <= pd.Timestamp("2025-12-31")

    holdout = pd.DataFrame({"Close": [100.0]}, index=[pd.Timestamp("2026-01-02")])
    data["AAA"] = pd.concat([data["AAA"], holdout])
    with pytest.raises(ValueError, match="holdout"):
        build_validation_price_matrix(data)


def test_research_values_cannot_enter_validation_diagnostics():
    first_data = _symbol_data()
    changed_data = {symbol: frame.copy() for symbol, frame in first_data.items()}
    for frame in changed_data.values():
        frame.loc[frame.index < "2024-01-01", "Close"] *= 100.0

    first = validate_frozen_pair_candidates(
        first_data, _research_diagnostics(), rolling_window=20
    )
    changed = validate_frozen_pair_candidates(
        changed_data, _research_diagnostics(), rolling_window=20
    )

    pd.testing.assert_frame_equal(first["diagnostics"], changed["diagnostics"])


def test_only_frozen_candidates_are_evaluated_and_bh_uses_that_family():
    result = validate_frozen_pair_candidates(
        _symbol_data(), _research_diagnostics(), rolling_window=20
    )
    tested = list(result["diagnostics"][["symbol_1", "symbol_2"]].itertuples(
        index=False, name=None
    ))

    assert tested == [("AAA", "BBB"), ("AAA", "CCC")]
    expected = benjamini_hochberg(
        result["diagnostics"]["validation_cointegration_p_value"]
    )
    assert result["diagnostics"]["validation_cointegration_q_value"].to_numpy() == pytest.approx(
        expected["q_value"].to_numpy()
    )
    assert result["diagnostics"]["validation_cointegration_q_value"].between(0, 1).all()


def test_missing_validation_dates_are_aligned_without_forward_fill():
    data = _symbol_data(include_research=False)
    missing_date = data["BBB"].index[10]
    data["BBB"] = data["BBB"].drop(index=missing_date)

    result = validate_frozen_pair_candidates(
        data, _research_diagnostics(), rolling_window=20
    )
    pair = result["diagnostics"].set_index(["symbol_1", "symbol_2"])

    assert pd.isna(result["validation_prices"].loc[missing_date, "BBB"])
    assert pair.loc[("AAA", "BBB"), "validation_aligned_observations"] == 139
    assert pair.loc[("AAA", "CCC"), "validation_aligned_observations"] == 140


def test_zero_survivors_are_not_rescued(monkeypatch):
    def no_rejections(p_values, alpha=0.05):
        raw = pd.Series(p_values, copy=True, dtype=float)
        return pd.DataFrame({
            "raw_p_value": raw,
            "q_value": pd.Series(0.5, index=raw.index),
            "fdr_significant": pd.Series(False, index=raw.index),
        })

    monkeypatch.setattr(pairs_module, "benjamini_hochberg", no_rejections)
    result = validate_frozen_pair_candidates(
        _symbol_data(), _research_diagnostics(), rolling_window=20
    )

    assert result["advancing_pairs"].empty
    assert len(result["diagnostics"]) == 2


def test_advancement_uses_only_validation_q_value():
    diagnostics = pd.DataFrame({
        "symbol_1": ["AAA", "BBB"],
        "symbol_2": ["CCC", "CCC"],
        "validation_cointegration_q_value": [0.049, 0.05],
        "validation_adf_p_value": [0.99, 0.001],
        "validation_half_life": [np.nan, 2.0],
        "validation_std_beta": [999.0, 0.01],
    })

    advancing = select_validation_advancing_pairs(diagnostics)

    assert list(advancing[["symbol_1", "symbol_2"]].itertuples(index=False, name=None)) == [
        ("AAA", "CCC")
    ]


def test_validation_results_are_reproducible():
    first = validate_frozen_pair_candidates(
        _symbol_data(), _research_diagnostics(), rolling_window=20
    )
    second = validate_frozen_pair_candidates(
        _symbol_data(), _research_diagnostics(), rolling_window=20
    )

    pd.testing.assert_frame_equal(first["diagnostics"], second["diagnostics"])
    pd.testing.assert_frame_equal(first["advancing_pairs"], second["advancing_pairs"])
