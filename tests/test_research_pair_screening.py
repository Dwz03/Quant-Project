import numpy as np
import pandas as pd
import pytest

from src.research.pairs import (
    benjamini_hochberg,
    build_research_price_matrix,
    generate_deterministic_pairs,
    screen_pair_diagnostics,
    screen_research_pairs,
)


def _symbol_data(include_validation=False):
    dates = pd.date_range("2021-01-01", periods=140, freq="B")
    rng = np.random.default_rng(23)
    base = 100.0 + np.cumsum(rng.normal(scale=0.5, size=len(dates)))
    result = {
        "CCC": pd.DataFrame({"Close": 30.0 + 0.5 * base + rng.normal(scale=0.2, size=len(dates))}, index=dates),
        "AAA": pd.DataFrame({"Close": base}, index=dates),
        "BBB": pd.DataFrame({"Close": 10.0 + 1.3 * base + rng.normal(scale=0.2, size=len(dates))}, index=dates),
    }
    if include_validation:
        validation_dates = pd.date_range("2024-01-02", periods=5, freq="B")
        for offset, frame in enumerate(result.values()):
            validation = pd.DataFrame(
                {"Close": 500.0 + offset + np.arange(len(validation_dates))},
                index=validation_dates,
            )
            result[list(result)[offset]] = pd.concat([frame, validation])
    return result


def test_unique_unordered_pairs_have_combinatorial_count_and_no_reverse_pairs():
    pairs = generate_deterministic_pairs(["D", "B", "A", "C"])

    assert len(pairs) == 4 * 3 // 2
    assert len({frozenset(pair) for pair in pairs}) == len(pairs)
    assert all((right, left) not in pairs for left, right in pairs)


def test_pair_orientation_is_deterministic():
    expected = [("A", "B"), ("A", "C"), ("B", "C")]

    assert generate_deterministic_pairs(["C", "A", "B"]) == expected
    assert generate_deterministic_pairs(["B", "C", "A"]) == expected


def test_research_matrix_excludes_validation_and_fails_closed_on_holdout():
    symbol_data = _symbol_data(include_validation=True)
    research = build_research_price_matrix(symbol_data)

    assert research.index.max() <= pd.Timestamp("2023-12-31")
    assert not (research.index >= pd.Timestamp("2024-01-01")).any()

    holdout = pd.DataFrame(
        {"Close": [100.0]}, index=[pd.Timestamp("2026-01-02")]
    )
    symbol_data["AAA"] = pd.concat([symbol_data["AAA"], holdout])
    with pytest.raises(ValueError, match="holdout"):
        build_research_price_matrix(symbol_data)


def test_missing_dates_align_without_forward_fill_and_counts_are_correct():
    dates = pd.date_range("2022-01-03", periods=80, freq="B")
    symbol_data = {
        "AAA": pd.Series(100.0 + np.arange(80), index=dates),
        "BBB": pd.Series(200.0 + np.arange(79), index=dates.delete(10)),
    }

    prices = build_research_price_matrix(symbol_data)
    assert pd.isna(prices.loc[dates[10], "BBB"])

    result = screen_pair_diagnostics(prices, rolling_window=20)
    assert result["diagnostics"].loc[0, "aligned_observations"] == 79


def test_bh_adjustment_is_bounded_monotonic_and_preserves_raw_values():
    raw = pd.Series([0.04, 0.001, 0.03, 0.20, np.nan], index=list("abcde"))

    result = benjamini_hochberg(raw)

    pd.testing.assert_series_equal(result["raw_p_value"], raw, check_names=False)
    assert result["q_value"].dropna().between(0.0, 1.0).all()
    ordered = result.dropna().sort_values("raw_p_value")
    assert ordered["q_value"].is_monotonic_increasing
    assert pd.isna(result.loc["e", "q_value"])


def test_screen_preserves_raw_and_adjusted_p_values_and_is_reproducible():
    symbol_data = _symbol_data()

    first = screen_research_pairs(symbol_data, rolling_window=20)
    second = screen_research_pairs(symbol_data, rolling_window=20)

    assert {"cointegration_p_value", "cointegration_q_value"}.issubset(
        first["diagnostics"].columns
    )
    pd.testing.assert_frame_equal(first["diagnostics"], second["diagnostics"])
    pd.testing.assert_frame_equal(first["skipped"], second["skipped"])


def test_validation_changes_cannot_alter_fixed_research_diagnostics():
    original = _symbol_data(include_validation=True)
    changed = {symbol: frame.copy() for symbol, frame in original.items()}
    for frame in changed.values():
        frame.loc[frame.index >= "2024-01-01", "Close"] *= 100.0

    original_result = screen_research_pairs(original, rolling_window=20)
    changed_result = screen_research_pairs(changed, rolling_window=20)

    pd.testing.assert_frame_equal(
        original_result["diagnostics"], changed_result["diagnostics"]
    )


def test_non_mean_reverting_half_life_nan_is_preserved():
    dates = pd.date_range("2022-01-03", periods=120, freq="B")
    rng = np.random.default_rng(1)
    x = 100.0 + np.cumsum(rng.normal(size=len(dates)))
    y = 100.0 + np.exp(np.arange(len(dates)) / 20.0)
    prices = pd.DataFrame(
        {"AAA": y, "BBB": x},
        index=dates,
    )

    result = screen_pair_diagnostics(prices, rolling_window=20)

    assert len(result["diagnostics"]) == 1
    assert np.isnan(result["diagnostics"].loc[0, "half_life"])
