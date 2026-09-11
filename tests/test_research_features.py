import numpy as np
import pandas as pd
import pytest

from src.research.features import (
    FEATURE_COLUMNS_V1,
    build_multi_asset_feature_panel,
    build_price_features,
    calculate_cross_sectional_ic,
    calculate_feature_redundancy,
    run_multi_asset_feature_screen,
    run_cross_sectional_ic_by_period,
    summarize_cross_sectional_ic,
    summarize_feature_stability,
    summarize_feature_screen,
)


def price_data(length=90, offset=0.0):
    index = pd.date_range("2024-01-01", periods=length)
    steps = np.arange(length, dtype=float)
    close = 100.0 + offset + 0.2 * steps + 2.0 * np.sin(steps / 4.0)
    return pd.DataFrame({"Close": close}, index=index)


def test_build_price_features_has_exact_names_and_alignment():
    data = price_data()
    result = build_price_features(data)
    close = data["Close"]
    returns = close.pct_change()
    row_index = result.index[2]

    assert FEATURE_COLUMNS_V1 == [
        "return_1",
        "return_2",
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "volatility_5",
        "volatility_20",
        "trend_20",
    ]
    assert "target" not in FEATURE_COLUMNS_V1
    assert result.columns.tolist() == [*FEATURE_COLUMNS_V1, "target"]
    assert not result.isna().any().any()
    assert result.index.is_monotonic_increasing
    assert result.index.equals(data.index[60:-1])

    location = data.index.get_loc(row_index)
    assert result.loc[row_index, "return_1"] == pytest.approx(
        returns.iloc[location]
    )
    assert result.loc[row_index, "return_2"] == pytest.approx(
        returns.iloc[location - 1]
    )
    assert result.loc[row_index, "momentum_5"] == pytest.approx(
        close.iloc[location] / close.iloc[location - 5] - 1
    )
    assert result.loc[row_index, "momentum_20"] == pytest.approx(
        close.iloc[location] / close.iloc[location - 20] - 1
    )
    assert result.loc[row_index, "momentum_60"] == pytest.approx(
        close.iloc[location] / close.iloc[location - 60] - 1
    )
    assert result.loc[row_index, "volatility_5"] == pytest.approx(
        returns.iloc[location - 4:location + 1].std()
    )
    assert result.loc[row_index, "volatility_20"] == pytest.approx(
        returns.iloc[location - 19:location + 1].std()
    )
    assert result.loc[row_index, "trend_20"] == pytest.approx(
        close.iloc[location] / close.iloc[location - 19:location + 1].mean() - 1
    )
    assert result.loc[row_index, "target"] == pytest.approx(
        returns.iloc[location + 1]
    )


def test_future_price_mutation_does_not_change_earlier_features():
    original_data = price_data()
    changed_data = original_data.copy()
    cutoff = original_data.index[70]
    changed_data.loc[changed_data.index > cutoff, "Close"] *= 3.0

    original = build_price_features(original_data)
    changed = build_price_features(changed_data)

    pd.testing.assert_frame_equal(
        original.loc[:cutoff, FEATURE_COLUMNS_V1],
        changed.loc[:cutoff, FEATURE_COLUMNS_V1],
    )


def test_build_price_features_rejects_duplicate_or_unsorted_indices():
    duplicated = price_data()
    duplicated.index = duplicated.index.where(
        np.arange(len(duplicated)) != 2,
        duplicated.index[1],
    )
    with pytest.raises(ValueError, match="unique"):
        build_price_features(duplicated)

    unsorted = price_data().iloc[::-1]
    with pytest.raises(ValueError, match="chronological"):
        build_price_features(unsorted)


def test_multi_asset_screen_returns_tidy_symbol_feature_results():
    symbol_data = {
        "AAA": price_data(offset=0.0),
        "BBB": price_data(offset=10.0),
    }
    result = run_multi_asset_feature_screen(symbol_data)

    assert result.columns.tolist() == [
        "symbol",
        "feature",
        "correlation",
        "sample_size",
    ]
    assert len(result) == len(symbol_data) * len(FEATURE_COLUMNS_V1)
    assert result.groupby("symbol")["feature"].apply(list).tolist() == [
        FEATURE_COLUMNS_V1,
        FEATURE_COLUMNS_V1,
    ]

    aaa_dataset = build_price_features(symbol_data["AAA"])
    aaa_return = result.loc[
        (result["symbol"] == "AAA") & (result["feature"] == "return_1")
    ].iloc[0]
    assert aaa_return["sample_size"] == len(aaa_dataset)
    assert aaa_return["correlation"] == pytest.approx(
        aaa_dataset["return_1"].corr(aaa_dataset["target"])
    )


def test_summarize_feature_screen_calculates_cross_asset_statistics():
    rows = []
    correlations = {
        "AAA": 0.3,
        "BBB": -0.1,
        "CCC": 0.2,
    }
    for feature in FEATURE_COLUMNS_V1:
        for symbol, correlation in correlations.items():
            rows.append({
                "symbol": symbol,
                "feature": feature,
                "correlation": correlation,
                "sample_size": 100,
            })

    summary = summarize_feature_screen(pd.DataFrame(rows))

    assert summary["feature"].tolist() == FEATURE_COLUMNS_V1
    assert summary["number_of_symbols"].tolist() == [3] * len(FEATURE_COLUMNS_V1)
    first = summary.iloc[0]
    expected = pd.Series(list(correlations.values()))
    assert first["mean_correlation"] == pytest.approx(expected.mean())
    assert first["median_correlation"] == pytest.approx(expected.median())
    assert first["std_correlation"] == pytest.approx(expected.std())
    assert first["fraction_positive"] == pytest.approx(2 / 3)


def test_build_multi_asset_panel_preserves_exact_date_symbol_rows():
    aaa = price_data()
    bbb = price_data(offset=10.0).drop(index=price_data().index[65])

    panel = build_multi_asset_feature_panel({"AAA": aaa, "BBB": bbb})

    assert panel.index.names == ["date", "symbol"]
    assert panel.index.is_monotonic_increasing
    assert panel.index.is_unique
    assert panel.columns.tolist() == [*FEATURE_COLUMNS_V1, "target"]
    pd.testing.assert_frame_equal(
        panel.xs("AAA", level="symbol"),
        build_price_features(aaa).rename_axis("date"),
        check_freq=False,
    )
    assert (aaa.index[65], "BBB") not in panel.index


def cross_sectional_panel():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=["date", "symbol"],
    )
    feature_values = np.tile([1.0, 2.0, 3.0, 4.0], len(dates))
    targets = np.concatenate([
        [1.0, 2.0, 4.0, 8.0],
        [8.0, 4.0, 2.0, 1.0],
    ])
    values = {
        feature: feature_values.copy()
        for feature in FEATURE_COLUMNS_V1
    }
    values["target"] = targets
    return pd.DataFrame(values, index=index)


def test_cross_sectional_ic_uses_symbols_from_the_same_date():
    panel = cross_sectional_panel()
    result = calculate_cross_sectional_ic(
        panel,
        min_symbols=4,
        method="pearson",
    )
    return_1 = result.loc[result["feature"] == "return_1"]

    expected_first = pd.Series([1.0, 2.0, 3.0, 4.0]).corr(
        pd.Series([1.0, 2.0, 4.0, 8.0])
    )
    expected_second = pd.Series([1.0, 2.0, 3.0, 4.0]).corr(
        pd.Series([8.0, 4.0, 2.0, 1.0])
    )
    assert return_1["date"].tolist() == list(
        panel.index.get_level_values("date").unique()
    )
    assert return_1["ic"].tolist() == pytest.approx([
        expected_first,
        expected_second,
    ])
    assert return_1["number_of_symbols"].tolist() == [4, 4]


def test_cross_sectional_ic_supports_spearman_and_minimum_symbols():
    panel = cross_sectional_panel()
    spearman = calculate_cross_sectional_ic(
        panel,
        min_symbols=4,
        method="spearman",
    )
    return_1 = spearman.loc[spearman["feature"] == "return_1", "ic"]
    assert return_1.tolist() == pytest.approx([1.0, -1.0])

    limited = panel.copy()
    first_date = limited.index.get_level_values("date").min()
    limited.loc[(first_date, ["CCC", "DDD"]), "target"] = np.nan
    insufficient = calculate_cross_sectional_ic(
        limited,
        min_symbols=3,
        method="pearson",
    )
    first_return = insufficient.loc[
        (insufficient["date"] == first_date)
        & (insufficient["feature"] == "return_1")
    ].iloc[0]
    assert first_return["number_of_symbols"] == 2
    assert np.isnan(first_return["ic"])


def test_cross_sectional_ic_rejects_duplicate_date_symbol_rows():
    panel = cross_sectional_panel()
    duplicated = pd.concat([panel, panel.iloc[[0]]]).sort_index()

    with pytest.raises(ValueError, match="unique"):
        calculate_cross_sectional_ic(duplicated, min_symbols=2)


def test_summarize_cross_sectional_ic_calculates_metrics_and_safe_icir():
    rows = []
    dates = pd.date_range("2024-01-01", periods=3)
    for feature in FEATURE_COLUMNS_V1:
        ic_values = (
            [0.2, 0.2, 0.2]
            if feature == "return_1"
            else [0.1, -0.2, 0.3]
        )
        for date, ic, number_of_symbols in zip(dates, ic_values, [5, 6, 7]):
            rows.append({
                "date": date,
                "feature": feature,
                "ic": ic,
                "number_of_symbols": number_of_symbols,
            })

    summary = summarize_cross_sectional_ic(pd.DataFrame(rows))

    assert summary["feature"].tolist() == FEATURE_COLUMNS_V1
    constant = summary.loc[summary["feature"] == "return_1"].iloc[0]
    assert np.isnan(constant["icir"])

    momentum = summary.loc[summary["feature"] == "momentum_5"].iloc[0]
    expected = pd.Series([0.1, -0.2, 0.3])
    assert momentum["mean_ic"] == pytest.approx(expected.mean())
    assert momentum["median_ic"] == pytest.approx(expected.median())
    assert momentum["std_ic"] == pytest.approx(expected.std())
    assert momentum["icir"] == pytest.approx(expected.mean() / expected.std())
    assert momentum["fraction_positive_ic"] == pytest.approx(2 / 3)
    assert momentum["number_of_dates"] == 3
    assert momentum["mean_number_of_symbols"] == pytest.approx(6.0)


def test_future_mutation_does_not_change_earlier_panel_or_ic():
    original_data = {
        f"S{number}": price_data(offset=number * 5.0)
        for number in range(5)
    }
    changed_data = {
        symbol: data.copy()
        for symbol, data in original_data.items()
    }
    cutoff = price_data().index[70]
    last_unchanged_date = price_data().index[71]
    for number, data in enumerate(changed_data.values(), start=2):
        data.loc[data.index > last_unchanged_date, "Close"] *= number

    original_panel = build_multi_asset_feature_panel(original_data)
    changed_panel = build_multi_asset_feature_panel(changed_data)
    original_mask = original_panel.index.get_level_values("date") <= cutoff
    changed_mask = changed_panel.index.get_level_values("date") <= cutoff
    pd.testing.assert_frame_equal(
        original_panel.loc[original_mask],
        changed_panel.loc[changed_mask],
    )

    original_ic = calculate_cross_sectional_ic(original_panel, min_symbols=5)
    changed_ic = calculate_cross_sectional_ic(changed_panel, min_symbols=5)
    pd.testing.assert_frame_equal(
        original_ic.loc[original_ic["date"] <= cutoff].reset_index(drop=True),
        changed_ic.loc[changed_ic["date"] <= cutoff].reset_index(drop=True),
    )


def period_panel():
    dates = pd.to_datetime([
        "2022-01-03",
        "2023-12-29",
        "2024-01-02",
        "2025-12-30",
        "2026-01-02",
    ])
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=["date", "symbol"],
    )
    base = np.tile(np.arange(1.0, 6.0), len(dates))
    data = {
        feature: base * (feature_number + 1)
        for feature_number, feature in enumerate(FEATURE_COLUMNS_V1)
    }
    data["target"] = np.tile([1.0, 4.0, 2.0, 5.0, 3.0], len(dates))
    return pd.DataFrame(data, index=index)


def test_cross_sectional_ic_summaries_are_separate_by_period():
    result = run_cross_sectional_ic_by_period(
        period_panel(),
        period_names=["research", "validation"],
        min_symbols=5,
        method="spearman",
    )

    assert list(result) == ["research", "validation"]
    research_dates = result["research"]["ic_results"]["date"].unique()
    validation_dates = result["validation"]["ic_results"]["date"].unique()
    assert list(research_dates) == list(pd.to_datetime([
        "2022-01-03", "2023-12-29"
    ]))
    assert list(validation_dates) == list(pd.to_datetime([
        "2024-01-02", "2025-12-30"
    ]))
    assert result["research"]["summary"]["number_of_dates"].tolist() == [2] * 8
    assert result["validation"]["summary"]["number_of_dates"].tolist() == [2] * 8


def test_feature_stability_reports_mean_ic_sign_agreement():
    research = pd.DataFrame({
        "feature": FEATURE_COLUMNS_V1,
        "mean_ic": [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4],
        "icir": np.arange(8, dtype=float),
    })
    validation = pd.DataFrame({
        "feature": FEATURE_COLUMNS_V1,
        "mean_ic": [0.2, -0.2, -0.1, 0.1, 0.2, -0.2, 0.3, -0.3],
        "icir": np.arange(8, dtype=float) + 10,
    })

    result = summarize_feature_stability(research, validation)

    assert result.columns.tolist() == [
        "feature",
        "research_mean_ic",
        "research_icir",
        "validation_mean_ic",
        "validation_icir",
        "sign_consistent",
    ]
    assert result["sign_consistent"].tolist() == [
        True, True, False, False, True, True, True, True
    ]


def test_feature_redundancy_averages_same_date_feature_correlations_only():
    panel = period_panel()
    research_panel = panel.loc[
        panel.index.get_level_values("date") <= pd.Timestamp("2023-12-31")
    ]

    result = calculate_feature_redundancy(
        research_panel,
        min_symbols=5,
        method="spearman",
    )
    changed_target = research_panel.copy()
    changed_target["target"] = np.arange(len(changed_target), dtype=float) ** 2
    target_changed_result = calculate_feature_redundancy(
        changed_target,
        min_symbols=5,
        method="spearman",
    )

    assert result.index.tolist() == FEATURE_COLUMNS_V1
    assert result.columns.tolist() == FEATURE_COLUMNS_V1
    assert "target" not in result.index
    assert "target" not in result.columns
    assert np.diag(result).tolist() == pytest.approx([1.0] * 8)
    pd.testing.assert_frame_equal(result, target_changed_result)
