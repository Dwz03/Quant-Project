import numpy as np
import pandas as pd
import pytest

from src.research.features import (
    FEATURE_COLUMNS_V1,
    MOMENTUM_FEATURE_COLUMNS,
    build_multi_asset_feature_panel,
    build_multi_asset_momentum_panel,
    build_momentum_features,
    build_price_features,
    calculate_cross_sectional_ic,
    calculate_feature_redundancy,
    run_multi_asset_feature_screen,
    run_cross_sectional_ic_by_period,
    run_momentum_feature_screen,
    summarize_cross_sectional_ic,
    summarize_feature_stability,
    summarize_feature_screen,
)


def price_data(length=90, offset=0.0):
    index = pd.date_range("2024-01-01", periods=length)
    steps = np.arange(length, dtype=float)
    close = 100.0 + offset + 0.2 * steps + 2.0 * np.sin(steps / 4.0)
    return pd.DataFrame({"Close": close}, index=index)


def test_build_momentum_features_calculates_exact_multi_horizon_returns():
    dates = pd.date_range("2020-01-01", periods=253)
    close = pd.Series(np.arange(100.0, 353.0), index=dates)

    result = build_momentum_features(pd.DataFrame({"Close": close}))

    assert result.columns.tolist() == MOMENTUM_FEATURE_COLUMNS
    for horizon in (5, 20, 60, 120, 252):
        assert result.loc[dates[-1], f"momentum_{horizon}"] == pytest.approx(
            close.iloc[-1] / close.iloc[-1 - horizon] - 1
        )


def test_build_momentum_features_keeps_insufficient_history_as_nan():
    data = pd.DataFrame(
        {"Close": [100.0, 102.0, 104.0, 106.0, 110.0, 110.0]},
        index=pd.date_range("2024-01-01", periods=6),
    )

    result = build_momentum_features(data)

    assert result["momentum_5"].iloc[:5].isna().all()
    assert result["momentum_5"].iloc[5] == pytest.approx(0.10)
    longer_horizons = [
        "momentum_20", "momentum_60", "momentum_120", "momentum_252"
    ]
    assert result[longer_horizons].isna().all().all()


def test_panel_momentum_is_calculated_independently_by_symbol():
    dates = pd.date_range("2024-01-01", periods=6)
    index = pd.MultiIndex.from_product(
        [dates, ["AAA", "BBB"]],
        names=["date", "symbol"],
    )
    panel = pd.DataFrame({
        "Close": np.column_stack([
            [100.0, 101.0, 102.0, 103.0, 104.0, 110.0],
            [200.0, 190.0, 180.0, 170.0, 160.0, 150.0],
        ]).reshape(-1),
    }, index=index)

    result = build_momentum_features(panel)

    assert result.loc[(dates[-1], "AAA"), "momentum_5"] == pytest.approx(0.10)
    assert result.loc[(dates[-1], "BBB"), "momentum_5"] == pytest.approx(-0.25)
    assert result.groupby(level="symbol")["momentum_5"].apply(
        lambda values: values.iloc[:5].isna().all()
    ).all()


def test_future_prices_do_not_affect_past_momentum_features():
    original = price_data(length=300)
    changed = original.copy()
    cutoff = original.index[270]
    changed.loc[changed.index > cutoff, "Close"] *= 100.0

    original_features = build_momentum_features(original)
    changed_features = build_momentum_features(changed)

    pd.testing.assert_frame_equal(
        original_features.loc[:cutoff],
        changed_features.loc[:cutoff],
    )


def test_build_momentum_features_does_not_mutate_input():
    data = price_data(length=300)
    original = data.copy(deep=True)

    build_momentum_features(data)

    pd.testing.assert_frame_equal(data, original)


def test_existing_momentum_60_definition_remains_compatible():
    data = price_data(length=90)

    momentum = build_momentum_features(data)["momentum_60"]
    legacy = data["Close"] / data["Close"].shift(60) - 1

    pd.testing.assert_series_equal(momentum, legacy, check_names=False)


def momentum_screen_panel():
    dates = pd.to_datetime([
        "2021-01-04", "2021-07-01", "2022-07-01", "2023-12-29",
        "2024-01-02", "2024-07-01", "2025-07-01", "2025-12-30",
    ])
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=["date", "symbol"],
    )
    scores = np.tile([1.0, 2.0, 3.0, 4.0], len(dates))
    positive = [1.0, 2.0, 3.0, 4.0]
    negative = positive[::-1]
    targets = np.concatenate([
        positive, positive, negative, negative,
        positive, positive, negative, negative,
    ])
    values = {
        feature: scores.copy()
        for feature in MOMENTUM_FEATURE_COLUMNS
    }
    values["target"] = targets
    return pd.DataFrame(values, index=index)


def test_momentum_rank_ic_sign_and_dates_are_independent():
    panel = momentum_screen_panel().loc[
        pd.IndexSlice[[pd.Timestamp("2021-01-04"), pd.Timestamp("2022-07-01")], :],
        :,
    ]

    result = calculate_cross_sectional_ic(
        panel,
        min_symbols=4,
        method="spearman",
        feature_columns=["momentum_5"],
    )

    assert result["date"].tolist() == [
        pd.Timestamp("2021-01-04"),
        pd.Timestamp("2022-07-01"),
    ]
    assert result["ic"].tolist() == pytest.approx([1.0, -1.0])


def test_momentum_rank_ic_drops_only_aligned_missing_pairs():
    date = pd.Timestamp("2022-01-03")
    index = pd.MultiIndex.from_product(
        [[date], ["AAA", "BBB", "CCC", "DDD"]],
        names=["date", "symbol"],
    )
    panel = pd.DataFrame({
        "momentum_5": [1.0, 2.0, np.nan, 4.0],
        "target": [1.0, 2.0, 3.0, np.nan],
    }, index=index)

    result = calculate_cross_sectional_ic(
        panel,
        min_symbols=2,
        method="spearman",
        feature_columns=["momentum_5"],
    ).iloc[0]

    assert result["number_of_symbols"] == 2
    assert result["ic"] == pytest.approx(1.0)


def test_momentum_screen_keeps_periods_disjoint_and_all_frozen_horizons():
    result = run_momentum_feature_screen(momentum_screen_panel(), min_symbols=4)
    research = result["period_results"]["research"]
    validation = result["period_results"]["validation"]
    research_dates = research["panel"].index.get_level_values("date")
    validation_dates = validation["panel"].index.get_level_values("date")

    assert research_dates.max() < validation_dates.min()
    assert result["comparison"]["feature"].tolist() == MOMENTUM_FEATURE_COLUMNS
    assert research["summary"]["feature"].tolist() == MOMENTUM_FEATURE_COLUMNS
    assert validation["summary"]["feature"].tolist() == MOMENTUM_FEATURE_COLUMNS
    assert research["summary"]["number_of_dates"].tolist() == [4] * 5
    assert validation["summary"]["number_of_dates"].tolist() == [4] * 5
    assert research["summary"]["mean_number_of_symbols"].tolist() == [4.0] * 5


def test_momentum_screen_rejects_holdout_rows():
    panel = momentum_screen_panel()
    holdout = panel.iloc[[0]].copy()
    holdout.index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-01-02"), "AAA")],
        names=["date", "symbol"],
    )

    with pytest.raises(ValueError, match="holdout rows"):
        run_momentum_feature_screen(pd.concat([panel, holdout]).sort_index())


def test_future_targets_do_not_change_earlier_momentum_rank_ic():
    panel = momentum_screen_panel()
    changed = panel.copy()
    changed_date = pd.Timestamp("2025-12-30")
    changed.loc[(changed_date, slice(None)), "target"] = [40.0, 30.0, 20.0, 10.0]

    original = run_momentum_feature_screen(panel, min_symbols=4)
    altered = run_momentum_feature_screen(changed, min_symbols=4)
    original_ic = original["period_results"]["validation"]["ic_results"]
    altered_ic = altered["period_results"]["validation"]["ic_results"]

    pd.testing.assert_frame_equal(
        original_ic.loc[original_ic["date"] < changed_date].reset_index(drop=True),
        altered_ic.loc[altered_ic["date"] < changed_date].reset_index(drop=True),
    )


def test_future_prices_do_not_change_earlier_momentum_screening():
    dates = pd.bdate_range("2020-01-02", "2025-12-30")
    steps = np.arange(len(dates), dtype=float)
    original_data = {
        f"S{number}": pd.DataFrame({
            "Close": 100.0 + number * 10.0 + steps * (0.05 + number * 0.01)
            + np.sin(steps / (7.0 + number)),
        }, index=dates)
        for number in range(1, 5)
    }
    changed_data = {
        symbol: data.copy()
        for symbol, data in original_data.items()
    }
    cutoff = pd.Timestamp("2025-07-01")
    for number, data in enumerate(changed_data.values(), start=2):
        data.loc[data.index >= cutoff, "Close"] *= number

    original_panel = build_multi_asset_momentum_panel(original_data)
    changed_panel = build_multi_asset_momentum_panel(changed_data)
    last_safe_date = dates[dates.get_loc(cutoff) - 2]
    original_mask = original_panel.index.get_level_values("date") <= last_safe_date
    changed_mask = changed_panel.index.get_level_values("date") <= last_safe_date
    pd.testing.assert_frame_equal(
        original_panel.loc[original_mask, MOMENTUM_FEATURE_COLUMNS],
        changed_panel.loc[changed_mask, MOMENTUM_FEATURE_COLUMNS],
    )

    original = run_momentum_feature_screen(original_panel, min_symbols=4)
    altered = run_momentum_feature_screen(changed_panel, min_symbols=4)
    original_ic = original["period_results"]["validation"]["ic_results"]
    altered_ic = altered["period_results"]["validation"]["ic_results"]
    pd.testing.assert_frame_equal(
        original_ic.loc[original_ic["date"] <= last_safe_date].reset_index(drop=True),
        altered_ic.loc[altered_ic["date"] <= last_safe_date].reset_index(drop=True),
    )


def test_momentum_stability_uses_ordered_chronological_halves():
    stability = run_momentum_feature_screen(
        momentum_screen_panel(),
        min_symbols=4,
    )["stability"]
    groups = stability[[
        "period", "subperiod", "start_date", "end_date"
    ]].drop_duplicates().reset_index(drop=True)

    assert groups[["period", "subperiod"]].values.tolist() == [
        ["research", "first_half"],
        ["research", "second_half"],
        ["validation", "first_half"],
        ["validation", "second_half"],
    ]
    assert (groups["start_date"] <= groups["end_date"]).all()
    assert groups.loc[0, "end_date"] < groups.loc[1, "start_date"]
    assert groups.loc[2, "end_date"] < groups.loc[3, "start_date"]
    assert stability.groupby(["period", "subperiod"], sort=False).size().tolist() == [5] * 4


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
