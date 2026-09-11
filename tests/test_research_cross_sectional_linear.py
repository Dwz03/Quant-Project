import numpy as np
import pandas as pd
import pytest

import src.research.cross_sectional_linear as cross_sectional_linear
from src.research.cross_sectional_linear import (
    MODEL_VARIANTS_V1,
    PRIMARY_FEATURES_V1,
    STANDARDIZED_FEATURE_COLUMNS_V1,
    calculate_prediction_rank_ic,
    run_cross_sectional_linear_research,
    standardize_cross_sectional_features,
    summarize_prediction_rank_ic,
)


def model_panel():
    dates = pd.to_datetime([
        "2021-03-31",
        "2021-09-30",
        "2022-03-31",
        "2022-09-30",
        "2023-03-31",
        "2023-09-29",
        "2024-03-28",
        "2024-09-30",
        "2025-03-31",
        "2025-09-30",
    ])
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rows = []
    index = []
    for date_number, date in enumerate(dates):
        for symbol_number, symbol in enumerate(symbols):
            volatility = 0.1 * symbol_number + 0.01 * date_number
            momentum = (symbol_number - 2.5) ** 2 + 0.05 * date_number
            target = (
                -0.002 * volatility
                + 0.001 * momentum
                + 0.0001 * date_number * symbol_number
            )
            rows.append({
                "volatility_20": volatility,
                "momentum_60": momentum,
                "target": target,
            })
            index.append((date, symbol))

    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["date", "symbol"]),
    )


def test_primary_features_and_model_variants_are_frozen():
    assert PRIMARY_FEATURES_V1 == ["volatility_20", "momentum_60"]
    assert MODEL_VARIANTS_V1 == {
        "volatility_20_only": ("volatility_20",),
        "momentum_60_only": ("momentum_60",),
        "volatility_20_and_momentum_60": (
            "volatility_20",
            "momentum_60",
        ),
    }


def test_standardization_is_same_date_target_free_and_zero_safe():
    panel = model_panel().iloc[:12].copy()
    first_date = panel.index.get_level_values("date").min()
    panel.loc[(first_date, slice(None)), "volatility_20"] = 3.0

    standardized = standardize_cross_sectional_features(panel)
    target_changed = panel.copy()
    target_changed["target"] = np.arange(len(target_changed)) * 100.0
    target_changed_standardized = standardize_cross_sectional_features(
        target_changed
    )

    assert standardized.columns.tolist() == [
        "z_volatility_20",
        "z_momentum_60",
    ]
    pd.testing.assert_frame_equal(standardized, target_changed_standardized)
    assert np.all(
        standardized.loc[(first_date, slice(None)), "z_volatility_20"] == 0.0
    )

    second_date = panel.index.get_level_values("date").max()
    second_scores = standardized.loc[(second_date, slice(None))]
    assert second_scores["z_momentum_60"].mean() == pytest.approx(0.0)
    assert second_scores["z_momentum_60"].std(ddof=0) == pytest.approx(1.0)


def test_future_date_mutation_does_not_change_earlier_standardization():
    panel = model_panel()
    cutoff = pd.Timestamp("2023-09-29")
    changed = panel.copy()
    future = changed.index.get_level_values("date") > cutoff
    changed.loc[future, "volatility_20"] *= 100.0
    changed.loc[future, "momentum_60"] -= 500.0

    original_scores = standardize_cross_sectional_features(panel)
    changed_scores = standardize_cross_sectional_features(changed)
    earlier = original_scores.index.get_level_values("date") <= cutoff
    pd.testing.assert_frame_equal(
        original_scores.loc[earlier],
        changed_scores.loc[earlier],
    )


def test_prediction_rank_ic_is_cross_sectional_and_summarized_safely():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=["date", "symbol"],
    )
    predictions = pd.Series(np.tile(np.arange(5.0), 2), index=index)
    actual = pd.Series(
        np.concatenate([np.arange(5.0), np.arange(5.0)[::-1]]),
        index=index,
    )

    rank_ic = calculate_prediction_rank_ic(predictions, actual, min_symbols=5)
    summary = summarize_prediction_rank_ic(rank_ic)

    assert rank_ic["rank_ic"].tolist() == pytest.approx([1.0, -1.0])
    assert rank_ic["number_of_symbols"].tolist() == [5, 5]
    assert summary["mean_prediction_ic"] == pytest.approx(0.0)
    assert summary["median_prediction_ic"] == pytest.approx(0.0)
    assert summary["std_prediction_ic"] == pytest.approx(np.sqrt(2.0))
    assert summary["prediction_icir"] == pytest.approx(0.0)
    assert summary["fraction_positive_prediction_ic"] == pytest.approx(0.5)
    assert summary["number_of_dates"] == 2
    assert summary["mean_number_of_symbols"] == pytest.approx(5.0)

    equal_predictions = pd.Series(0.0, index=index)
    equal_ic = calculate_prediction_rank_ic(
        equal_predictions,
        actual,
        min_symbols=5,
    )
    assert equal_ic["rank_ic"].isna().all()
    assert np.isnan(summarize_prediction_rank_ic(equal_ic)["prediction_icir"])


def test_research_fits_only_research_and_returns_fixed_comparison(monkeypatch):
    fit_indices = []
    original_model = cross_sectional_linear.LinearRegression

    class RecordingLinearRegression(original_model):
        def fit(self, X, y):
            fit_indices.append(X.index.copy())
            return super().fit(X, y)

    monkeypatch.setattr(
        cross_sectional_linear,
        "LinearRegression",
        RecordingLinearRegression,
    )
    result = run_cross_sectional_linear_research(model_panel(), min_symbols=5)

    assert len(fit_indices) == 3
    for fit_index in fit_indices:
        assert fit_index.get_level_values("date").max() <= pd.Timestamp(
            "2023-12-31"
        )

    assert result["research_panel"].index.intersection(
        result["validation_panel"].index
    ).empty
    assert result["validation_comparison"].columns.tolist() == [
        "model",
        "validation_rmse",
        "validation_r2",
        "validation_mean_rank_ic",
        "validation_icir",
        "validation_fraction_positive_ic",
    ]
    assert result["validation_comparison"]["model"].tolist() == list(
        MODEL_VARIANTS_V1
    )

    for model_result in result["models"].values():
        assert model_result["research"]["predictions"].index.equals(
            model_result["research"]["actual"].index
        )
        assert model_result["validation"]["predictions"].index.equals(
            model_result["validation"]["actual"].index
        )
        assert set(model_result["coefficients"]) == set(model_result["features"])


def test_validation_targets_cannot_change_fit_or_research_mean_baseline():
    panel = model_panel()
    original = run_cross_sectional_linear_research(panel, min_symbols=5)
    changed_panel = panel.copy()
    validation = changed_panel.index.get_level_values("date") >= pd.Timestamp(
        "2024-01-01"
    )
    changed_panel.loc[validation, "target"] += np.arange(validation.sum()) + 10.0
    changed = run_cross_sectional_linear_research(changed_panel, min_symbols=5)

    for model_name in MODEL_VARIANTS_V1:
        assert original["models"][model_name]["intercept"] == pytest.approx(
            changed["models"][model_name]["intercept"]
        )
        assert original["models"][model_name]["coefficients"] == pytest.approx(
            changed["models"][model_name]["coefficients"]
        )
        pd.testing.assert_series_equal(
            original["models"][model_name]["validation"]["predictions"],
            changed["models"][model_name]["validation"]["predictions"],
        )

    expected_mean = original["research_panel"]["target"].mean()
    original_baseline = original["validation_baselines"]["research_mean"]
    changed_baseline = changed["validation_baselines"]["research_mean"]
    assert original_baseline["research_mean"] == pytest.approx(expected_mean)
    assert changed_baseline["research_mean"] == pytest.approx(expected_mean)
    assert np.all(original_baseline["predictions"] == expected_mean)
    assert original["validation_baselines"]["equal_score"]["rank_ic"] is None


def test_holdout_rows_are_rejected():
    panel = model_panel()
    holdout_rows = panel.iloc[:6].copy()
    holdout_rows.index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2026-01-02")], ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]],
        names=["date", "symbol"],
    )
    with pytest.raises(ValueError, match="holdout"):
        run_cross_sectional_linear_research(
            pd.concat([panel, holdout_rows]).sort_index(),
            min_symbols=5,
        )
