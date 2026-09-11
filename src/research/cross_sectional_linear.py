"""Frozen cross-sectional linear-alpha diagnostics for Research and Validation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from .periods import HOLDOUT_PERIOD, filter_panel_by_period


PRIMARY_FEATURES_V1 = [
    "volatility_20",
    "momentum_60",
]

STANDARDIZED_FEATURE_COLUMNS_V1 = {
    feature: f"z_{feature}"
    for feature in PRIMARY_FEATURES_V1
}

MODEL_VARIANTS_V1 = {
    "volatility_20_only": ("volatility_20",),
    "momentum_60_only": ("momentum_60",),
    "volatility_20_and_momentum_60": tuple(PRIMARY_FEATURES_V1),
}


def _validate_panel_index(panel: pd.DataFrame) -> None:
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must use a (date, symbol) MultiIndex")
    if panel.index.names != ["date", "symbol"]:
        raise ValueError("panel index levels must be named date and symbol")
    if not panel.index.is_unique:
        raise ValueError("panel (date, symbol) index must be unique")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("panel index must be chronological")


def standardize_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score the two frozen features independently within each date.

    Population standard deviation is used. A valid feature with zero same-date
    dispersion receives a zero score. Target values are neither required nor
    consulted.
    """
    _validate_panel_index(panel)
    missing_features = set(PRIMARY_FEATURES_V1).difference(panel.columns)
    if missing_features:
        missing = ", ".join(sorted(missing_features))
        raise ValueError(f"panel is missing primary features: {missing}")

    standardized = pd.DataFrame(index=panel.index)
    for feature in PRIMARY_FEATURES_V1:
        feature_values = panel[feature].astype(float)
        grouped = feature_values.groupby(level="date")
        same_date_mean = grouped.transform("mean")
        same_date_std = grouped.transform("std", ddof=0)
        z_score = (feature_values - same_date_mean) / same_date_std
        zero_dispersion = same_date_std.eq(0) & feature_values.notna()
        standardized[STANDARDIZED_FEATURE_COLUMNS_V1[feature]] = z_score.mask(
            zero_dispersion,
            0.0,
        )

    return standardized


def calculate_prediction_rank_ic(
    predictions: pd.Series,
    actual: pd.Series,
    min_symbols: int = 5,
) -> pd.DataFrame:
    """Calculate same-date Spearman IC for model predictions."""
    if isinstance(min_symbols, bool) or not isinstance(min_symbols, int):
        raise TypeError("min_symbols must be an integer")
    if min_symbols < 2:
        raise ValueError("min_symbols must be at least 2")
    if not predictions.index.equals(actual.index):
        raise ValueError("prediction and actual indices must match exactly")

    values = pd.DataFrame({
        "prediction": predictions,
        "actual": actual,
    })
    _validate_panel_index(values)
    rows = []

    for date, date_values in values.groupby(level="date", sort=True):
        paired_values = date_values.dropna()
        number_of_symbols = len(paired_values)
        rank_ic = (
            paired_values["prediction"].corr(
                paired_values["actual"],
                method="spearman",
            )
            if (
                number_of_symbols >= min_symbols
                and paired_values["prediction"].nunique() > 1
                and paired_values["actual"].nunique() > 1
            )
            else np.nan
        )
        rows.append({
            "date": date,
            "rank_ic": rank_ic,
            "number_of_symbols": number_of_symbols,
        })

    return pd.DataFrame(
        rows,
        columns=["date", "rank_ic", "number_of_symbols"],
    )


def summarize_prediction_rank_ic(rank_ic: pd.DataFrame) -> dict[str, float]:
    """Summarize a prediction Rank-IC time series without selecting a model."""
    required_columns = {"date", "rank_ic", "number_of_symbols"}
    missing_columns = required_columns.difference(rank_ic.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"rank_ic is missing required columns: {missing}")

    valid = rank_ic.dropna(subset=["rank_ic"])
    values = valid["rank_ic"]
    mean_ic = values.mean()
    std_ic = values.std()
    icir = (
        mean_ic / std_ic
        if pd.notna(std_ic) and not np.isclose(std_ic, 0.0)
        else np.nan
    )

    return {
        "mean_prediction_ic": mean_ic,
        "median_prediction_ic": values.median(),
        "std_prediction_ic": std_ic,
        "prediction_icir": icir,
        "fraction_positive_prediction_ic": (
            float((values > 0).mean()) if not values.empty else np.nan
        ),
        "number_of_dates": int(valid["date"].nunique()),
        "mean_number_of_symbols": rank_ic["number_of_symbols"].mean(),
    }


def _numerical_metrics(actual: pd.Series, predictions: pd.Series) -> dict[str, float]:
    return {
        "rmse": np.sqrt(mean_squared_error(actual, predictions)),
        "r2": r2_score(actual, predictions),
    }


def _evaluate_predictions(
    actual: pd.Series,
    predictions: pd.Series,
    min_symbols: int,
) -> dict:
    rank_ic = calculate_prediction_rank_ic(
        predictions,
        actual,
        min_symbols=min_symbols,
    )
    return {
        "actual": actual.copy(),
        "predictions": predictions.copy(),
        "numerical_metrics": _numerical_metrics(actual, predictions),
        "rank_ic": rank_ic,
        "rank_ic_summary": summarize_prediction_rank_ic(rank_ic),
    }


def _validation_baselines(
    research_target: pd.Series,
    validation_target: pd.Series,
) -> dict:
    zero_predictions = pd.Series(
        0.0,
        index=validation_target.index,
        name="zero_return_prediction",
    )
    research_mean = float(research_target.mean())
    research_mean_predictions = pd.Series(
        research_mean,
        index=validation_target.index,
        name="research_mean_prediction",
    )
    equal_scores = pd.Series(
        0.0,
        index=validation_target.index,
        name="equal_score",
    )

    return {
        "zero_return": {
            "predictions": zero_predictions,
            **_numerical_metrics(validation_target, zero_predictions),
        },
        "research_mean": {
            "research_mean": research_mean,
            "predictions": research_mean_predictions,
            **_numerical_metrics(validation_target, research_mean_predictions),
        },
        "equal_score": {
            "scores": equal_scores,
            "rank_ic": None,
        },
    }


def run_cross_sectional_linear_research(
    panel: pd.DataFrame,
    min_symbols: int = 5,
) -> dict:
    """Fit three frozen pooled OLS variants on Research and predict Validation."""
    _validate_panel_index(panel)
    required_columns = {*PRIMARY_FEATURES_V1, "target"}
    missing_columns = required_columns.difference(panel.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"panel is missing required columns: {missing}")

    dates = panel.index.get_level_values("date")
    if (dates >= HOLDOUT_PERIOD.start).any():
        raise ValueError("holdout rows must not enter cross-sectional linear research")

    research_panel = filter_panel_by_period(panel, "research")
    validation_panel = filter_panel_by_period(panel, "validation")
    if research_panel.empty:
        raise ValueError("Research period contains no observations")
    if validation_panel.empty:
        raise ValueError("Validation period contains no observations")
    if research_panel.index.intersection(validation_panel.index).size:
        raise ValueError("Research and Validation rows must be disjoint")

    research_standardized = standardize_cross_sectional_features(research_panel)
    validation_standardized = standardize_cross_sectional_features(
        validation_panel
    )
    research_target = research_panel["target"].astype(float)
    validation_target = validation_panel["target"].astype(float)
    model_results = {}
    comparison_rows = []

    for model_name, features in MODEL_VARIANTS_V1.items():
        standardized_columns = [
            STANDARDIZED_FEATURE_COLUMNS_V1[feature]
            for feature in features
        ]
        research_data = pd.concat([
            research_standardized[standardized_columns],
            research_target.rename("target"),
        ], axis=1).dropna()
        validation_data = pd.concat([
            validation_standardized[standardized_columns],
            validation_target.rename("target"),
        ], axis=1).dropna()

        model = LinearRegression()
        model.fit(
            research_data[standardized_columns],
            research_data["target"],
        )
        research_predictions = pd.Series(
            model.predict(research_data[standardized_columns]),
            index=research_data.index,
            name="prediction",
        )
        validation_predictions = pd.Series(
            model.predict(validation_data[standardized_columns]),
            index=validation_data.index,
            name="prediction",
        )

        research_evaluation = _evaluate_predictions(
            research_data["target"],
            research_predictions,
            min_symbols,
        )
        validation_evaluation = _evaluate_predictions(
            validation_data["target"],
            validation_predictions,
            min_symbols,
        )
        coefficients = {
            feature: float(coefficient)
            for feature, coefficient in zip(features, model.coef_)
        }
        model_results[model_name] = {
            "features": features,
            "model": model,
            "intercept": float(model.intercept_),
            "coefficients": coefficients,
            "research": research_evaluation,
            "validation": validation_evaluation,
        }

        validation_metrics = validation_evaluation["numerical_metrics"]
        validation_ic = validation_evaluation["rank_ic_summary"]
        comparison_rows.append({
            "model": model_name,
            "validation_rmse": validation_metrics["rmse"],
            "validation_r2": validation_metrics["r2"],
            "validation_mean_rank_ic": validation_ic["mean_prediction_ic"],
            "validation_icir": validation_ic["prediction_icir"],
            "validation_fraction_positive_ic": validation_ic[
                "fraction_positive_prediction_ic"
            ],
        })

    return {
        "primary_features": tuple(PRIMARY_FEATURES_V1),
        "research_panel": research_panel,
        "validation_panel": validation_panel,
        "research_standardized": research_standardized,
        "validation_standardized": validation_standardized,
        "models": model_results,
        "validation_baselines": _validation_baselines(
            research_target,
            validation_target,
        ),
        "validation_comparison": pd.DataFrame(comparison_rows),
    }
