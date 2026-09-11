"""Leakage-aware price features and descriptive multi-asset screening.

Large isolated correlations can occur by chance. Screening many features across
many symbols creates multiple-testing risk, so these diagnostics do not by
themselves validate alpha and must not be treated as a feature-selection rule.
"""

import numpy as np
import pandas as pd

from .periods import filter_panel_by_period


FEATURE_COLUMNS_V1 = [
    "return_1",
    "return_2",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "volatility_5",
    "volatility_20",
    "trend_20",
]


def build_price_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build V1 features known at time t and the next-return target at t+1."""
    if "Close" not in data.columns:
        raise ValueError("data must contain a Close column")
    if data.empty:
        raise ValueError("data cannot be empty")
    if not data.index.is_monotonic_increasing:
        raise ValueError("data index must be chronological")
    if not data.index.is_unique:
        raise ValueError("data index must be unique")

    close = data["Close"].astype(float)
    close_values = close.to_numpy(dtype=float)
    if not np.isfinite(close_values).all() or (close_values <= 0).any():
        raise ValueError("Close prices must be finite and positive")

    return_1 = close.pct_change()
    moving_average_20 = close.rolling(20).mean()

    dataset = pd.DataFrame({
        "return_1": return_1,
        "return_2": return_1.shift(1),
        "momentum_5": close.pct_change(5),
        "momentum_20": close.pct_change(20),
        "momentum_60": close.pct_change(60),
        "volatility_5": return_1.rolling(5).std(),
        "volatility_20": return_1.rolling(20).std(),
        "trend_20": close / moving_average_20 - 1,
        "target": return_1.shift(-1),
    }, index=data.index)

    return dataset.dropna()


def run_multi_asset_feature_screen(
    symbol_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Calculate descriptive feature/target correlations for each symbol.

    Results are diagnostics only. They do not account for multiple testing and
    must not be interpreted as validated alpha or an automated selection rule.
    """
    if not symbol_data:
        raise ValueError("symbol_data must contain at least one symbol")

    rows = []
    for symbol, data in symbol_data.items():
        dataset = build_price_features(data)

        for feature in FEATURE_COLUMNS_V1:
            paired_values = dataset[[feature, "target"]].dropna()
            rows.append({
                "symbol": symbol,
                "feature": feature,
                "correlation": paired_values[feature].corr(
                    paired_values["target"]
                ),
                "sample_size": len(paired_values),
            })

    return pd.DataFrame(
        rows,
        columns=["symbol", "feature", "correlation", "sample_size"],
    )


def summarize_feature_screen(screen_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize correlations across symbols without ranking or selecting."""
    required_columns = {"symbol", "feature", "correlation", "sample_size"}
    missing_columns = required_columns.difference(screen_results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"screen_results is missing required columns: {missing}")

    rows = []
    for feature in FEATURE_COLUMNS_V1:
        feature_results = screen_results.loc[
            screen_results["feature"] == feature
        ]
        if feature_results.empty:
            continue

        correlations = feature_results["correlation"].dropna()
        fraction_positive = (
            float((correlations > 0).mean())
            if not correlations.empty
            else np.nan
        )
        rows.append({
            "feature": feature,
            "mean_correlation": correlations.mean(),
            "median_correlation": correlations.median(),
            "std_correlation": correlations.std(),
            "fraction_positive": fraction_positive,
            "number_of_symbols": feature_results["symbol"].nunique(),
        })

    return pd.DataFrame(rows, columns=[
        "feature",
        "mean_correlation",
        "median_correlation",
        "std_correlation",
        "fraction_positive",
        "number_of_symbols",
    ])


def build_multi_asset_feature_panel(
    symbol_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a chronological (date, symbol) panel without filling missing rows."""
    if not symbol_data:
        raise ValueError("symbol_data must contain at least one symbol")

    symbol_features = {
        symbol: build_price_features(data)
        for symbol, data in symbol_data.items()
    }
    panel = pd.concat(
        symbol_features,
        names=["symbol", "date"],
    ).reorder_levels(["date", "symbol"]).sort_index()

    if not panel.index.is_unique:
        raise ValueError("panel (date, symbol) index must be unique")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("panel index must be chronological")

    return panel[[*FEATURE_COLUMNS_V1, "target"]]


def calculate_cross_sectional_ic(
    panel: pd.DataFrame,
    min_symbols: int = 5,
    method: str = "pearson",
) -> pd.DataFrame:
    """Calculate same-date cross-sectional feature ICs across symbols.

    This is exploratory descriptive research, not a feature-selection or alpha
    validation procedure. Rows below ``min_symbols`` are retained with NaN ICs
    so that insufficient cross-sectional coverage remains visible.
    """
    if isinstance(min_symbols, bool) or not isinstance(min_symbols, int):
        raise TypeError("min_symbols must be an integer")
    if min_symbols < 2:
        raise ValueError("min_symbols must be at least 2")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must use a (date, symbol) MultiIndex")
    if panel.index.names != ["date", "symbol"]:
        raise ValueError("panel index levels must be named date and symbol")
    if not panel.index.is_unique:
        raise ValueError("panel (date, symbol) index must be unique")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("panel index must be chronological")

    required_columns = {*FEATURE_COLUMNS_V1, "target"}
    missing_columns = required_columns.difference(panel.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"panel is missing required columns: {missing}")

    rows = []
    for date, date_data in panel.groupby(level="date", sort=True):
        for feature in FEATURE_COLUMNS_V1:
            paired_values = date_data[[feature, "target"]].dropna()
            number_of_symbols = len(paired_values)
            ic = (
                paired_values[feature].corr(
                    paired_values["target"],
                    method=method,
                )
                if number_of_symbols >= min_symbols
                else np.nan
            )
            rows.append({
                "date": date,
                "feature": feature,
                "ic": ic,
                "number_of_symbols": number_of_symbols,
            })

    return pd.DataFrame(
        rows,
        columns=["date", "feature", "ic", "number_of_symbols"],
    )


def summarize_cross_sectional_ic(ic_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize exploratory ICs without ranking or selecting features."""
    required_columns = {"date", "feature", "ic", "number_of_symbols"}
    missing_columns = required_columns.difference(ic_results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ic_results is missing required columns: {missing}")

    rows = []
    for feature in FEATURE_COLUMNS_V1:
        feature_results = ic_results.loc[ic_results["feature"] == feature]
        if feature_results.empty:
            continue

        valid_ic = feature_results.dropna(subset=["ic"])
        ic_values = valid_ic["ic"]
        mean_ic = ic_values.mean()
        std_ic = ic_values.std()
        icir = (
            mean_ic / std_ic
            if pd.notna(std_ic) and not np.isclose(std_ic, 0.0)
            else np.nan
        )
        fraction_positive = (
            float((ic_values > 0).mean())
            if not ic_values.empty
            else np.nan
        )
        rows.append({
            "feature": feature,
            "mean_ic": mean_ic,
            "median_ic": ic_values.median(),
            "std_ic": std_ic,
            "icir": icir,
            "fraction_positive_ic": fraction_positive,
            "number_of_dates": valid_ic["date"].nunique(),
            "mean_number_of_symbols": feature_results[
                "number_of_symbols"
            ].mean(),
        })

    return pd.DataFrame(rows, columns=[
        "feature",
        "mean_ic",
        "median_ic",
        "std_ic",
        "icir",
        "fraction_positive_ic",
        "number_of_dates",
        "mean_number_of_symbols",
    ])


def run_cross_sectional_ic_by_period(
    panel: pd.DataFrame,
    period_names: tuple[str, ...] | list[str],
    min_symbols: int = 5,
    method: str = "spearman",
    allow_holdout: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run the same fixed IC analysis independently within named periods."""
    periods_used = list(period_names)
    if not periods_used:
        raise ValueError("period_names must contain at least one period")
    if len(set(periods_used)) != len(periods_used):
        raise ValueError("period_names must be unique")

    results = {}
    for period_name in periods_used:
        period_panel = filter_panel_by_period(
            panel,
            period_name,
            allow_holdout=allow_holdout,
        )
        ic_results = calculate_cross_sectional_ic(
            period_panel,
            min_symbols=min_symbols,
            method=method,
        )
        results[period_name] = {
            "panel": period_panel,
            "ic_results": ic_results,
            "summary": summarize_cross_sectional_ic(ic_results),
        }

    return results


def summarize_feature_stability(
    research_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Research and Validation IC signs without selecting features."""
    required_columns = {"feature", "mean_ic", "icir"}
    for name, summary in (
        ("research_summary", research_summary),
        ("validation_summary", validation_summary),
    ):
        missing_columns = required_columns.difference(summary.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{name} is missing required columns: {missing}")

    research = research_summary.set_index("feature").reindex(FEATURE_COLUMNS_V1)
    validation = validation_summary.set_index("feature").reindex(
        FEATURE_COLUMNS_V1
    )
    rows = []

    for feature in FEATURE_COLUMNS_V1:
        research_mean_ic = research.loc[feature, "mean_ic"]
        validation_mean_ic = validation.loc[feature, "mean_ic"]
        sign_consistent = (
            bool(np.sign(research_mean_ic) == np.sign(validation_mean_ic))
            if pd.notna(research_mean_ic) and pd.notna(validation_mean_ic)
            else pd.NA
        )
        rows.append({
            "feature": feature,
            "research_mean_ic": research_mean_ic,
            "research_icir": research.loc[feature, "icir"],
            "validation_mean_ic": validation_mean_ic,
            "validation_icir": validation.loc[feature, "icir"],
            "sign_consistent": sign_consistent,
        })

    return pd.DataFrame(rows, columns=[
        "feature",
        "research_mean_ic",
        "research_icir",
        "validation_mean_ic",
        "validation_icir",
        "sign_consistent",
    ])


def calculate_feature_redundancy(
    research_panel: pd.DataFrame,
    min_symbols: int = 5,
    method: str = "spearman",
) -> pd.DataFrame:
    """Average same-date feature correlations over the Research period only.

    The target is intentionally excluded. This is a descriptive redundancy
    diagnostic and does not select or eliminate features.
    """
    if isinstance(min_symbols, bool) or not isinstance(min_symbols, int):
        raise TypeError("min_symbols must be an integer")
    if min_symbols < 2:
        raise ValueError("min_symbols must be at least 2")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    if not isinstance(research_panel.index, pd.MultiIndex):
        raise ValueError("research_panel must use a (date, symbol) MultiIndex")
    if research_panel.index.names != ["date", "symbol"]:
        raise ValueError("panel index levels must be named date and symbol")
    if not research_panel.index.is_unique:
        raise ValueError("panel (date, symbol) index must be unique")

    missing_features = set(FEATURE_COLUMNS_V1).difference(
        research_panel.columns
    )
    if missing_features:
        missing = ", ".join(sorted(missing_features))
        raise ValueError(f"research_panel is missing features: {missing}")

    correlation_sum = pd.DataFrame(
        0.0,
        index=FEATURE_COLUMNS_V1,
        columns=FEATURE_COLUMNS_V1,
    )
    correlation_count = pd.DataFrame(
        0,
        index=FEATURE_COLUMNS_V1,
        columns=FEATURE_COLUMNS_V1,
    )

    for _, date_data in research_panel.groupby(level="date", sort=True):
        feature_data = date_data[FEATURE_COLUMNS_V1].dropna()
        if len(feature_data) < min_symbols:
            continue

        date_correlation = feature_data.corr(method=method)
        valid = date_correlation.notna()
        correlation_sum += date_correlation.fillna(0.0)
        correlation_count += valid.astype(int)

    return correlation_sum.div(correlation_count.replace(0, np.nan))
