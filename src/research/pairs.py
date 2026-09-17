import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations

from src.research.common import (normalize_positions, calculate_zscore, calculate_equity_curve, calculate_zscore_with_history)
from src.research.periods import HOLDOUT_PERIOD, RESEARCH_PERIOD, VALIDATION_PERIOD


PAIR_COLUMNS = ("symbol_1", "symbol_2")
PAIR_DIAGNOSTIC_ROLLING_WINDOW = 60
RESEARCH_CANDIDATE_P_THRESHOLD = 0.05
VALIDATION_FDR_LEVEL = 0.05
PAIR_DIAGNOSTIC_COLUMNS = (
    "symbol_1",
    "symbol_2",
    "aligned_observations",
    "alpha",
    "beta",
    "cointegration_test_statistic",
    "cointegration_p_value",
    "adf_statistic",
    "adf_p_value",
    "half_life",
    "mean_beta",
    "median_beta",
    "std_beta",
    "min_beta",
    "max_beta",
    "valid_estimates",
)


def _validate_pair_prices(data, *, min_observations=2):
    """Validate aligned Y/X prices without modifying the caller's data."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    missing_columns = [column for column in PAIR_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"data is missing required columns: {missing_columns}")
    if data.index.has_duplicates:
        raise ValueError("pair price index must not contain duplicate dates")
    if not data.index.is_monotonic_increasing:
        raise ValueError("pair prices must be in chronological order")
    if len(data) < min_observations:
        raise ValueError(
            f"at least {min_observations} aligned observations are required"
        )

    prices = data.loc[:, list(PAIR_COLUMNS)].copy()
    try:
        values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("pair prices must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("pair prices must be finite with no missing observations")
    if (values <= 0).any():
        raise ValueError("pair prices must be positive")

    return prices


def _validate_series(series, *, name, min_observations):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if series.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicates")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} must be in chronological order")
    if len(series) < min_observations:
        raise ValueError(
            f"at least {min_observations} {name} observations are required"
        )
    try:
        values = series.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite with no missing observations")
    return pd.Series(values, index=series.index, name=series.name)

def calculate_spread(data, beta, alpha = 0.0):
    """Calculate Y - alpha - beta * X using pre-fitted parameters."""
    _validate_pair_prices(data)
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("alpha and beta must be finite")

    data_used = data.copy(deep=True)

    data_used["spread"] = data_used["symbol_1"] - beta * data_used["symbol_2"] - alpha

    return data_used

def estimate_beta(data):

    data_used = data.copy()
    data_used["product"] = data_used["symbol_1"] * data_used["symbol_2"]
    data_used["squared"] = data_used["symbol_2"] ** 2
    beta = data_used["product"].sum() / data_used["squared"].sum()

    return beta

def estimate_hedge_ratio(data):
    """Fit the static OLS model Y = alpha + beta * X."""
    prices = _validate_pair_prices(data)
    y = prices["symbol_1"]
    x = prices["symbol_2"]

    if x.nunique() < 2:
        raise ValueError("symbol_2 must vary to estimate a hedge ratio")

    X = np.column_stack([np.ones(len(x)), x])

    coefficients = np.linalg.lstsq(X, y.to_numpy(dtype=float), rcond=None)[0]

    alpha = coefficients[0]
    beta = coefficients[1]

    return {
        "alpha": float(alpha),
        "beta": float(beta)
    }


def rolling_hedge_ratio(data, window):
    """Estimate trailing OLS alpha/beta using each row and its prior history."""
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an integer")
    if window < 2:
        raise ValueError("window must be at least 2")

    prices = _validate_pair_prices(data)
    y = prices["symbol_1"]
    x = prices["symbol_2"]
    x_mean = x.rolling(window=window, min_periods=window).mean()
    y_mean = y.rolling(window=window, min_periods=window).mean()
    covariance = x.rolling(window=window, min_periods=window).cov(y)
    variance = x.rolling(window=window, min_periods=window).var()
    beta = covariance / variance.replace(0.0, np.nan)
    alpha = y_mean - beta * x_mean

    return pd.DataFrame({"alpha": alpha, "beta": beta}, index=prices.index)


# Explicit aliases make the estimation direction clear at call sites.
estimate_rolling_hedge_ratio = rolling_hedge_ratio


def summarize_beta_stability(rolling_beta):
    """Return descriptive statistics for finite rolling beta estimates only."""
    if isinstance(rolling_beta, pd.DataFrame):
        if "beta" not in rolling_beta:
            raise ValueError("rolling beta DataFrame must contain a 'beta' column")
        beta = rolling_beta["beta"]
    elif isinstance(rolling_beta, pd.Series):
        beta = rolling_beta
    else:
        raise TypeError("rolling_beta must be a pandas Series or DataFrame")

    if beta.index.has_duplicates:
        raise ValueError("rolling beta index must not contain duplicates")
    if not beta.index.is_monotonic_increasing:
        raise ValueError("rolling beta estimates must be in chronological order")

    try:
        numeric_beta = beta.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("rolling beta estimates must be numeric") from exc
    valid = numeric_beta[np.isfinite(numeric_beta)]

    return {
        "mean_beta": float(valid.mean()) if len(valid) else np.nan,
        "median_beta": float(valid.median()) if len(valid) else np.nan,
        "std_beta": float(valid.std(ddof=1)) if len(valid) else np.nan,
        "min_beta": float(valid.min()) if len(valid) else np.nan,
        "max_beta": float(valid.max()) if len(valid) else np.nan,
        "valid_estimates": int(len(valid)),
    }

def generate_pair_positions(zscore, threshold, beta):

    position_1 = pd.Series(0.0, index = zscore.index)
    position_2 = pd.Series(0.0, index = zscore.index)

    position_1.loc[zscore > threshold] = -1
    position_2.loc[zscore > threshold] = +beta

    position_1.loc[zscore < -threshold] = 1
    position_2.loc[zscore < -threshold] = -beta

    result = pd.DataFrame({
        "position_1": position_1,
        "position_2": position_2
        })

    return result

def calculate_pair_weights(data, positions):

    notionals = pd.DataFrame(index=data.index)

    notionals["symbol_1"] = (
        positions["position_1"] * data["symbol_1"]
    )

    notionals["symbol_2"] = (
        positions["position_2"] * data["symbol_2"]
    )

    weights = normalize_positions(notionals)

    return weights

def calculate_pair_returns(data, weights):

    data_used = data.copy()

    data_used["return_1"] = data_used["symbol_1"].pct_change()
    data_used["return_2"] = data_used["symbol_2"].pct_change()

    weights_used = weights.shift(1)

    data_used["strategy_return"] = (
        weights_used["symbol_1"] * data_used["return_1"]
        + weights_used["symbol_2"] * data_used["return_2"]
    )

    data_used.loc[data_used.index[0], "strategy_return"] = 0.0

    return data_used

def spread_stationarity_diagnostic(spread):
    """Run an ADF diagnostic without applying a decision threshold."""
    spread_used = _validate_series(
        spread, name="spread", min_observations=4
    )
    result = adfuller(spread_used, result_object=False)

    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_observations": result[3],
        "critical_values": result[4],
    }


def check_spread_stationarity(spread):
    """Legacy ADF wrapper retaining the V1 fixed-threshold output."""
    result = spread_stationarity_diagnostic(spread)
    return {
        "adf_statistic": result["adf_statistic"],
        "p_value": result["p_value"],
        "is_stationary": result["p_value"] < 0.05,
    }


def cointegration_diagnostic(data):
    """Run the Engle-Granger cointegration test without declaring tradability."""
    prices = _validate_pair_prices(data, min_observations=4)
    result = coint(prices["symbol_1"], prices["symbol_2"])

    return {
        "test_statistic": result[0],
        "p_value": result[1],
        "critical_values": result[2],
    }


def check_cointegration(data):
    """Legacy Engle-Granger wrapper retaining the V1 threshold output."""
    result = cointegration_diagnostic(data)
    return {
        "test_statistic": result["test_statistic"],
        "p_value": result["p_value"],
        "is_cointegrated": result["p_value"] < 0.05,
    }


def estimate_spread_half_life(spread):
    """Estimate half-life from delta(spread) = a + lambda * spread[-1].

    A non-negative lambda does not imply mean reversion and therefore returns
    NaN instead of a misleading positive horizon.
    """
    spread_used = _validate_series(spread, name="spread", min_observations=3)
    lagged = spread_used.iloc[:-1].to_numpy(dtype=float)
    changes = np.diff(spread_used.to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(lagged)), lagged])

    if np.linalg.matrix_rank(design) < 2:
        return {
            "intercept": np.nan,
            "lambda": np.nan,
            "half_life": np.nan,
        }

    intercept, lambda_coefficient = np.linalg.lstsq(
        design, changes, rcond=None
    )[0]
    half_life = (
        -np.log(2.0) / lambda_coefficient
        if lambda_coefficient < 0
        else np.nan
    )

    return {
        "intercept": float(intercept),
        "lambda": float(lambda_coefficient),
        "half_life": float(half_life),
    }


estimate_half_life = estimate_spread_half_life


def generate_deterministic_pairs(symbols):
    """Generate unique unordered pairs in deterministic Y/X orientation.

    Symbols are sorted lexicographically. The earlier symbol is always
    ``symbol_1`` (Y) and the later symbol is always ``symbol_2`` (X).
    """
    symbols_used = list(symbols)
    if len(set(symbols_used)) != len(symbols_used):
        raise ValueError("symbols must be unique")
    if any(not isinstance(symbol, str) or not symbol for symbol in symbols_used):
        raise ValueError("symbols must be non-empty strings")
    return list(combinations(sorted(symbols_used), 2))


def _build_period_price_matrix(symbol_data, period):
    if not isinstance(symbol_data, dict) or len(symbol_data) < 2:
        raise ValueError("symbol_data must contain at least two symbols")

    closes = []
    for symbol in sorted(symbol_data):
        data = symbol_data[symbol]
        if isinstance(data, pd.DataFrame):
            if "Close" not in data:
                raise ValueError(f"market data for {symbol} must contain Close")
            close = data["Close"].copy()
        elif isinstance(data, pd.Series):
            close = data.copy()
        else:
            raise TypeError(f"market data for {symbol} must be a Series or DataFrame")

        try:
            close.index = pd.DatetimeIndex(close.index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"market data index for {symbol} must contain dates") from exc
        if close.index.has_duplicates:
            raise ValueError(f"market data index for {symbol} contains duplicate dates")
        if not close.index.is_monotonic_increasing:
            raise ValueError(f"market data for {symbol} must be chronological")
        if (close.index >= HOLDOUT_PERIOD.start).any():
            raise ValueError(f"holdout observations are not allowed for {symbol}")

        period_close = close.loc[
            (close.index >= period.start)
            & (close.index <= period.end)
        ].astype(float)
        valid_values = period_close.dropna().to_numpy(dtype=float)
        if not np.isfinite(valid_values).all() or (valid_values <= 0).any():
            raise ValueError(
                f"{period.name.title()} prices for {symbol} must be finite and positive"
            )
        period_close.name = symbol
        closes.append(period_close)

    prices = pd.concat(closes, axis=1, join="outer").sort_index()
    if prices.empty:
        raise ValueError(f"no {period.name.title()}-period observations are available")
    return prices


def build_research_price_matrix(symbol_data):
    """Build an unfilled wide close-price matrix for the Research period.

    Validation and pre-Research rows are excluded. Any Holdout row fails
    closed before diagnostics can run.
    """
    return _build_period_price_matrix(symbol_data, RESEARCH_PERIOD)


def build_validation_price_matrix(symbol_data):
    """Build an unfilled Validation-only close matrix and reject Holdout rows."""
    return _build_period_price_matrix(symbol_data, VALIDATION_PERIOD)


def benjamini_hochberg(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction while preserving raw values."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    raw = pd.Series(p_values, copy=True, dtype=float)
    finite = np.isfinite(raw)
    if ((raw[finite] < 0) | (raw[finite] > 1)).any():
        raise ValueError("p-values must be between 0 and 1")

    q_values = pd.Series(np.nan, index=raw.index, dtype=float)
    significant = pd.Series(False, index=raw.index, dtype=bool)
    if finite.any():
        rejected, adjusted, _, _ = multipletests(
            raw.loc[finite].to_numpy(), alpha=alpha, method="fdr_bh"
        )
        q_values.loc[finite] = adjusted
        significant.loc[finite] = rejected

    return pd.DataFrame({
        "raw_p_value": raw,
        "q_value": q_values,
        "fdr_significant": significant,
    })


def _evaluate_pair_diagnostics(
    prices,
    candidates,
    rolling_window,
    min_observations,
):
    diagnostics = []
    skipped = []
    for symbol_1, symbol_2 in candidates:
        missing_symbols = [
            symbol for symbol in (symbol_1, symbol_2) if symbol not in prices
        ]
        if missing_symbols:
            skipped.append({
                "symbol_1": symbol_1,
                "symbol_2": symbol_2,
                "aligned_observations": 0,
                "reason": f"missing symbols: {missing_symbols}",
            })
            continue

        aligned = prices.loc[:, [symbol_1, symbol_2]].dropna(how="any")
        aligned.columns = list(PAIR_COLUMNS)
        observation_count = len(aligned)
        if observation_count < min_observations:
            skipped.append({
                "symbol_1": symbol_1,
                "symbol_2": symbol_2,
                "aligned_observations": observation_count,
                "reason": f"fewer than {min_observations} aligned observations",
            })
            continue

        try:
            fitted = estimate_hedge_ratio(aligned)
            spread = calculate_spread(
                aligned, beta=fitted["beta"], alpha=fitted["alpha"]
            )["spread"]
            cointegration = cointegration_diagnostic(aligned)
            stationarity = spread_stationarity_diagnostic(spread)
            half_life = estimate_spread_half_life(spread)["half_life"]
            rolling = rolling_hedge_ratio(aligned, window=rolling_window)
            stability = summarize_beta_stability(rolling)
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            skipped.append({
                "symbol_1": symbol_1,
                "symbol_2": symbol_2,
                "aligned_observations": observation_count,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue

        diagnostics.append({
            "symbol_1": symbol_1,
            "symbol_2": symbol_2,
            "aligned_observations": observation_count,
            "alpha": fitted["alpha"],
            "beta": fitted["beta"],
            "cointegration_test_statistic": cointegration["test_statistic"],
            "cointegration_p_value": cointegration["p_value"],
            "adf_statistic": stationarity["adf_statistic"],
            "adf_p_value": stationarity["p_value"],
            "half_life": half_life,
            **stability,
        })

    return (
        pd.DataFrame(diagnostics, columns=PAIR_DIAGNOSTIC_COLUMNS),
        pd.DataFrame(
            skipped,
            columns=("symbol_1", "symbol_2", "aligned_observations", "reason"),
        ),
    )


def screen_pair_diagnostics(
    research_prices,
    rolling_window=PAIR_DIAGNOSTIC_ROLLING_WINDOW,
    min_observations=None,
    fdr_level=0.05,
):
    """Evaluate all unique pairs in an already bounded Research price matrix."""
    if not isinstance(research_prices, pd.DataFrame):
        raise TypeError("research_prices must be a pandas DataFrame")
    if not isinstance(research_prices.index, pd.DatetimeIndex):
        raise ValueError("research_prices must use a DatetimeIndex")
    if research_prices.index.has_duplicates:
        raise ValueError("research_prices index must not contain duplicates")
    if not research_prices.index.is_monotonic_increasing:
        raise ValueError("research_prices must be chronological")
    if (
        (research_prices.index < RESEARCH_PERIOD.start).any()
        or (research_prices.index > RESEARCH_PERIOD.end).any()
    ):
        raise ValueError("screening input must contain Research-period rows only")
    if isinstance(rolling_window, bool) or not isinstance(rolling_window, int):
        raise TypeError("rolling_window must be an integer")
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")
    if min_observations is None:
        min_observations = rolling_window
    if min_observations < rolling_window:
        raise ValueError("min_observations must be at least rolling_window")

    candidates = generate_deterministic_pairs(research_prices.columns)
    diagnostics_frame, skipped_frame = _evaluate_pair_diagnostics(
        research_prices,
        candidates,
        rolling_window=rolling_window,
        min_observations=min_observations,
    )
    if not diagnostics_frame.empty:
        correction = benjamini_hochberg(
            diagnostics_frame["cointegration_p_value"], alpha=fdr_level
        )
        diagnostics_frame["cointegration_q_value"] = correction["q_value"]
        diagnostics_frame["fdr_significant_05"] = correction["fdr_significant"]
    else:
        diagnostics_frame["cointegration_q_value"] = pd.Series(dtype=float)
        diagnostics_frame["fdr_significant_05"] = pd.Series(dtype=bool)

    return {
        "diagnostics": diagnostics_frame,
        "skipped": skipped_frame,
        "total_pairs": len(candidates),
        "rolling_window": rolling_window,
        "fdr_level": fdr_level,
    }


def screen_research_pairs(
    symbol_data,
    rolling_window=PAIR_DIAGNOSTIC_ROLLING_WINDOW,
    min_observations=None,
    fdr_level=0.05,
):
    """Build a Research-only matrix and run deterministic pair screening."""
    research_prices = build_research_price_matrix(symbol_data)
    result = screen_pair_diagnostics(
        research_prices,
        rolling_window=rolling_window,
        min_observations=min_observations,
        fdr_level=fdr_level,
    )
    result["research_prices"] = research_prices
    return result


def freeze_research_candidates(research_diagnostics):
    """Freeze the cohort using only the pre-specified Research raw-p rule."""
    required = {
        "symbol_1",
        "symbol_2",
        "cointegration_p_value",
        "cointegration_q_value",
    }
    if not isinstance(research_diagnostics, pd.DataFrame):
        raise TypeError("research_diagnostics must be a pandas DataFrame")
    missing = required.difference(research_diagnostics.columns)
    if missing:
        raise ValueError(f"research diagnostics are missing columns: {sorted(missing)}")
    if research_diagnostics[["symbol_1", "symbol_2"]].duplicated().any():
        raise ValueError("research diagnostics contain duplicate pairs")

    candidates = research_diagnostics.loc[
        research_diagnostics["cointegration_p_value"]
        < RESEARCH_CANDIDATE_P_THRESHOLD,
        [
            "symbol_1",
            "symbol_2",
            "cointegration_p_value",
            "cointegration_q_value",
        ],
    ].copy()
    candidates = candidates.rename(columns={
        "cointegration_p_value": "research_cointegration_p_value",
        "cointegration_q_value": "research_cointegration_q_value",
    })
    return candidates.sort_values(["symbol_1", "symbol_2"]).reset_index(drop=True)


def select_validation_advancing_pairs(validation_diagnostics):
    """Apply the sole frozen advancement rule: Validation BH q < 0.05."""
    if "validation_cointegration_q_value" not in validation_diagnostics:
        raise ValueError("validation diagnostics are missing BH q-values")
    return validation_diagnostics.loc[
        validation_diagnostics["validation_cointegration_q_value"]
        < VALIDATION_FDR_LEVEL
    ].copy()


def validate_frozen_pair_candidates(
    symbol_data,
    research_diagnostics,
    rolling_window=PAIR_DIAGNOSTIC_ROLLING_WINDOW,
    min_observations=None,
):
    """Replicate the frozen Research cohort using Validation observations only."""
    if min_observations is None:
        min_observations = rolling_window
    frozen = freeze_research_candidates(research_diagnostics)
    validation_prices = build_validation_price_matrix(symbol_data)
    candidate_pairs = list(frozen[["symbol_1", "symbol_2"]].itertuples(
        index=False, name=None
    ))
    evaluated, skipped = _evaluate_pair_diagnostics(
        validation_prices,
        candidate_pairs,
        rolling_window=rolling_window,
        min_observations=min_observations,
    )

    validation_names = {
        column: f"validation_{column}"
        for column in evaluated.columns
        if column not in PAIR_COLUMNS
    }
    evaluated = evaluated.rename(columns=validation_names)
    diagnostics = frozen.merge(
        evaluated, on=list(PAIR_COLUMNS), how="inner", validate="one_to_one"
    )
    correction = benjamini_hochberg(
        diagnostics["validation_cointegration_p_value"],
        alpha=VALIDATION_FDR_LEVEL,
    )
    diagnostics["validation_cointegration_q_value"] = correction["q_value"]
    diagnostics["validation_fdr_significant_05"] = correction["fdr_significant"]
    advancing = select_validation_advancing_pairs(diagnostics)

    return {
        "frozen_candidates": frozen,
        "diagnostics": diagnostics,
        "advancing_pairs": advancing,
        "skipped": skipped,
        "validation_prices": validation_prices,
        "rolling_window": rolling_window,
        "fdr_level": VALIDATION_FDR_LEVEL,
    }

def run_pairs_trading(data, alpha, beta, window, threshold):

    data_spread = calculate_spread(data, beta, alpha)

    zscore = calculate_zscore(data_spread, window, column="spread")

    data_spread["zscore"] = zscore

    positions = generate_pair_positions(zscore, threshold, beta)

    weights = calculate_pair_weights(data_spread, positions)

    data_spread["position_1"] = positions["position_1"]
    data_spread["position_2"] = positions["position_2"]

    data_spread["weight_1"] = weights["symbol_1"]
    data_spread["weight_2"] = weights["symbol_2"]

    result = calculate_pair_returns(data_spread, weights)

    equity = calculate_equity_curve(result["strategy_return"])

    result["equity"] = equity

    return result

def run_pairs_trading_with_history(data, historical_data, alpha, beta, window, threshold):

    historical_spread = calculate_spread(historical_data, beta, alpha)

    data_spread = calculate_spread(data, beta, alpha)

    zscore = calculate_zscore_with_history(historical_spread, data_spread, window, column="spread")

    data_spread["zscore"] = zscore

    positions = generate_pair_positions(zscore, threshold, beta)

    weights = calculate_pair_weights(data_spread, positions)

    data_spread["position_1"] = positions["position_1"]
    data_spread["position_2"] = positions["position_2"]

    data_spread["weight_1"] = weights["symbol_1"]
    data_spread["weight_2"] = weights["symbol_2"]

    result = calculate_pair_returns(data_spread, weights)

    result["equity"] = calculate_equity_curve(result["strategy_return"])

    return result

def generate_pairs(symbols):

    return list(combinations(symbols, 2))

def screen_pairs(train_prices, candidate_pairs):

    results = []

    for symbol_1, symbol_2 in candidate_pairs:

        pair_data = pd.DataFrame({
            "symbol_1": train_prices[symbol_1],
            "symbol_2": train_prices[symbol_2]
        })

        hedge_ratio = estimate_hedge_ratio(pair_data)

        alpha = hedge_ratio["alpha"]
        beta = hedge_ratio["beta"]

        spread_data = calculate_spread(
            pair_data,
            beta,
            alpha
        )

        cointegration = check_cointegration(pair_data)

        stationarity = check_spread_stationarity(
            spread_data["spread"]
        )

        results.append({
            "symbol_1": symbol_1,
            "symbol_2": symbol_2,
            "alpha": alpha,
            "beta": beta,
            "coint_pvalue": cointegration["p_value"],
            "adf_pvalue": stationarity["p_value"],
            "is_cointegrated": cointegration["is_cointegrated"],
            "is_stationary": stationarity["is_stationary"]
        })

    return pd.DataFrame(results)

def select_pairs(screening_results):

    selected = screening_results[
        screening_results["is_cointegrated"]
    ]

    selected = selected.sort_values(
        "coint_pvalue",
        ascending=True
    )

    return selected

def run_best_pair_validation(train_prices, validation_prices, selected_pairs, window, threshold):

    best_pair = selected_pairs.iloc[0]

    symbol_1 = best_pair["symbol_1"]
    symbol_2 = best_pair["symbol_2"]

    alpha = best_pair["alpha"]
    beta = best_pair["beta"]

    pair_train = pd.DataFrame({
        "symbol_1": train_prices[symbol_1],
        "symbol_2": train_prices[symbol_2]
    })

    pair_validation = pd.DataFrame({
        "symbol_1": validation_prices[symbol_1],
        "symbol_2": validation_prices[symbol_2]
    })

    return run_pairs_trading_with_history(pair_validation, pair_train, alpha, beta, window, threshold)

