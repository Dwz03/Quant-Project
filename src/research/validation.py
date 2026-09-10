import pandas as pd
import numpy as np

from src.research.mean_reversion import run_mean_reversion_with_history, run_mean_reversion
from src.research.pairs import (screen_pairs,select_pairs,run_best_pair_validation,)
from .pca import (fit_pca, run_pca_stat_arb)
from ..metrics import performance_summary
from .common import calculate_equity_curve

def calculate_pca_stability(folds):

    results = []

    for i in range(1, len(folds)):

        previous_components = folds[i - 1]["pca_components"]
        current_components = folds[i]["pca_components"]

        similarities = []

        for component_index in range(previous_components.shape[0]):

            previous_component = previous_components[component_index]
            current_component = current_components[component_index]

            similarity = abs(np.dot(previous_component,current_component))

            similarities.append(similarity)

        results.append({
            "fold": i,
            "pc1_similarity": similarities[0],
            "pc2_similarity": similarities[1]
        })

    return pd.DataFrame(results)

def calculate_pca_stability_to_initial(folds):

    results = []

    initial_components = folds[0]["pca_components"]

    for i in range(len(folds)):

        current_components = folds[i]["pca_components"]

        similarities = []

        for component_index in range(initial_components.shape[0]):

            initial_component = initial_components[component_index]

            current_component = current_components[component_index]

            similarity = abs(np.dot(initial_component,current_component))

            similarities.append(similarity)

        results.append({
            "fold": i,
            "pc1_similarity_to_initial": similarities[0],
            "pc2_similarity_to_initial": similarities[1]
        })

    return pd.DataFrame(results)

def walk_forward_split(data, initial_train_size, test_size):

    if not isinstance(initial_train_size, int):
        raise TypeError("initial_train_size must be an integer")

    if not isinstance(test_size, int):
        raise TypeError("test_size must be an integer")

    if initial_train_size <= 0:
        raise ValueError("initial_train_size must be positive")

    if test_size <= 0:
        raise ValueError("test_size must be positive")

    if initial_train_size + test_size > len(data):
        raise ValueError("not enough data for the first walk-forward split")

    train_end = initial_train_size

    while train_end + test_size <= len(data):

        test_end = train_end + test_size

        train_data = data.iloc[:train_end]
        test_data = data.iloc[train_end:test_end]

        yield train_data, test_data

        train_end += test_size

def combine_history_and_test(train_data, test_data, window):

    history = train_data.iloc[-window:]

    combined_data = pd.concat([history, test_data])

    return combined_data

def walk_forward_mean_reversion(data, initial_train_size, test_size, window, threshold):

    results = []

    for train_data, test_data in walk_forward_split(data, initial_train_size, test_size):

        combined_data = combine_history_and_test(train_data, test_data, window)

        result = run_mean_reversion(combined_data, window, threshold)

        test_result = result.loc[test_data.index].copy()

        results.append(test_result)

    walk_forward_result = pd.concat(results)

    walk_forward_result["equity"] = (1 + walk_forward_result["strategy_return"]).cumprod()

    return walk_forward_result

def calculate_signal_disagreement(fixed_signals, refitted_signals):

    fixed = fixed_signals.loc[refitted_signals.index, refitted_signals.columns]

    disagreement = fixed != refitted_signals

    overall_disagreement_rate = (disagreement.sum().sum() / disagreement.size)

    daily_disagreement_rate = (disagreement.any(axis=1).mean())

    per_symbol_disagreement = disagreement.mean()

    return {
        "overall_disagreement_rate": overall_disagreement_rate,
        "daily_disagreement_rate": daily_disagreement_rate,
        "per_symbol_disagreement": per_symbol_disagreement,
        "disagreement": disagreement
    }

def walk_forward_pca(prices, initial_train_size, test_size, n_components, window, threshold):

    results = []

    for train_prices, test_prices in walk_forward_split(prices, initial_train_size, test_size):

        # 1. Train returns
        train_returns = train_prices.pct_change().dropna()

        # 2. Fit PCA using train only
        pca = fit_pca(train_returns, n_components)

        # 3. Construct test returns
        combined_prices = pd.concat([ train_prices.iloc[-1:], test_prices])

        test_returns = (combined_prices.pct_change().iloc[1:])

        # 4. Keep the last train return as context
        context_return = train_returns.iloc[-1:]

        historical_returns = train_returns.iloc[:-1]

        data_with_context = pd.concat([context_return, test_returns])

        # 5. Run PCA strategy
        fold_result = run_pca_stat_arb(data=data_with_context, historical_data=historical_returns, pca=pca,
            window=window, threshold=threshold)

        # 6. Keep OOS test only
        test_index = test_returns.index

        test_result = {
            "strategy_return":
                fold_result["strategy_return"].loc[test_index],

            "equity":
                fold_result["equity"].loc[test_index],

            "positions":
                fold_result["positions"].loc[test_index],

            "signals": 
                fold_result["signals"].loc[test_index],

            "zscores":
                fold_result["zscores"].loc[test_index],

            "pca_components":
                pca.components_.copy(),

            "explained_variance_ratio":
                pca.explained_variance_ratio_.copy()
        }

        results.append(test_result)

    strategy_returns = pd.concat([result["strategy_return"] for result in results])

    equity = calculate_equity_curve(strategy_returns)

    return {
        "strategy_return": strategy_returns,
        "equity": equity,
        "folds": results
    }

def compare_strategies(strategy_returns):

    results = {}

    for strategy_name, returns in strategy_returns.items():

        results[strategy_name] = performance_summary(returns)

    return pd.DataFrame(results).T

def build_strategy_comparison(mean_result, pair_result, pca_result):

    strategy_returns = {
        "mean_reversion": mean_result["strategy_return"],
        "pca": pca_result["strategy_return"]
    }

    if pair_result is not None:
        strategy_returns["pairs"] = pair_result["strategy_return"]

    return compare_strategies(strategy_returns)

def run_validation_comparison(train_prices,validation_prices,mean_symbol,candidate_pairs,pca_symbols,
    window,threshold,n_components):

    # 1. Pair screening using TRAIN only
    screening_results = screen_pairs(train_prices,candidate_pairs)

    selected_pairs = select_pairs(screening_results)

    if selected_pairs.empty:

        pair_result = None

    else:

        pair_result = run_best_pair_validation(train_prices,validation_prices,selected_pairs,window,threshold)

    # 2. Mean reversion
    mean_train = pd.DataFrame({"Close": train_prices[mean_symbol]})

    mean_validation = pd.DataFrame({"Close": validation_prices[mean_symbol]})

    mean_result = run_mean_reversion_with_history(mean_validation,mean_train,window,threshold)

    # 4. PCA train returns
    pca_train_returns = (train_prices[pca_symbols].pct_change().dropna())

    pca = fit_pca(pca_train_returns,n_components)

    # 5. PCA validation returns
    combined_prices = pd.concat([train_prices[pca_symbols].iloc[-1:],validation_prices[pca_symbols]])

    pca_validation_returns = (combined_prices.pct_change().iloc[1:])

    # 6. PCA strategy
    pca_result = run_pca_stat_arb(pca_validation_returns,pca_train_returns,pca,window,threshold)

    # 7. Comparison
    comparison = build_strategy_comparison(mean_result,pair_result,pca_result)

    return {
        "comparison": comparison,
        "mean_result": mean_result,
        "pair_result": pair_result,
        "pca_result": pca_result,
        "screening_results": screening_results,
        "selected_pairs": selected_pairs
    }

def tune_mean_reversion(train_data, validation_data, windows, thresholds):

    results = []

    for window in windows:

        for threshold in thresholds:

            result = run_mean_reversion_with_history(validation_data,train_data,window,threshold)

            summary = performance_summary(result["strategy_return"])

            results.append({
                "window": window,
                "threshold": threshold,
                **summary
            })

    return pd.DataFrame(results)

def select_best_mean_reversion_parameters(tuning_results):

    best_index = tuning_results["Sharpe Ratio"].idxmax()

    best_row = tuning_results.loc[best_index]

    return {
        "window": int(best_row["window"]),
        "threshold": float(best_row["threshold"])
    }

def evaluate_mean_reversion_on_test(train_data,validation_data,test_data,best_parameters):

    historical_data = pd.concat([train_data,validation_data])

    result = run_mean_reversion_with_history(test_data,historical_data,best_parameters["window"],
                                             best_parameters["threshold"])

    return result

def run_mean_reversion_research(train_data,validation_data,test_data,windows,thresholds):

    tuning_results = tune_mean_reversion(train_data,validation_data,windows,thresholds)

    best_parameters = select_best_mean_reversion_parameters(tuning_results)

    test_result = evaluate_mean_reversion_on_test(train_data,validation_data,test_data,best_parameters)

    test_performance = performance_summary(test_result["strategy_return"])

    return {
        "tuning_results": tuning_results,
        "best_parameters": best_parameters,
        "test_result": test_result,
        "test_performance": test_performance
    }

def build_parameter_surface(tuning_results, metric="Sharpe Ratio"):

    return tuning_results.pivot(index="window", columns="threshold", values=metric)

def calculate_local_robustness(tuning_results,metric="Sharpe Ratio"):

    surface = build_parameter_surface(tuning_results, metric)

    row, col = np.unravel_index(np.nanargmax(surface.values), surface.shape)

    row_start = max(0, row - 1)
    row_end = min(surface.shape[0], row + 2)

    col_start = max(0, col - 1)
    col_end = min(surface.shape[1], col + 2)

    neighborhood = surface.iloc[row_start:row_end, col_start:col_end].copy()

    best_value = surface.iloc[row, col]

    # best point 在 neighborhood 里的相对位置
    local_row = row - row_start
    local_col = col - col_start

    neighbor_values = neighborhood.values.copy()
    neighbor_values[local_row, local_col] = np.nan

    neighbor_values = neighbor_values[~np.isnan(neighbor_values)]

    return {
        "best_window": int(surface.index[row]),
        "best_threshold": float(surface.columns[col]),
        "best_metric": best_value,
        "neighbor_mean": neighbor_values.mean(),
        "neighbor_std": neighbor_values.std(),
        "performance_drop": best_value - neighbor_values.mean()
    }

def is_parameter_robust(robustness_result, max_performance_drop=0.3, max_neighbor_std=0.3):

    return bool(
        robustness_result["performance_drop"] <= max_performance_drop
        and
        robustness_result["neighbor_std"] <= max_neighbor_std
    )

def summarize_parameter_robustness(tuning_results, metric="Sharpe Ratio", max_performance_drop=0.3,
                                    max_neighbor_std=0.3):

    robustness = calculate_local_robustness(tuning_results, metric)

    robustness["robust"] = is_parameter_robust(robustness, max_performance_drop, max_neighbor_std)

    return robustness

def calculate_turnover(positions, initial_positions=None):

    if positions.empty:
        raise ValueError("positions cannot be empty")

    positions_used = positions.fillna(0)

    turnover = positions_used.diff().abs()

    if isinstance(positions_used, pd.DataFrame):
        if initial_positions is None:
            initial = pd.Series(0.0, index=positions_used.columns)
        else:
            if not isinstance(initial_positions, pd.Series):
                raise TypeError("initial_positions must be a Series for DataFrame positions")
            if set(initial_positions.index) != set(positions_used.columns):
                raise ValueError("initial_positions must contain exactly the position columns")
            initial = initial_positions.reindex(positions_used.columns)

        turnover.iloc[0] = (positions_used.iloc[0] - initial).abs()
    else:
        if isinstance(initial_positions, pd.Series):
            raise TypeError("initial_positions must be scalar for Series positions")
        initial = 0.0 if initial_positions is None else float(initial_positions)
        turnover.iloc[0] = abs(positions_used.iloc[0] - initial)

    if isinstance(turnover, pd.DataFrame):
        turnover = turnover.sum(axis=1)

    return turnover


def apply_transaction_costs(
    strategy_returns,
    positions,
    cost_rate,
    initial_positions=None,
):

    if cost_rate < 0:
        raise ValueError("cost_rate must be non-negative")

    turnover = calculate_turnover(positions, initial_positions=initial_positions)

    costs = turnover * cost_rate

    net_returns = strategy_returns - costs

    result = pd.DataFrame({
        "strategy_return": strategy_returns,
        "turnover": turnover,
        "cost": costs,
        "net_strategy_return": net_returns
    })

    if isinstance(positions, pd.Series):
        result.insert(1, "position", positions)

    return result

def run_cost_sensitivity(strategy_returns, positions, cost_rates):

    results = []

    for cost_rate in cost_rates:

        net_result = apply_transaction_costs(strategy_returns, positions, cost_rate)

        summary = performance_summary(net_result["net_strategy_return"])

        results.append({"cost_rate": cost_rate, **summary})

    return pd.DataFrame(results)

def find_break_even_cost(cost_sensitivity_results):

    unprofitable = cost_sensitivity_results[cost_sensitivity_results["Total Return"] <= 0]

    if unprofitable.empty:
        return None

    return unprofitable["cost_rate"].min()

def classify_volatility_regime(returns,window,volatility_threshold):

    rolling_volatility = (returns.rolling(window).std().shift(1))

    regime = pd.Series(index=returns.index, dtype="object")

    regime[rolling_volatility <= volatility_threshold] = "calm"
    regime[rolling_volatility > volatility_threshold] = "volatile"

    return pd.DataFrame({
        "returns": returns,
        "rolling_volatility": rolling_volatility,
        "regime": regime
    })

def compare_performance_by_regime(strategy_returns, regimes):

    results = {}

    for regime_name in ["calm", "volatile"]:

        mask = regimes == regime_name

        regime_returns = strategy_returns[mask]

        if regime_returns.empty:
            continue

        results[regime_name] = performance_summary(regime_returns)

    return pd.DataFrame(results).T

def run_regime_analysis(market_returns, strategy_returns, window, volatility_threshold):

    regime_result = classify_volatility_regime(market_returns, window, volatility_threshold)

    comparison = compare_performance_by_regime(strategy_returns, regime_result["regime"])

    return {
        "regime_result": regime_result,
        "comparison": comparison
    }

def attribute_performance_by_position(strategy_returns, positions):

    data = pd.DataFrame({
        "strategy_return": strategy_returns,
        "position": positions
    })

    data["position_bucket"] = data["position"].map({
        1: "long",
        0: "flat",
        -1: "short"
    })

    result = (data.groupby("position_bucket")["strategy_return"].agg(["count", "sum", "mean"]))

    return result

def attribute_performance_by_month(strategy_returns):

    data = pd.DataFrame({"strategy_return": strategy_returns})

    data["month"] = data.index.to_period("M")

    result = (data.groupby("month")["strategy_return"].agg(["count", "sum", "mean"]))

    return result

def attribute_performance_by_asset(asset_strategy_returns):

    results = []

    for symbol in asset_strategy_returns.columns:

        returns = asset_strategy_returns[symbol]

        results.append({
            "symbol": symbol,
            "count": returns.count(),
            "sum": returns.sum(),
            "mean": returns.mean()
        })

    return pd.DataFrame(results).set_index("symbol")

def run_performance_attribution(strategy_returns, positions, asset_strategy_returns=None):

    position_attribution = attribute_performance_by_position(strategy_returns, positions)

    monthly_attribution = attribute_performance_by_month(strategy_returns)

    if asset_strategy_returns is not None:

        asset_attribution = attribute_performance_by_asset(asset_strategy_returns)

    else:
        asset_attribution = None

    return {
        "position_attribution": position_attribution,
        "monthly_attribution": monthly_attribution,
        "asset_attribution": asset_attribution
    }
