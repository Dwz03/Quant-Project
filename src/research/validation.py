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

            similarity = abs(
                np.dot(
                    previous_component,
                    current_component
                )
            )

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

            similarity = abs(
                np.dot(
                    initial_component,
                    current_component
                )
            )

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

    fixed = fixed_signals.loc[
        refitted_signals.index,
        refitted_signals.columns
    ]

    disagreement = fixed != refitted_signals

    overall_disagreement_rate = (
        disagreement.sum().sum()
        / disagreement.size
    )

    daily_disagreement_rate = (
        disagreement.any(axis=1).mean()
    )

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
    screening_results = screen_pairs(
        train_prices,
        candidate_pairs
    )

    selected_pairs = select_pairs(
        screening_results
    )

    if selected_pairs.empty:

        pair_result = None

    else:

        pair_result = run_best_pair_validation(
            train_prices,
            validation_prices,
            selected_pairs,
            window,
            threshold
        )

    # 2. Mean reversion
    mean_train = pd.DataFrame({
        "Close": train_prices[mean_symbol]
    })

    mean_validation = pd.DataFrame({
        "Close": validation_prices[mean_symbol]
    })

    mean_result = run_mean_reversion_with_history(
        mean_validation,
        mean_train,
        window,
        threshold
    )

    # 4. PCA train returns
    pca_train_returns = (
        train_prices[pca_symbols]
        .pct_change()
        .dropna()
    )

    pca = fit_pca(
        pca_train_returns,
        n_components
    )

    # 5. PCA validation returns
    combined_prices = pd.concat([
        train_prices[pca_symbols].iloc[-1:],
        validation_prices[pca_symbols]
    ])

    pca_validation_returns = (
        combined_prices
        .pct_change()
        .iloc[1:]
    )

    # 6. PCA strategy
    pca_result = run_pca_stat_arb(
        pca_validation_returns,
        pca_train_returns,
        pca,
        window,
        threshold
    )

    # 7. Comparison
    comparison = build_strategy_comparison(
        mean_result,
        pair_result,
        pca_result
    )

    return {
        "comparison": comparison,
        "mean_result": mean_result,
        "pair_result": pair_result,
        "pca_result": pca_result,
        "screening_results": screening_results,
        "selected_pairs": selected_pairs
    }