import pandas as pd

from src.research.mean_reversion import run_mean_reversion_with_history
from src.research.pairs import (screen_pairs,select_pairs,run_best_pair_validation,)
from .pca import (fit_pca, run_pca_stat_arb)
from ..metrics import performance_summary

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