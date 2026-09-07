import pandas as pd

from src.data_loader import load_market_data
from src.research.common import split_data
from src.research.pairs import (screen_pairs, select_pairs)
from src.research.validation import (run_validation_comparison, walk_forward_mean_reversion,
                                     walk_forward_pca, calculate_pca_stability, 
                                     calculate_pca_stability_to_initial, calculate_signal_disagreement)
from src.research.mean_reversion import run_mean_reversion
from src.research.pca import fit_pca, run_pca_stat_arb
from src.metrics import performance_summary

def main():

    pca_symbols = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "NVDA"
    ]

    candidate_pairs = [
        ("KO", "PEP"),
        ("XOM", "CVX"),
        ("JPM", "BAC"),
        ("V", "MA")
    ]

    mean_symbol = "AAPL"

    pair_symbols = [
        symbol
        for pair in candidate_pairs
        for symbol in pair
    ]

    all_symbols = sorted(
        set(
            pca_symbols
            + pair_symbols
            + [mean_symbol]
        )
    )

    start = "2021-01-01"
    end = "2026-01-01"

    prices = pd.DataFrame()

    for symbol in all_symbols:

        data = load_market_data(
            symbol,
            start,
            end
        )

        prices[symbol] = data["Close"]

    prices = prices.dropna()

    print("Price data:")
    print(prices.head())

    print("\nNumber of observations:")
    print(len(prices))

    # -------------------------
    # 2. Train / validation / test split
    # -------------------------

    train_prices, validation_prices, test_prices = split_data(
        prices,
        train_ratio=0.6,
        validation_ratio=0.2
    )

    print("\nData split:")
    print("Train:", len(train_prices))
    print("Validation:", len(validation_prices))
    print("Test:", len(test_prices))

    # -------------------------
    # 3. Baseline parameters
    # -------------------------

    window = 20
    threshold = 2.0
    n_components = 2

    # -------------------------
    # Week 7 Day 1
    # Walk-forward sanity check
    # -------------------------

    mean_data = pd.DataFrame({
        "Close": prices[mean_symbol]
    }).dropna()

    initial_train_size = 252
    test_size = 21

    walk_forward_result = walk_forward_mean_reversion(
        mean_data,
        initial_train_size=initial_train_size,
        test_size=test_size,
        window=window,
        threshold=threshold
    )

    walk_forward_summary = performance_summary(
        walk_forward_result["strategy_return"]
    )

    print("\n=== WALK-FORWARD MEAN REVERSION ===")
    print(walk_forward_summary)

    full_result = run_mean_reversion(
        mean_data.copy(),
        window=window,
        threshold=threshold
    )

    continuous_oos = full_result.iloc[initial_train_size:].copy()

    continuous_summary = performance_summary(
        continuous_oos["strategy_return"]
    )

    print("\n=== CONTINUOUS MEAN REVERSION ===")
    print(continuous_summary)

    comparison = pd.DataFrame({
        "walk_forward": walk_forward_result["strategy_return"],
        "continuous": continuous_oos["strategy_return"]
    })

    comparison["difference"] = (
        comparison["walk_forward"]
        - comparison["continuous"]
    )

    print("\n=== WALK-FORWARD SANITY CHECK ===")
    print(comparison.head())

    print(
        "\nMaximum return difference:",
        comparison["difference"].abs().max()
    )
    # -------------------------
    # Week 7 Day 1
    # Walk-forward PCA
    # -------------------------

    pca_prices = prices[pca_symbols].copy()

    walk_forward_pca_result = walk_forward_pca(
        pca_prices,
        initial_train_size=252,
        test_size=21,
        n_components=n_components,
        window=window,
        threshold=threshold
    )

    walk_forward_pca_summary = performance_summary(
        walk_forward_pca_result["strategy_return"]
    )

    print("\n=== WALK-FORWARD PCA ===")
    # -------------------------
    # Fixed PCA OOS benchmark
    # -------------------------

    pca_prices = prices[pca_symbols].copy()

    initial_train_prices = pca_prices.iloc[:initial_train_size]

    initial_train_returns = (
        initial_train_prices
        .pct_change()
        .dropna()
    )

    fixed_pca = fit_pca(
        initial_train_returns,
        n_components
    )

    # Use exactly the same OOS dates as walk-forward PCA
    oos_index = walk_forward_pca_result["strategy_return"].index

    oos_prices = pca_prices.loc[oos_index]

    combined_prices = pd.concat([
        initial_train_prices.iloc[-1:],
        oos_prices
    ])

    oos_returns = (
        combined_prices
        .pct_change()
        .iloc[1:]
    )

    # Last train return provides position continuity
    context_return = initial_train_returns.iloc[-1:]

    historical_returns = initial_train_returns.iloc[:-1]

    data_with_context = pd.concat([
        context_return,
        oos_returns
    ])

    fixed_pca_result = run_pca_stat_arb(
        data=data_with_context,
        historical_data=historical_returns,
        pca=fixed_pca,
        window=window,
        threshold=threshold
    )

    fixed_pca_strategy_returns = (
        fixed_pca_result["strategy_return"]
        .loc[oos_index]
    )

    fixed_pca_summary = performance_summary(
        fixed_pca_strategy_returns
    )

    print("\n=== FIXED PCA OOS ===")
    print(fixed_pca_summary)

    print("\n=== REFITTED PCA OOS ===")
    print(walk_forward_pca_summary)
    print(walk_forward_pca_summary)

    print(
        "\nNumber of PCA walk-forward folds:",
        len(walk_forward_pca_result["folds"])
    )

    print(
        "Number of PCA OOS observations:",
        len(walk_forward_pca_result["strategy_return"])
    )

    stability = calculate_pca_stability(
        walk_forward_pca_result["folds"]
    )

    print("\n=== PCA STABILITY ===")
    print(stability)

    print("\n=== PCA STABILITY SUMMARY ===")
    print(
        stability[
            ["pc1_similarity", "pc2_similarity"]
        ].describe()
    )

    initial_stability = calculate_pca_stability_to_initial(
        walk_forward_pca_result["folds"]
    )

    print("\n=== PCA STABILITY TO INITIAL ===")
    print(initial_stability)

    print("\n=== PCA STABILITY TO INITIAL SUMMARY ===")
    print(
        initial_stability[
            [
                "pc1_similarity_to_initial",
                "pc2_similarity_to_initial"
            ]
        ].describe()
    )

    refitted_signals = pd.concat([
        fold["signals"]
        for fold in walk_forward_pca_result["folds"]
    ])

    fixed_signals = (
        fixed_pca_result["signals"]
        .loc[refitted_signals.index]
    )

    signal_comparison = calculate_signal_disagreement(
        fixed_signals,
        refitted_signals
    )

    print("\n=== PCA SIGNAL DISAGREEMENT ===")

    print(
        "Overall disagreement rate:",
        signal_comparison["overall_disagreement_rate"]
    )

    print(
        "Daily disagreement rate:",
        signal_comparison["daily_disagreement_rate"]
    )

    print("\nPer-symbol disagreement:")
    print(
        signal_comparison["per_symbol_disagreement"]
    )

    refitted_positions = pd.concat([
        fold["positions"]
        for fold in walk_forward_pca_result["folds"]
    ])

    fixed_positions = (
        fixed_pca_result["positions"]
        .loc[refitted_positions.index]
    )

    position_difference = (
        fixed_positions - refitted_positions
    ).abs()

    daily_position_distance = (
        position_difference.sum(axis=1)
    )

    print("\n=== PCA POSITION DIFFERENCE ===")

    print(
        daily_position_distance.describe()
    )

    print(
        "\nFraction of days with different positions:",
        (daily_position_distance > 0).mean()
    )

    fixed_returns = fixed_pca_strategy_returns

    refitted_returns = (
        walk_forward_pca_result["strategy_return"]
    )

    pnl_comparison = pd.DataFrame({
        "fixed": fixed_returns,
        "refitted": refitted_returns
    })

    pnl_comparison["difference"] = (
        pnl_comparison["fixed"]
        - pnl_comparison["refitted"]
    )

    print("\n=== DAILY PNL DIFFERENCE ===")

    print(
        pnl_comparison["difference"].describe()
    )

    signal_disagreement_day = (
        signal_comparison["disagreement"]
        .any(axis=1)
    )

    affected_pnl_day = (
        signal_disagreement_day
        .shift(1, fill_value=False)
        .astype(bool)
    )

    affected_pnl_day = affected_pnl_day.reindex(
        pnl_comparison.index,
        fill_value=False
    )

    pnl_on_affected_days = pnl_comparison.loc[
        affected_pnl_day,
        "difference"
    ]

    pnl_on_normal_days = pnl_comparison.loc[
        ~affected_pnl_day,
        "difference"
    ]

    print("\n=== PNL ATTRIBUTION ===")

    print(
        "Affected days:",
        len(pnl_on_affected_days)
    )

    print(
        "Mean PnL difference on affected days:",
        pnl_on_affected_days.mean()
    )

    print(
        "Total PnL difference on affected days:",
        pnl_on_affected_days.sum()
    )

    print(
        "Mean PnL difference on normal days:",
        pnl_on_normal_days.mean()
    )

    print(
        "Total PnL difference on normal days:",
        pnl_on_normal_days.sum()
    )

    # -------------------------
    # 4. Run validation experiment
    # -------------------------

    screening = screen_pairs(
    train_prices,
    candidate_pairs
)

    print("\n=== TRAIN PAIR SCREENING ===")
    print(screening[[
                "symbol_1",
                "symbol_2",
                "coint_pvalue",
                "adf_pvalue",
                "is_cointegrated",
                "is_stationary"
            ]].sort_values("coint_pvalue"))

    selected = select_pairs(screening)

    print("\n=== SELECTED PAIRS ===")
    print(selected)

    result = run_validation_comparison(
    train_prices=train_prices,
    validation_prices=validation_prices,
    mean_symbol=mean_symbol,
    candidate_pairs=candidate_pairs,
    pca_symbols=pca_symbols,
    window=20,
    threshold=2.0,
    n_components=2
)

    # -------------------------
    # 5. Results
    # -------------------------

    print("\n=== TRAIN PAIR SCREENING ===")

    print(
        screening[
            [
                "symbol_1",
                "symbol_2",
                "coint_pvalue",
                "adf_pvalue",
                "is_cointegrated",
                "is_stationary"
            ]
        ].sort_values("coint_pvalue")
    )

    print("\n=== Selected Pairs ===")
    print(
        result["selected_pairs"][
            [
                "symbol_1",
                "symbol_2",
                "alpha",
                "beta",
                "coint_pvalue",
                "adf_pvalue"
            ]
        ]
    )

    print("\n=== Validation Performance ===")
    print(result["comparison"])


if __name__ == "__main__":
    main()