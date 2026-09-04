import pandas as pd

from src.data_loader import load_market_data
from src.research.common import split_data
from src.research.validation import run_validation_comparison
from src.research.pairs import (screen_pairs, select_pairs)


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