from .strategy import (
    MomentumTradingStrategy,
    MovingAverageTradingStrategy,
    MeanReversionTradingStrategy,
    PairsTradingStrategy,
    PCAResidualTradingStrategy,
)


def build_strategy(
    strategy_name,
    symbols=None,
    config=None
):

    if config is None:
        config = {}

    strategy_name = (
        strategy_name
        .strip()
        .lower()
    )

    symbols = (
        []
        if symbols is None
        else symbols
    )


    if strategy_name == "momentum":

        return MomentumTradingStrategy(

            lookback=config.get(
                "lookback",
                20
            ),

            target_weight=config.get(
                "target_weight",
                0.01
            ),

            allow_short=config.get(
                "allow_short",
                False
            )
        )


    elif strategy_name in (
        "moving_average",
        "ma"
    ):

        return MovingAverageTradingStrategy(
            short_window=10,
            long_window=30,
            target_weight=0.01,
            allow_short=False
        )


    elif strategy_name in (
        "mean_reversion",
        "mr"
    ):

        return MeanReversionTradingStrategy(
            window=20,
            target_weight=0.01,
            max_gross_exposure=0.03,
            allow_short=False
        )


    elif strategy_name == "pairs":

        if len(symbols) != 2:

            raise ValueError(
                "pairs strategy requires "
                "exactly two symbols"
            )

        return PairsTradingStrategy(
            symbol_1=symbols[0],
            symbol_2=symbols[1],
            window=20,
            threshold=1.5,

            # First paper deployment
            target_gross_exposure=0.01
        )


    elif strategy_name in (
        "pca",
        "pca_residual"
    ):

        if len(symbols) < 2:

            raise ValueError(
                "PCA strategy requires "
                "at least two symbols"
            )

        return PCAResidualTradingStrategy(
            n_components=1,
            window=20,
            threshold=1.5,
            target_gross_exposure=0.01
        )


    else:

        raise ValueError(
            f"unknown strategy: "
            f"{strategy_name}"
        )