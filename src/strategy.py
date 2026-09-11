# Import
from abc import ABC, abstractmethod
from numbers import Integral, Real
from .events import SignalEvent
from src.research.pairs import (
    estimate_hedge_ratio,
    calculate_spread,
    generate_pair_positions,
    calculate_pair_weights
)
from src.research.pca import (
    fit_pca,
    calculate_pca_residuals,
    calculate_residual_zscore_with_history,
    generate_residual_signals
)

from src.research.common import calculate_zscore, normalize_positions
import pandas as pd
import numpy as np

# Main script
class Strategy(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_signal(self, prices):
        pass

class TradingStrategy(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_target_weights(self, history):
        pass

class AlwaysBuyStrategy(Strategy):

    def __init__(self):
        super().__init__("AlwaysBuys")

    def generate_signal(self, prices):
        return "BUY"

class AlwaysSellStrategy(Strategy):

    def __init__(self):
        super().__init__("AlwaysSell")

    def generate_signal(self, prices):
        return "SELL"

class MomentumStrategy(Strategy):

    def __init__(self, lookback):

        super().__init__("Momentum")
        self.lookback = lookback
        self.last_prices = {}

    def generate_signal(self, prices):

        if len(prices) < 2:
            raise ValueError("Momentum Strategy require at least 2 prices")

        if prices[-1] > prices[-2]:
            return "BUY"
        elif prices[-1] == prices[-2]:
            return "HOLD"
        else:
            return "SELL"
    def on_market_event(self, event):

        symbol = event.symbol
        price = event.price

        if symbol not in self.last_prices:
            self.last_prices[symbol] = price
            return None

        previous_price = self.last_prices[symbol]

        if price > previous_price:
            signal = SignalEvent(symbol, "BUY")

        elif price < previous_price:
            signal = SignalEvent(symbol, "SELL")

        else:
            signal = None

        self.last_prices[symbol] = price

        return signal

class MeanReversionStrategy(Strategy):

    def __init__(self, window):
        super().__init__("MeanReversion")
        self.window = window

    def generate_signal(self, prices):

        if len(prices) < self.window + 1:
            raise ValueError(f"we need at least {self.window} prices to calculate the mean revision strategy")
        
        historical_prices = prices[-self.window - 1: -1]
        average_price = sum(historical_prices) / len(historical_prices)

        if prices[-1] < average_price:
            return "BUY"
        else:
            return "SELL"

class MeanReversionTradingStrategy(TradingStrategy):

    def __init__(
        self,
        window,
        target_weight,
        max_gross_exposure=1.0,
        allow_short=False
    ):
        super().__init__("MeanReversion")

        self.window = window
        self.target_weight = target_weight
        self.max_gross_exposure = max_gross_exposure
        self.allow_short = allow_short

    def generate_target_weights(self, history):

        target_weights = {}

        for symbol in history.columns:

            prices = history[symbol]

            if len(prices) < self.window + 1:
                continue

            historical_prices = prices.iloc[
                -self.window - 1:-1
            ]

            current_price = prices.iloc[-1]

            mean_price = historical_prices.mean()

            if current_price < mean_price:
                target_weights[symbol] = self.target_weight

            elif current_price > mean_price:

                if self.allow_short:
                    target_weights[symbol] = -self.target_weight
                else:
                    target_weights[symbol] = 0.0

            else:
                target_weights[symbol] = 0.0

        gross_exposure = sum(
            abs(weight)
            for weight in target_weights.values()
        )

        if gross_exposure > self.max_gross_exposure:

            scale = (
                self.max_gross_exposure
                / gross_exposure
            )

            target_weights = {
                symbol: weight * scale
                for symbol, weight in target_weights.items()
            }

        return target_weights
    
class MovingAverageStrategy(Strategy):

    def __init__(self, short_window, long_window):

        super().__init__("MovingAverage")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, data):

        data["Short_MA"] = data["Close"].rolling(window = self.short_window).mean()
        data["Long_MA"] = data["Close"].rolling(window = self.long_window).mean()

        data["Signal"] = 0

        data.loc[data["Short_MA"] > data["Long_MA"], "Signal"] = 1
        data.loc[data["Short_MA"] < data["Long_MA"], "Signal"] = -1

        return data

class PairsTradingStrategy(TradingStrategy):

    def __init__(
        self,
        symbol_1,
        symbol_2,
        window=20,
        threshold=1.5,
        target_gross_exposure=0.10
    ):

        super().__init__("Pairs Trading")

        self.symbol_1 = symbol_1
        self.symbol_2 = symbol_2
        self.window = window
        self.threshold = threshold
        self.target_gross_exposure = target_gross_exposure


    def generate_target_weights(self, history):

        required_symbols = {
            self.symbol_1,
            self.symbol_2
        }

        if not required_symbols.issubset(
            history.columns
        ):
            raise ValueError(
                "History is missing pair symbols"
            )

        if len(history) < self.window + 2:
            return {
                self.symbol_1: 0.0,
                self.symbol_2: 0.0
            }

        # -------------------------
        # 1. Fit hedge ratio using
        #    historical observations
        #    excluding current price
        # -------------------------

        training_history = history.iloc[:-1]

        train_pair = pd.DataFrame({
            "symbol_1":
                training_history[self.symbol_1],

            "symbol_2":
                training_history[self.symbol_2]
        })

        hedge_ratio = estimate_hedge_ratio(
            train_pair
        )

        alpha = hedge_ratio["alpha"]
        beta = hedge_ratio["beta"]

        # -------------------------
        # 2. Calculate current spread
        # -------------------------

        pair_data = pd.DataFrame({
            "symbol_1":
                history[self.symbol_1],

            "symbol_2":
                history[self.symbol_2]
        })

        spread_data = calculate_spread(
            pair_data,
            beta,
            alpha
        )

        # -------------------------
        # 3. Current z-score
        # -------------------------

        zscore = calculate_zscore(
            spread_data,
            self.window,
            column="spread"
        )

        current_zscore = zscore.iloc[-1]

        # -------------------------
        # 4. Generate pair position
        # -------------------------

        current_zscore_series = pd.Series(
            [current_zscore],
            index=[history.index[-1]]
        )

        positions = generate_pair_positions(
            current_zscore_series,
            self.threshold,
            beta
        )

        current_prices = pd.DataFrame({
            "symbol_1": [
                history[self.symbol_1].iloc[-1]
            ],
            "symbol_2": [
                history[self.symbol_2].iloc[-1]
            ]
        }, index=[history.index[-1]])

        weights = calculate_pair_weights(
            current_prices,
            positions
        )

        weight_1 = (
            weights["symbol_1"].iloc[-1]
            * self.target_gross_exposure
        )

        weight_2 = (
            weights["symbol_2"].iloc[-1]
            * self.target_gross_exposure
        )

        return {
            self.symbol_1: float(weight_1),
            self.symbol_2: float(weight_2)
        }

class PCAResidualTradingStrategy(
    TradingStrategy
):

    def __init__(
        self,
        n_components=1,
        window=20,
        threshold=1.5,
        target_gross_exposure=0.10
    ):

        super().__init__(
            "PCA Residual Stat Arb"
        )

        self.n_components = n_components
        self.window = window
        self.threshold = threshold
        self.target_gross_exposure = (
            target_gross_exposure
        )


    def generate_target_weights(
        self,
        history
    ):

        if history.empty:
            raise ValueError(
                "history cannot be empty"
            )

        symbols = list(
            history.columns
        )

        if (
            self.n_components
            >= len(symbols)
        ):
            raise ValueError(
                "n_components must be "
                "smaller than number of assets"
            )

        # --------------------------------
        # 1. Prices -> returns
        # --------------------------------

        returns = (
            history
            .pct_change()
            .dropna()
        )

        if len(returns) < self.window + 1:

            return {
                symbol: 0.0
                for symbol in symbols
            }

        # Current return must NOT be used
        # to fit PCA
        training_returns = (
            returns.iloc[:-1]
        )

        current_return = (
            returns.iloc[-1:]
        )

        # --------------------------------
        # 2. Fit PCA on historical data
        # --------------------------------

        pca = fit_pca(
            training_returns,
            self.n_components
        )

        # --------------------------------
        # 3. PCA residuals
        # --------------------------------

        historical_residuals = (
            calculate_pca_residuals(
                training_returns,
                pca
            )
        )

        current_residual = (
            calculate_pca_residuals(
                current_return,
                pca
            )
        )

        # --------------------------------
        # 4. Residual z-score
        # --------------------------------

        zscores = (
            calculate_residual_zscore_with_history(
                historical_residuals,
                current_residual,
                self.window
            )
        )

        # --------------------------------
        # 5. Mean-reversion signal
        # --------------------------------

        signals = (
            generate_residual_signals(
                zscores,
                self.threshold
            )
        )

        # --------------------------------
        # 6. Normalize into portfolio
        # --------------------------------

        positions = normalize_positions(
            signals
        )

        weights = (
            positions.iloc[-1]
            * self.target_gross_exposure
        )

        return {
            symbol: float(weights[symbol])
            for symbol in symbols
        }

class MovingAverageTradingStrategy(
    TradingStrategy
):

    def __init__(
        self,
        short_window,
        long_window,
        target_weight=0.10,
        allow_short=True
    ):

        super().__init__(
            "Moving Average"
        )

        if (
            isinstance(short_window, bool)
            or not isinstance(short_window, Integral)
            or short_window <= 0
        ):
            raise ValueError(
                "short_window must be a positive integer"
            )

        if (
            isinstance(long_window, bool)
            or not isinstance(long_window, Integral)
            or long_window <= 0
        ):
            raise ValueError(
                "long_window must be a positive integer"
            )

        if short_window >= long_window:
            raise ValueError(
                "short_window must be "
                "smaller than long_window"
            )

        if (
            isinstance(target_weight, bool)
            or not isinstance(target_weight, Real)
            or not np.isfinite(target_weight)
            or target_weight < 0
            or target_weight > 1
        ):
            raise ValueError(
                "target_weight must be a finite number "
                "between 0 and 1"
            )

        self.short_window = short_window
        self.long_window = long_window
        self.target_weight = target_weight
        self.allow_short = allow_short


    def generate_target_weights(
        self,
        history
    ):

        if history.empty:
            raise ValueError(
                "history cannot be empty"
            )

        target_weights = {}

        for symbol in history.columns:

            prices = history[symbol]

            if len(prices) < self.long_window:
                raise ValueError(
                    f"insufficient completed history for {symbol}: "
                    f"requires {self.long_window} observations"
                )

            required_prices = prices.iloc[
                -self.long_window:
            ]

            try:
                required_values = required_prices.to_numpy(
                    dtype=float
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"required prices for {symbol} must be numeric"
                ) from error

            if not np.isfinite(required_values).all():
                raise ValueError(
                    f"required prices for {symbol} must be finite"
                )

            short_ma = (
                required_prices
                .iloc[-self.short_window:]
                .mean()
            )

            long_ma = (
                required_prices
                .mean()
            )

            if short_ma > long_ma:

                target_weights[symbol] = (
                    self.target_weight
                )

            elif short_ma < long_ma:

                if self.allow_short:
                    target_weights[symbol] = (
                        -self.target_weight
                    )
                else:
                    target_weights[symbol] = 0.0

            else:

                target_weights[symbol] = 0.0

        return target_weights

class MomentumTradingStrategy(
    TradingStrategy
):

    def __init__(
        self,
        lookback,
        target_weight=0.10,
        allow_short=True
    ):

        super().__init__("Momentum")

        self.lookback = lookback
        self.target_weight = target_weight
        self.allow_short = allow_short


    def generate_target_weights(
        self,
        history
    ):

        if history.empty:
            raise ValueError(
                "history cannot be empty"
            )

        target_weights = {}

        for symbol in history.columns:

            prices = history[symbol]

            if len(prices) < self.lookback + 1:

                target_weights[symbol] = 0.0
                continue

            current_price = prices.iloc[-1]

            historical_price = prices.iloc[
                -self.lookback - 1
            ]

            if current_price > historical_price:

                target_weights[symbol] = (
                    self.target_weight
                )

            elif current_price < historical_price:

                if self.allow_short:
                    target_weights[symbol] = (
                        -self.target_weight
                    )
                else:
                    target_weights[symbol] = 0.0

            else:

                target_weights[symbol] = 0.0

        return target_weights
