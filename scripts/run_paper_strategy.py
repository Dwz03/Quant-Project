import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from alpaca.data.historical.stock import (
    StockHistoricalDataClient
)
from alpaca.data.enums import DataFeed

from src.alpaca_broker import AlpacaPaperBroker
from src.market_data import AlpacaMarketDataAdapter
from src.portfolio import Portfolio
from src.risk_manager import RiskManager
from src.execution import ExecutionHandler
from src.rebalancer import Rebalancer
from src.trading_engine import TradingEngine
from src.paper_trading_engine import PaperTradingEngine

from src.strategy_factory import (
    build_strategy
)
from src.research.universes import LARGE_LIQUID_US_EQUITIES_V1


VOLATILITY_20_MAX_POSITION_PCT = 0.06
DEFAULT_MAX_POSITION_PCT = 0.02
VOLATILITY_20_LOOKBACK_DAYS = 60
PAPER_QUOTE_MAX_STALENESS_SECONDS = 300
VOLATILITY_20_EXECUTION_FEED = DataFeed.SIP
DEFAULT_EXECUTION_FEED = DataFeed.IEX
# Backward-compatible name for the shared quote/trade-fallback threshold.
PAPER_MAX_STALENESS_SECONDS = PAPER_QUOTE_MAX_STALENESS_SECONDS


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run or safely preview an Alpaca paper strategy.",
    )
    parser.add_argument("--strategy")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show targets and risk-checked orders without submitting",
    )
    return parser


def _require_paper_broker(broker):

    if getattr(broker, "is_paper", False) is not True:
        raise RuntimeError(
            "automated paper strategy requires "
            "an explicitly paper-mode broker"
        )


def _execution_feed_for_strategy(strategy_name):
    if strategy_name == "volatility_20":
        return VOLATILITY_20_EXECUTION_FEED
    return DEFAULT_EXECUTION_FEED


def _order_text(order):
    notional = getattr(order, "notional", None)
    if notional is not None:
        return f"{order.side} ${notional:.2f} notional {order.symbol}"
    return f"{order.side} {order.quantity} {order.symbol}"


def _print_preview(result):
    print("PAPER PREVIEW — NO ORDERS SUBMITTED")
    print("Completed signal date:", result["signal_date"])
    print("Eligible symbols:", result["eligible_symbols"])
    ranking = result["ranking"]
    selected = (
        ranking.loc[ranking["target_weight"] > 0]
        if "target_weight" in ranking.columns
        else ranking
    )
    print("Selected symbols:", len(selected))
    if not selected.empty:
        print(
            selected[[
                "symbol", "volatility_20", "rank", "target_weight"
            ]].to_string(index=False)
        )
    print("Current portfolio weights:", result["current_weights"])
    if "account_equity" in result:
        print("Broker account equity:", result["account_equity"])
        print(
            "Current broker position market values:",
            result["current_position_market_values"],
        )
        print("Target notionals:", result["target_notionals"])
        print("Delta notionals:", result["delta_notionals"])
    print("Proposed orders:")
    for order in result["orders"]:
        print(" ", _order_text(order))
    print("Accepted orders:")
    for order in result["accepted_orders"]:
        print(" ", _order_text(order))
    print("Risk rejections:")
    for rejection in result["rejected_orders"]:
        print(" ", _order_text(rejection["order"]), "—", rejection["reason"])


def _print_volatility_risk_profile():
    print("Volatility 20 V1 paper risk profile:")
    print("  frozen target weight per selected symbol: approximately 5.88%")
    print("  max_position_pct: 6.00%")
    print("  max_leverage: 1.00")


def main(argv=None):

    args = build_parser().parse_args(argv)

    # ==================================
    # 1. Alpaca Broker
    # ==================================

    broker = AlpacaPaperBroker()

    _require_paper_broker(broker)


    # ==================================
    # 2. Alpaca Market Data
    # ==================================

    load_dotenv()

    strategy_name = (
        args.strategy
        if args.strategy is not None
        else os.getenv("STRATEGY_NAME", "moving_average")
    ).strip().lower()

    api_key = os.getenv(
        "ALPACA_API_KEY"
    )

    secret_key = os.getenv(
        "ALPACA_SECRET_KEY"
    )

    if not api_key or not secret_key:

        raise ValueError(
            "Alpaca API credentials "
            "are missing"
        )


    data_client = (
        StockHistoricalDataClient(
            api_key,
            secret_key
        )
    )

    market_data = (
        AlpacaMarketDataAdapter(
            data_client,
            max_staleness_seconds=PAPER_QUOTE_MAX_STALENESS_SECONDS,
            execution_feed=_execution_feed_for_strategy(strategy_name),
        )
    )


    # ==================================
    # 3. Portfolio
    # ==================================

    account = broker.get_account()

    portfolio = Portfolio(
        float(account.equity)
    )

    # Actual cash/positions will be
    # synchronised by PaperTradingEngine.


    # ==================================
    # 4. Strategy
    # ==================================

    if strategy_name == "volatility_20":
        symbols = list(LARGE_LIQUID_US_EQUITIES_V1)
    else:
        symbols_text = os.getenv(
            "SYMBOLS",
            "AAPL,MSFT,GOOG"
        )
        symbols = [
            symbol.strip().upper()
            for symbol
            in symbols_text.split(",")
            if symbol.strip()
        ]

    strategy_config = {

        "short_window": int(
            os.getenv(
                "STRATEGY_SHORT_WINDOW",
                "10"
            )
        ),

        "long_window": int(
            os.getenv(
                "STRATEGY_LONG_WINDOW",
                "30"
            )
        ),

        "lookback": int(
            os.getenv(
                "STRATEGY_LOOKBACK",
                "20"
            )
        ),

        "target_weight": float(
            os.getenv(
                "STRATEGY_TARGET_WEIGHT",
                "0.01"
            )
        ),

        "allow_short": (
            os.getenv(
                "STRATEGY_ALLOW_SHORT",
                "false"
            ).lower()
            == "true"
        )
    }

    strategy = build_strategy(
        strategy_name,
        symbols=symbols,
        config=strategy_config
    )


    # ==================================
    # 5. Risk
    # ==================================

    max_position_pct = (
        VOLATILITY_20_MAX_POSITION_PCT
        if strategy_name == "volatility_20"
        else DEFAULT_MAX_POSITION_PCT
    )
    risk_manager = RiskManager(
        max_position_pct=max_position_pct,
        max_leverage=1.0
    )


    # ==================================
    # 6. Execution / Rebalancer
    # ==================================

    execution = ExecutionHandler(
        commission_rate=0.0,
        slippage_rate=0.0
    )

    rebalancer = Rebalancer()


    # ==================================
    # 7. Trading Engine
    # ==================================

    trading_engine = TradingEngine(
        portfolio,
        risk_manager,
        execution,
        rebalancer,
        strategy,
        broker=broker
    )


    # ==================================
    # 8. Paper Trading Engine
    # ==================================

    paper_engine = (
        PaperTradingEngine(
            broker=broker,
            market_data=market_data,
            trading_engine=trading_engine,
            symbols=symbols,

            lookback_days=(
                VOLATILITY_20_LOOKBACK_DAYS
                if strategy_name == "volatility_20"
                else max(
                    90,
                    getattr(
                        strategy,
                        "long_window",
                        0
                    ) * 2
                )
            ),

            # Existing strategies retain their
            # final-15-minute schedule.
            minutes_before_close=15,

            # volatility_20 forms after the prior
            # completed close and executes in the
            # first five minutes of the next session.
            execution_window=(
                "after_open"
                if strategy_name == "volatility_20"
                else "before_close"
            ),
            minutes_after_open=5,

            # Check scheduler every minute.
            poll_interval_seconds=60
        )
    )


    # ==================================
    # 9. Preview or Start Bot
    # ==================================

    if strategy_name == "volatility_20":
        _print_volatility_risk_profile()

    if args.preview:
        preview = paper_engine.preview_cycle()
        if preview["status"] != "PREVIEW":
            print("Paper preview status:", preview["status"])
            return preview
        _print_preview(preview)
        return preview

    print(
        "Starting Alpaca paper bot..."
    )

    print(
        "Strategy:",
        strategy.name
    )

    print(
        "Symbols:",
        symbols
    )

    print(
        "Paper mode only."
    )

    paper_engine.run_forever()


if __name__ == "__main__":
    main()
