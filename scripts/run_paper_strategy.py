import os

from dotenv import load_dotenv

from alpaca.data.historical.stock import (
    StockHistoricalDataClient
)

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


def main():

    # ==================================
    # 1. Alpaca Broker
    # ==================================

    broker = AlpacaPaperBroker()


    # ==================================
    # 2. Alpaca Market Data
    # ==================================

    load_dotenv()

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
            data_client
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


    strategy_name = os.getenv(
        "STRATEGY_NAME",
        "momentum"
    )

    strategy = build_strategy(
        strategy_name,
        symbols=symbols
    )


    strategy_name = os.getenv(
        "STRATEGY_NAME",
        "momentum"
    )

    strategy = build_strategy(
        strategy_name
    )


    # ==================================
    # 5. Risk
    # ==================================

    risk_manager = RiskManager(
        max_position_pct=0.02,
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

            lookback_days=90,

            # Daily strategy runs once
            # during the final 15 min.
            minutes_before_close=15,

            # Check scheduler every minute.
            poll_interval_seconds=60
        )
    )


    # ==================================
    # 9. Start Bot
    # ==================================

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

    result = (
        paper_engine
        .run_scheduled_step()
    )

    print(result)


if __name__ == "__main__":
    main()