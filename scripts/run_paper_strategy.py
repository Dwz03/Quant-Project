from datetime import datetime, timezone, timedelta

import os
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical.stock import StockHistoricalDataClient

from src.alpaca_broker import AlpacaPaperBroker
from src.market_data import AlpacaMarketDataAdapter
from src.portfolio import Portfolio
from src.risk_manager import RiskManager
from src.execution import ExecutionHandler
from src.rebalancer import Rebalancer
from src.trading_engine import TradingEngine
from src.strategy import MeanReversionTradingStrategy


def main():

    # -------------------------
    # 1. Broker
    # -------------------------

    broker = AlpacaPaperBroker()

    clock = broker.get_clock()

    print("Market open:", clock.is_open)

    market_open = clock.is_open

    if not market_open:
        print("Market is closed.")
        print("Running strategy in dry-run mode.")

    # -------------------------
    # 2. Account
    # -------------------------

    account = broker.get_account()

    broker_positions = broker.get_positions()

    portfolio = Portfolio(
        float(account.equity)
    )

    portfolio.sync_from_broker(
        account,
        broker_positions
    )

    print("Paper equity:", float(account.equity))
    print("Paper cash:", portfolio.cash)

    print("Synced positions:")

    for symbol, position in portfolio.positions.items():
        print(
            symbol,
            position.quantity,
            position.average_cost
        )

    # -------------------------
    # 3. Market data client
    # -------------------------

    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    data_client = StockHistoricalDataClient(
        api_key,
        secret_key
    )

    market_data = AlpacaMarketDataAdapter(
        data_client
    )

    # -------------------------
    # 4. Historical data
    # -------------------------

    symbol = "AAPL"

    now = datetime.now(timezone.utc)

    history = market_data.get_history(
        [symbol],
        start=now - timedelta(days=60),
        end=now
    )

    if market_open:

        latest_price = market_data.get_latest_price(
            symbol
        )

        history.loc[now, symbol] = latest_price

    else:

        latest_price = history[symbol].iloc[-1]

        print(
            "Using last historical close:",
            latest_price
        )

    print(history.tail())

    # -------------------------
    # 6. Strategy
    # -------------------------

    strategy = MeanReversionTradingStrategy(
        window=20,
        target_weight=0.10,
        max_gross_exposure=0.10,
        allow_short=False
    )
    historical_window = history[symbol].iloc[
        -strategy.window - 1:-1
    ]

    historical_mean = historical_window.mean()
    current_price = history[symbol].iloc[-1]

    print("Strategy diagnostics:")
    print("Current price:", current_price)
    print("Historical mean:", historical_mean)
    # -------------------------
    # 7. Trading components
    # -------------------------

    risk_manager = RiskManager(
        max_position_pct=0.10,
        max_leverage=1.0
    )

    execution = ExecutionHandler(
        commission_rate=0.0,
        slippage_rate=0.0
    )

    rebalancer = Rebalancer()

    engine = TradingEngine(
        portfolio,
        risk_manager,
        execution,
        rebalancer,
        strategy,
        broker=broker
    )

    # -------------------------
    # 8. Run ONE strategy cycle
    # -------------------------

    if market_open:

        result = engine.run_broker_cycle(history)

        print("LIVE PAPER MODE")

        print("Target weights:")
        print(result["target_weights"])

        print("Orders:")

        for order in result["orders"]:
            print(
                order.symbol,
                order.side,
                order.quantity
            )

        order_ids = result["submitted_orders"]

        print("Submitted order IDs:")
        print(order_ids)


        # -------------------------
        # Wait for fills
        # -------------------------

        if order_ids:

            order_updates = engine.wait_for_orders(
                order_ids,
                timeout=30,
                poll_interval=1
            )

            print("Order updates:")

            for update in order_updates:
                print(update)

            print("Synced portfolio:")

            print("Cash:", portfolio.cash)

            for symbol, position in portfolio.positions.items():

                print(
                    symbol,
                    position.quantity,
                    position.average_cost
                )

        else:

            print("No orders submitted.")

        print("LIVE PAPER MODE")

        print("Target weights:")
        print(result["target_weights"])

        print("Orders:")

        for order in result["orders"]:
            print(
                order.symbol,
                order.side,
                order.quantity
            )

        print("Submitted order IDs:")
        print(result["submitted_orders"])

    else:

        target_weights = (
            strategy.generate_target_weights(history)
        )

        prices = history.iloc[-1].to_dict()

        orders = rebalancer.generate_orders(
            target_weights,
            portfolio,
            prices
        )

        approved_orders = []

        for order in orders:

            if risk_manager.check_order(
                order,
                portfolio,
                prices
            ):
                approved_orders.append(order)

        print("DRY RUN")

        print("Target weights:")
        print(target_weights)

        print("Approved orders:")

        for order in approved_orders:
            print(
                order.symbol,
                order.side,
                order.quantity
            )

        print("No orders submitted.")




if __name__ == "__main__":
    main()