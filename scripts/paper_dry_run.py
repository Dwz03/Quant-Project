import os
from dotenv import load_dotenv

load_dotenv()
import time
import uuid

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from src.market_data import AlpacaMarketDataAdapter

from src.order_state import (
    OrderState,
    OrderStateManager,
    alpaca_status_to_order_state
)

from src.safe_order_submitter import SafeOrderSubmitter
from src.order_store import OrderStore
from src.logger import create_logger


def main():

    api_key = os.environ["ALPACA_API_KEY"]

    secret_key = os.environ[
        "ALPACA_SECRET_KEY"
    ]

    trading_client = TradingClient(
        api_key,
        secret_key,
        paper=True
    )

    data_client = StockHistoricalDataClient(
        api_key,
        secret_key
    )

    logger = create_logger()

    market_data = AlpacaMarketDataAdapter(
        data_client,
        max_staleness_seconds=60
    )

    submitter = SafeOrderSubmitter(
        trading_client,
        max_retries=3
    )

    state_manager = OrderStateManager()

    store = OrderStore(
        "data/paper_orders.json"
    )

    # -------------------------
    # 1. Check market
    # -------------------------

    clock = trading_client.get_clock()

    if not clock.is_open:

        print("Market is closed. No order submitted.")
        return

    # -------------------------
    # 2. Market data
    # -------------------------

    symbol = "AAPL"
    quantity = 1

    price = market_data.get_latest_price(
        symbol
    )

    print(
        f"MARKET: {symbol} @ {price}"
    )

    logger.info(
        "Market data %s price=%s",
        symbol,
        price
    )

    # -------------------------
    # 3. Controlled signal
    # -------------------------

    signal = "BUY"

    print(
        f"SIGNAL: {signal} {symbol}"
    )

    logger.info(
        "Signal generated %s %s",
        signal,
        symbol
    )

    # -------------------------
    # 4. Logical order
    # -------------------------

    client_order_id = (
        "dryrun-"
        + uuid.uuid4().hex[:16]
    )

    state_manager.submit(
        client_order_id
    )

    store.record_order(
        client_order_id=client_order_id,
        symbol=symbol,
        side=signal,
        quantity=quantity,
        state=OrderState.SUBMITTED.value
    )

    # -------------------------
    # 5. Alpaca order request
    # -------------------------

    def create_request(order_id):

        return MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=order_id
        )

    alpaca_order = submitter.submit(
        create_request,
        client_order_id
    )

    print(
        f"ORDER SUBMITTED: {alpaca_order.id}"
    )

    logger.info(
        "Order submitted client_id=%s broker_id=%s",
        client_order_id,
        alpaca_order.id
    )

    # save broker order id
    store.record_order(
        client_order_id=client_order_id,
        broker_order_id=str(
            alpaca_order.id
        ),
        symbol=symbol,
        side=signal,
        quantity=quantity,
        state=OrderState.SUBMITTED.value
    )

    # -------------------------
    # 6. Poll lifecycle
    # -------------------------

    for _ in range(15):

        broker_order = (
            trading_client
            .get_order_by_client_id(
                client_order_id
            )
        )

        new_state = (
            alpaca_status_to_order_state(
                broker_order.status
            )
        )

        filled_quantity = float(
            broker_order.filled_qty or 0
        )

        if new_state is not None:

            current_state = (
                state_manager.get_state(
                    client_order_id
                )
            )

            if new_state != current_state:

                state_manager.update(
                    client_order_id,
                    new_state
                )

            store.update_state(
                client_order_id,
                new_state.value,
                filled_quantity
            )

            print(
                f"STATE: "
                f"{new_state.value}, "
                f"filled={filled_quantity}"
            )

        if new_state in {
            OrderState.FILLED,
            OrderState.CANCELLED
        }:
            break

        time.sleep(1)

    # -------------------------
    # 7. Final inspection
    # -------------------------

    final_order = store.get_order(
        client_order_id
    )

    print()
    print("FINAL ORDER:")
    print(final_order)

    positions = (
        trading_client.get_all_positions()
    )

    print()
    print("BROKER POSITIONS:")

    for position in positions:

        print(
            position.symbol,
            position.qty,
            position.market_value
        )


if __name__ == "__main__":
    main()