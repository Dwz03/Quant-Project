import os

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest
)
from alpaca.trading.enums import (
    OrderSide,
    QueryOrderStatus,
    TimeInForce
)

from .broker import Broker
from alpaca.common.exceptions import APIError


class AlpacaPaperBroker(Broker):

    is_paper = True

    def __init__(self):

        load_dotenv()

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials are missing")

        self.client = TradingClient(
            api_key,
            secret_key,
            paper=self.is_paper
        )

    def submit_order(
        self,
        order
    ):

        client_order_id = getattr(
            order,
            "client_order_id",
            None
        )


        # ----------------------------------
        # Idempotency check
        # ----------------------------------

        if client_order_id is not None:

            try:

                existing_order = (
                    self.get_order_by_client_id(
                        client_order_id
                    )
                )

                return str(
                    existing_order.id
                )


            except APIError as error:

                if error.status_code != 404:
                    raise

    # 404 means the client_order_id
    # does not exist yet.
    # It is therefore safe to submit.


        # ----------------------------------
        # Normal submission
        # ----------------------------------

        if order.side == "BUY":

            side = OrderSide.BUY

        elif order.side == "SELL":

            side = OrderSide.SELL

        else:

            raise ValueError(
                "order side must be BUY or SELL"
            )


        request_data = {
            "symbol": order.symbol,
            "qty": order.quantity,
            "side": side,
            "time_in_force":
                TimeInForce.DAY
        }


        if client_order_id is not None:

            request_data[
                "client_order_id"
            ] = client_order_id


        request = MarketOrderRequest(
            **request_data
        )


        broker_order = (
            self.client.submit_order(
                order_data=request
            )
        )


        return str(
            broker_order.id
        )

    def cancel_order(self, order_id):

        self.client.cancel_order_by_id(order_id)

        return True


    def get_order_status(self, order_id):

        broker_order = self.client.get_order_by_id(order_id)

        return broker_order.status.value


    def get_account(self):

        return self.client.get_account()

    def get_order(self, order_id):

        return self.client.get_order_by_id(order_id)

    def get_positions(self):

        return self.client.get_all_positions()


    def get_open_orders(self):

        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN
        )

        return self.client.get_orders(
            filter=request
        )


    def get_order_history(
        self,
        start,
        end
    ):

        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=start,
            until=end,
            limit=500
        )

        return self.client.get_orders(
            filter=request
        )


    def get_clock(self):

        return self.client.get_clock()

    def get_asset(
        self,
        symbol
    ):

        return self.client.get_asset(
            symbol
        )


    def can_short(
        self,
        symbol
    ):

        account = self.get_account()

        if not getattr(
            account,
            "shorting_enabled",
            False
        ):

            return False


        asset = self.get_asset(
            symbol
        )


        return (
            getattr(
                asset,
                "tradable",
                False
            )
            and
            getattr(
                asset,
                "shortable",
                False
            )
        )

    def get_order_by_client_id(
        self,
        client_order_id
    ):

        return (
            self.client
            .get_order_by_client_id(
                client_order_id
            )
        )
