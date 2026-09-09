import os

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from .broker import Broker


class AlpacaPaperBroker(Broker):

    def __init__(self):

        load_dotenv()

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials are missing")

        self.client = TradingClient(
            api_key,
            secret_key,
            paper=True
        )


    def submit_order(self, order):

        if order.side == "BUY":
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL

        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=TimeInForce.DAY
        )

        broker_order = self.client.submit_order(
            order_data=request
        )

        return str(broker_order.id)


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