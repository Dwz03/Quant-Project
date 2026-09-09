from src.alpaca_broker import AlpacaPaperBroker
from src.order import Order


broker = AlpacaPaperBroker()

broker.cancel_order(
    "cd011cdd-721f-4b65-8d35-9f93e58975f9"
)

broker.cancel_order(
    "ba840d91-a56d-407c-a1a3-4800211fd60a"
)
