from src.alpaca_broker import AlpacaPaperBroker


broker = AlpacaPaperBroker()

account = broker.get_account()

print("Account status:", account.status)
print("Cash:", account.cash)
print("Equity:", account.equity)
print("Buying power:", account.buying_power)