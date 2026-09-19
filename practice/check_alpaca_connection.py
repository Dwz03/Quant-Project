import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from src.alpaca_broker import AlpacaPaperBroker
from src.alpaca_credentials import (
    ALPACA_ACCOUNT_ENV_VARS,
    load_alpaca_credentials,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect an Alpaca paper account without placing orders.",
    )
    parser.add_argument(
        "--account",
        required=True,
        choices=list(ALPACA_ACCOUNT_ENV_VARS),
    )
    args = parser.parse_args(argv)

    load_dotenv()
    api_key, secret_key = load_alpaca_credentials(args.account)
    broker = AlpacaPaperBroker(api_key, secret_key)
    account = broker.get_account()

    print("Account status:", account.status)
    print("Cash:", account.cash)
    print("Equity:", account.equity)
    print("Buying power:", account.buying_power)


if __name__ == "__main__":
    main()
