import os

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient

from src.daily_report import DailyReport


load_dotenv()


def main():

    api_key = os.environ[
        "ALPACA_API_KEY"
    ]

    secret_key = os.environ[
        "ALPACA_SECRET_KEY"
    ]

    trading_client = TradingClient(
        api_key,
        secret_key,
        paper=True
    )

    reporter = DailyReport(
        trading_client
    )

    report, filepath = (
        reporter.run()
    )

    print()
    print("=== PAPER TRADING REPORT ===")
    print()

    account = report["account"]

    print(
        f"Equity: "
        f"${account['equity']:.2f}"
    )

    print(
        f"Daily PnL: "
        f"${account['daily_pnl']:.2f}"
    )

    print(
        f"Daily Return: "
        f"{account['daily_return']:.2%}"
    )

    print(
        f"Cash: "
        f"${account['cash']:.2f}"
    )

    print()
    print("Positions:")

    for position in report[
        "positions"
    ]:

        print(
            position["symbol"],
            position["quantity"],
            f"${position['market_value']:.2f}",
            f"PnL={position['unrealized_pnl']:.2f}"
        )

    print()
    print(
        "Orders:",
        len(report["orders"])
    )

    print(
        "Errors:",
        len(report["errors"])
    )

    print()
    print(
        f"Report saved to: "
        f"{filepath}"
    )


if __name__ == "__main__":

    main()