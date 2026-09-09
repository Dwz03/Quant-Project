import json

from datetime import datetime, timezone
from pathlib import Path

from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


class DailyReport:

    def __init__(
        self,
        trading_client,
        log_file="logs/trading.log",
        report_directory="reports"
    ):

        self.trading_client = trading_client

        self.log_file = Path(log_file)

        self.report_directory = Path(
            report_directory
        )


    def get_account_summary(self):

        account = (
            self.trading_client.get_account()
        )

        equity = float(
            account.equity or 0
        )

        last_equity = float(
            account.last_equity or 0
        )

        daily_pnl = (
            equity - last_equity
        )

        if last_equity == 0:

            daily_return = 0

        else:

            daily_return = (
                daily_pnl / last_equity
            )

        return {

            "equity": equity,

            "last_equity": last_equity,

            "daily_pnl": daily_pnl,

            "daily_return": daily_return,

            "cash": float(
                account.cash or 0
            ),

            "buying_power": float(
                account.buying_power or 0
            )
        }

    def get_positions(self):

        positions = (
            self.trading_client
            .get_all_positions()
        )

        result = []

        for position in positions:

            result.append({

                "symbol": position.symbol,

                "quantity": float(
                    position.qty
                ),

                "market_value": float(
                    position.market_value
                ),

                "unrealized_pnl": float(
                    position.unrealized_pl
                )
            })

        return result

    def get_orders(self):

        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=100
        )

        orders = (
            self.trading_client
            .get_orders(
                filter=request
            )
        )

        result = []

        for order in orders:

            result.append({

                "id": str(order.id),

                "client_order_id":
                    order.client_order_id,

                "symbol":
                    order.symbol,

                "side":
                    order.side.value,

                "quantity":
                    float(order.qty),

                "filled_quantity":
                    float(
                        order.filled_qty or 0
                    ),

                "status":
                    order.status.value
            })

        return result

    def get_errors(self):

        if not self.log_file.exists():

            return []

        errors = []

        with open(
            self.log_file,
            "r"
        ) as file:

            for line in file:

                if "ERROR" in line:

                    errors.append(
                        line.strip()
                    )

        return errors

    def generate(self):

        report = {

            "generated_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "account":
                self.get_account_summary(),

            "positions":
                self.get_positions(),

            "orders":
                self.get_orders(),

            "errors":
                self.get_errors()
        }

        return report

    def save(self, report):

        self.report_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        date_string = (
            datetime.now(
                timezone.utc
            )
            .strftime("%Y-%m-%d")
        )

        filepath = (
            self.report_directory
            / f"paper_report_{date_string}.json"
        )

        with open(
            filepath,
            "w"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

        return filepath

    def run(self):

        report = self.generate()

        filepath = self.save(
            report
        )

        return report, filepath