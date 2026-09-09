from datetime import datetime, timezone, timedelta
import time


class PaperTradingEngine:

    def __init__(
        self,
        broker,
        market_data,
        trading_engine,
        symbols,
        lookback_days=90,
        minutes_before_close=15,
        poll_interval_seconds=60
    ):

        if not symbols:
            raise ValueError(
                "symbols cannot be empty"
            )

        self.broker = broker
        self.market_data = market_data
        self.trading_engine = trading_engine

        self.symbols = symbols
        self.lookback_days = lookback_days

        # Idempotency:
        # current strategies are daily, so we allow
        # only one decision cycle per trading day
        # within the current Python process.
        self.last_cycle_key = None
        self.kill_switch_active = False
        self.kill_switch_reason = None

        self.minutes_before_close = (
            minutes_before_close
        )

        self.poll_interval_seconds = (
            poll_interval_seconds
        )


    def _get_cycle_key(
        self,
        clock
    ):

        timestamp = getattr(
            clock,
            "timestamp",
            None
        )

        if timestamp is None:

            timestamp = datetime.now(
                timezone.utc
            )

        return timestamp.date().isoformat()


    def sync_portfolio(self):

        account = self.broker.get_account()

        broker_positions = (
            self.broker.get_positions()
        )

        self.trading_engine.portfolio.sync_from_broker(
            account,
            broker_positions
        )


    def load_history(self):

        now = datetime.now(
            timezone.utc
        )

        # Use completed historical bars only.
        # The current market observation is added
        # separately in add_latest_prices().
        history = self.market_data.get_history(
            self.symbols,
            start=(
                now
                - timedelta(
                    days=self.lookback_days
                )
            ),
            end=(
                now
                - timedelta(days=1)
            )
        )

        return history


    def add_latest_prices(
        self,
        history
    ):

        now = datetime.now(
            timezone.utc
        )

        history = history.copy()

        for symbol in self.symbols:

            latest_price = (
                self.market_data
                .get_latest_price(symbol)
            )

            history.loc[
                now,
                symbol
            ] = latest_price

        return history


    def run_cycle(self):

        if self.kill_switch_active:

            return {
                "status": "KILL_SWITCH_ACTIVE",
                "reason": self.kill_switch_reason,
                "target_weights": {},
                "orders": [],
                "order_updates": []
            }

        # -------------------------
        # 1. Broker state
        # -------------------------

        self.sync_portfolio()


        # -------------------------
        # 2. Market clock
        # -------------------------

        clock = self.broker.get_clock()

        if not clock.is_open:

            return {
                "status": "MARKET_CLOSED",
                "target_weights": {},
                "orders": [],
                "order_updates": []
            }


        # -------------------------
        # 3. Idempotency
        # -------------------------

        cycle_key = self._get_cycle_key(
            clock
        )

        if (
            self.last_cycle_key
            == cycle_key
        ):

            return {
                "status": "DUPLICATE_SKIPPED",
                "cycle_key": cycle_key,
                "target_weights": {},
                "orders": [],
                "order_updates": []
            }


        # -------------------------
        # 4. Data
        # -------------------------

        history = self.load_history()

        if history.empty:

            raise ValueError(
                "market history is empty"
            )

        history = self.add_latest_prices(
            history
        )


        # -------------------------
        # 5. Consume cycle
        # -------------------------

        # From this point onward, the cycle is
        # considered consumed.
        #
        # If submission partially fails, we do
        # not want an automatic retry of the
        # same strategy decision.
        self.last_cycle_key = cycle_key


        # -------------------------
        # 6. Strategy → Broker
        # -------------------------

        try:

            result = (
                self.trading_engine
                .run_broker_cycle(
                    history,
                    cycle_key=cycle_key
                )
            )

            order_ids = (
                result["submitted_orders"]
            )

            if order_ids:

                order_updates = (
                    self.trading_engine
                    .wait_for_orders(
                        order_ids,
                        timeout=30,
                        poll_interval=1
                    )
                )

            else:

                order_updates = []


        except Exception as error:

            self.activate_kill_switch(
                reason=error
            )

            raise

        return {
            "status": "COMPLETED",
            "cycle_key": cycle_key,
            "target_weights":
                result["target_weights"],
            "orders":
                result["orders"],
            "order_updates":
                order_updates
        }

    def activate_kill_switch(
        self,
        reason
    ):

        self.kill_switch_active = True
        self.kill_switch_reason = str(reason)


    def reset_kill_switch(self):

        self.kill_switch_active = False
        self.kill_switch_reason = None

    def _should_run_now(
        self,
        clock
    ):

        if not clock.is_open:
            return False

        time_to_close = (
            clock.next_close
            - clock.timestamp
        )

        seconds_to_close = (
            time_to_close.total_seconds()
        )

        execution_window = (
            self.minutes_before_close
            * 60
        )

        return (
            0
            <= seconds_to_close
            <= execution_window
        )

    def run_scheduled_step(self):

        if self.kill_switch_active:

            return {
                "status":
                    "KILL_SWITCH_ACTIVE",
                "reason":
                    self.kill_switch_reason,
                "target_weights": {},
                "orders": [],
                "order_updates": []
            }


        clock = self.broker.get_clock()


        if not clock.is_open:

            return {
                "status":
                    "MARKET_CLOSED",
                "target_weights": {},
                "orders": [],
                "order_updates": []
            }


        cycle_key = self._get_cycle_key(
            clock
        )


        if (
            self.last_cycle_key
            == cycle_key
        ):

            return {
                "status":
                    "DUPLICATE_SKIPPED",
                "cycle_key":
                    cycle_key,
                "target_weights": {},
                "orders": [],
                "order_updates": []
            }


        if not self._should_run_now(
            clock
        ):

            return {
                "status":
                    "WAITING_FOR_EXECUTION_WINDOW",
                "cycle_key":
                    cycle_key,
                "target_weights": {},
                "orders": [],
                "order_updates": []
            }


        return self.run_cycle()

    def run_forever(self):

        while True:

            try:

                result = (
                    self.run_scheduled_step()
                )

                print(
                    "Paper bot status:",
                    result["status"]
                )

                if (
                    result["status"]
                    == "KILL_SWITCH_ACTIVE"
                ):

                    print(
                        "Kill switch reason:",
                        result.get(
                            "reason"
                        )
                    )

                    break


            except KeyboardInterrupt:

                print(
                    "Paper trading bot "
                    "stopped manually."
                )

                break


            except Exception as error:

                self.activate_kill_switch(
                    error
                )

                print(
                    "Paper trading bot "
                    "stopped due to error:",
                    error
                )

                break


            time.sleep(
                self.poll_interval_seconds
            )
