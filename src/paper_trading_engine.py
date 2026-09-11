from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import tempfile
import time

from .trading_engine import (
    _cycle_client_order_id_prefix
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STATE_FILE = (
    _PROJECT_ROOT
    / ".state"
    / "paper_cycle.json"
)


class PaperTradingEngine:

    def __init__(
        self,
        broker,
        market_data,
        trading_engine,
        symbols,
        lookback_days=90,
        minutes_before_close=15,
        poll_interval_seconds=60,
        state_file=_DEFAULT_STATE_FILE
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
        self.pending_cycle_key = None
        self.pending_order_ids = []
        self.state_file = (
            None
            if state_file is None
            else Path(state_file)
        )
        self.persisted_cycle_key = (
            self._load_persisted_cycle_key()
        )
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


    def _load_persisted_cycle_key(self):

        if (
            self.state_file is None
            or not self.state_file.exists()
        ):
            return None

        with self.state_file.open("r") as file:
            state = json.load(file)

        cycle_key = state.get(
            "last_attempted_cycle_key"
        )

        if not isinstance(cycle_key, str):
            raise ValueError(
                "paper cycle state is invalid"
            )

        return cycle_key


    def _persist_cycle_attempt(self, cycle_key):

        if self.state_file is None:
            self.persisted_cycle_key = cycle_key
            return

        self.state_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.state_file.parent,
                prefix=f".{self.state_file.name}.",
                suffix=".tmp",
                delete=False
            ) as temporary_file:
                temporary_path = Path(
                    temporary_file.name
                )
                json.dump(
                    {
                        "last_attempted_cycle_key":
                            cycle_key
                    },
                    temporary_file
                )
                temporary_file.flush()
                os.fsync(
                    temporary_file.fileno()
                )

            os.replace(
                temporary_path,
                self.state_file
            )
        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

        self.persisted_cycle_key = cycle_key


    def sync_portfolio(self):

        account = self.broker.get_account()

        broker_positions = (
            self.broker.get_positions()
        )

        self.trading_engine.portfolio.sync_from_broker(
            account,
            broker_positions
        )


    def _order_summary(self, order):

        def value(attribute):

            attribute = getattr(
                attribute,
                "value",
                attribute
            )

            if attribute is None:
                return None

            return str(attribute)

        return {
            "order_id": value(
                getattr(order, "id", None)
            ),
            "client_order_id": value(
                getattr(
                    order,
                    "client_order_id",
                    None
                )
            ),
            "symbol": value(
                getattr(order, "symbol", None)
            ),
            "side": value(
                getattr(order, "side", None)
            ),
            "quantity": value(
                getattr(order, "qty", None)
            ),
            "filled_quantity": value(
                getattr(order, "filled_qty", None)
            ),
            "status": value(
                getattr(order, "status", None)
            )
        }


    def _empty_cycle_result(
        self,
        status,
        **extra
    ):

        result = {
            "status": status,
            "target_weights": {},
            "orders": [],
            "order_updates": []
        }
        result.update(extra)

        return result


    def _is_fully_filled(self, order):

        status = getattr(
            getattr(order, "status", None),
            "value",
            getattr(order, "status", None)
        )

        if str(status).lower() != "filled":
            return False

        try:
            return float(order.filled_qty) >= float(
                order.qty
            )
        except (TypeError, ValueError):
            return False


    def _reconcile_cycle_state(
        self,
        clock
    ):

        self.sync_portfolio()

        cycle_key = self._get_cycle_key(
            clock
        )
        cycle_start = datetime.fromisoformat(
            cycle_key
        ).replace(tzinfo=timezone.utc)
        cycle_end = cycle_start + timedelta(
            days=1
        )

        open_orders = (
            self.broker.get_open_orders()
        )
        order_history = (
            self.broker.get_order_history(
                start=cycle_start,
                end=cycle_end
            )
        )

        client_order_id_prefix = (
            _cycle_client_order_id_prefix(
                cycle_key
            )
        )
        cycle_orders = [
            order
            for order in order_history
            if str(
                getattr(
                    order,
                    "client_order_id",
                    ""
                )
            ).startswith(
                client_order_id_prefix
            )
        ]
        cycle_open_orders = [
            order
            for order in open_orders
            if str(
                getattr(
                    order,
                    "client_order_id",
                    ""
                )
            ).startswith(
                client_order_id_prefix
            )
        ]

        if open_orders:

            if cycle_open_orders:
                cycle_state = "PENDING"
            elif cycle_orders:
                if all(
                    self._is_fully_filled(order)
                    for order in cycle_orders
                ):
                    cycle_state = "FILLED_COMPLETED"
                else:
                    cycle_state = "TERMINAL_INCOMPLETE"
            else:
                cycle_state = "NOT_ATTEMPTED"

            return self._empty_cycle_result(
                "OPEN_ORDERS_PENDING",
                cycle_key=cycle_key,
                cycle_state=cycle_state,
                outstanding_orders=[
                    self._order_summary(order)
                    for order in open_orders
                ]
            )

        if cycle_orders:

            self.last_cycle_key = cycle_key
            self.pending_cycle_key = None
            self.pending_order_ids = []

            if all(
                self._is_fully_filled(order)
                for order in cycle_orders
            ):
                status = "FILLED_COMPLETED"
            else:
                status = "TERMINAL_INCOMPLETE"

            return self._empty_cycle_result(
                status,
                cycle_key=cycle_key,
                cycle_state=status,
                cycle_orders=[
                    self._order_summary(order)
                    for order in cycle_orders
                ]
            )

        if self.persisted_cycle_key == cycle_key:

            self.last_cycle_key = cycle_key

            return self._empty_cycle_result(
                "DUPLICATE_SKIPPED",
                cycle_key=cycle_key,
                reason="cycle already attempted"
            )

        return None


    def load_history(self):

        now = datetime.now(
            timezone.utc
        )

        # Strategy signals use completed daily bars
        # only. Current prices are fetched separately
        # for execution and rebalancing.
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


    def run_cycle(self):

        if self.kill_switch_active:

            return self._empty_cycle_result(
                "KILL_SWITCH_ACTIVE",
                reason=self.kill_switch_reason
            )

        clock = self.broker.get_clock()

        reconciliation = (
            self._reconcile_cycle_state(clock)
        )

        if reconciliation is not None:
            return reconciliation

        return self._run_cycle_after_preflight(
            clock=clock
        )


    def _run_cycle_after_preflight(
        self,
        clock=None
    ):


        # -------------------------
        # 1. Market clock
        # -------------------------

        if clock is None:
            clock = self.broker.get_clock()

        if not clock.is_open:

            return {
                "status": "MARKET_CLOSED",
                "target_weights": {},
                "orders": [],
                "order_updates": []
            }


        # -------------------------
        # 2. Idempotency
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
        # 3. Data
        # -------------------------

        history = self.load_history()

        if history.empty:

            raise ValueError(
                "market history is empty"
            )

        execution_prices = {
            symbol: self.market_data.get_latest_price(
                symbol
            )
            for symbol in self.symbols
        }


        # -------------------------
        # 4. Strategy → Broker
        # -------------------------

        # This is the point at which today's
        # strategy decision is consumed. The
        # local marker covers cycles that create
        # no durable broker order record.
        self._persist_cycle_attempt(
            cycle_key
        )

        try:

            result = (
                self.trading_engine
                .run_broker_cycle(
                    history,
                    cycle_key=cycle_key,
                    execution_prices=execution_prices
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


        except TimeoutError:

            self.pending_cycle_key = cycle_key
            self.pending_order_ids = list(
                order_ids
            )

            try:
                open_orders = (
                    self.broker.get_open_orders()
                )
                outstanding_orders = [
                    self._order_summary(order)
                    for order in open_orders
                ]
            except Exception:
                outstanding_orders = [
                    {"order_id": str(order_id)}
                    for order_id in order_ids
                ]

            return self._empty_cycle_result(
                "ORDERS_PENDING",
                cycle_key=cycle_key,
                outstanding_orders=(
                    outstanding_orders
                )
            )


        except Exception as error:

            self.activate_kill_switch(
                reason=error
            )

            raise

        self.last_cycle_key = cycle_key

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

            return self._empty_cycle_result(
                "KILL_SWITCH_ACTIVE",
                reason=self.kill_switch_reason
            )


        clock = self.broker.get_clock()

        reconciliation = (
            self._reconcile_cycle_state(clock)
        )


        if reconciliation is not None:
            return reconciliation


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


        return self._run_cycle_after_preflight(
            clock=clock
        )

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
