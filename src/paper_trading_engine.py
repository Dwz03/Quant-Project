from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import tempfile
import time
from zoneinfo import ZoneInfo

import pandas as pd

from .trading_engine import (
    _cycle_client_order_id_prefix
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STATE_FILE = (
    _PROJECT_ROOT
    / ".state"
    / "paper_cycle.json"
)
_NEW_YORK = ZoneInfo("America/New_York")


def filter_completed_daily_history(history, as_of):
    """Exclude the active New York session from a daily close panel."""
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a DataFrame")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("daily history must use a DatetimeIndex")
    if not history.index.is_unique:
        raise ValueError("daily history index must be unique")
    if not history.index.is_monotonic_increasing:
        raise ValueError("daily history must be chronological")

    current_time = pd.Timestamp(as_of)
    if current_time.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    current_new_york_date = (
        current_time.tz_convert(_NEW_YORK).normalize().tz_localize(None)
    )

    if history.index.tz is None:
        if not history.index.equals(history.index.normalize()):
            raise ValueError(
                "timezone-naive daily history must use normalized session dates"
            )
        session_dates = history.index.normalize()
    else:
        session_dates = (
            history.index.tz_convert(_NEW_YORK).normalize().tz_localize(None)
        )

    completed = history.loc[session_dates < current_new_york_date].copy()
    completed.index = session_dates[session_dates < current_new_york_date]
    if completed.empty:
        raise ValueError("no completed New York daily bars are available")
    if not completed.index.is_unique:
        raise ValueError("daily bars do not map uniquely to New York sessions")
    return completed


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
        state_file=_DEFAULT_STATE_FILE,
        execution_window="before_close",
        minutes_after_open=5,
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
        if execution_window not in {"before_close", "after_open"}:
            raise ValueError(
                "execution_window must be before_close or after_open"
            )
        self.execution_window = execution_window
        self.minutes_after_open = minutes_after_open

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

        return account, broker_positions


    def _uses_notional_orders(self):
        return (
            getattr(
                getattr(self.trading_engine, "strategy", None),
                "paper_order_mode",
                "quantity",
            )
            == "notional"
        )


    @staticmethod
    def _notional_account_state(account, broker_positions):
        equity = float(account.equity)
        buying_power_value = getattr(account, "buying_power", None)
        if buying_power_value is None:
            buying_power_value = account.cash
        available_buying_power = float(buying_power_value)
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("broker account equity must be positive and finite")
        if (
            not math.isfinite(available_buying_power)
            or available_buying_power < 0
        ):
            raise ValueError("broker buying power must be finite and non-negative")

        current_market_values = {}
        for position in broker_positions:
            symbol = position.symbol
            if symbol in current_market_values:
                raise ValueError(f"duplicate broker position for {symbol}")
            quantity = float(position.qty)
            if not math.isfinite(quantity) or quantity < 0:
                raise ValueError(
                    "volatility_20 notional mode requires long-only positions"
                )
            market_value = getattr(position, "market_value", None)
            if market_value is None:
                raise ValueError(
                    f"broker market value is required for position {symbol}"
                )
            value = float(market_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"broker market value for {symbol} must be non-negative"
                )
            current_market_values[symbol] = value

        return {
            "account_equity": equity,
            "available_buying_power": available_buying_power,
            "current_position_market_values": current_market_values,
        }


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

        summary = {
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
        notional = value(getattr(order, "notional", None))
        if notional is not None:
            summary["notional"] = notional
        return summary


    def _empty_cycle_result(
        self,
        status,
        **extra
    ):

        result = {
            "status": status,
            "target_weights": {},
            "orders": [],
            "accepted_orders": [],
            "rejected_orders": [],
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

        quantity = getattr(order, "qty", None)
        notional = getattr(order, "notional", None)
        if quantity is None and notional is not None:
            try:
                return float(order.filled_qty) > 0
            except (TypeError, ValueError):
                return False

        try:
            return float(order.filled_qty) >= float(
                quantity
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


    def load_history(self, as_of=None):

        now = (
            datetime.now(timezone.utc)
            if as_of is None
            else as_of
        )
        current_time = pd.Timestamp(now)
        if current_time.tzinfo is None:
            raise ValueError("history as_of must be timezone-aware")
        current_new_york_date = current_time.tz_convert(
            _NEW_YORK
        ).normalize()

        # Strategy signals use completed daily bars
        # only. Current prices are fetched separately
        # for execution and rebalancing.
        history = self.market_data.get_history(
            self.symbols,
            start=(
                current_new_york_date
                - pd.Timedelta(
                    days=self.lookback_days
                )
            ),
            end=current_new_york_date,
        )

        return filter_completed_daily_history(
            history,
            as_of=current_time,
        )


    def preview_cycle(self):
        """Build and risk-check one paper plan without submitting orders."""
        account, broker_positions = self.sync_portfolio()
        clock = self.broker.get_clock()
        if not clock.is_open:
            return self._empty_cycle_result("PREVIEW_MARKET_CLOSED")

        history = self.load_history(
            as_of=getattr(clock, "timestamp", None),
        )
        if self._uses_notional_orders():
            notional_state = self._notional_account_state(
                account,
                broker_positions,
            )
            plan = self.trading_engine.preview_notional_broker_cycle(
                history=history,
                **notional_state,
            )
            execution_prices = None
            current_weights = {
                symbol: value / notional_state["account_equity"]
                for symbol, value in notional_state[
                    "current_position_market_values"
                ].items()
            }
        else:
            execution_prices = self._load_execution_prices(
                now=getattr(clock, "timestamp", None),
            )
            plan = self.trading_engine.preview_broker_cycle(
                history,
                execution_prices=execution_prices,
            )
            current_weights = self.trading_engine.rebalancer.current_weights(
                self.trading_engine.portfolio,
                execution_prices,
            )
        strategy = self.trading_engine.strategy
        result = {
            "status": "PREVIEW",
            "signal_date": getattr(strategy, "last_signal_date", None),
            "eligible_symbols": len(getattr(strategy, "last_scores", {})),
            "ranking": getattr(strategy, "last_ranking", pd.DataFrame()).copy(),
            "target_weights": plan["target_weights"],
            "current_weights": current_weights,
            "orders": plan["orders"],
            "accepted_orders": plan["accepted_orders"],
            "rejected_orders": plan["rejected_orders"],
            "execution_prices": execution_prices,
            "order_updates": [],
        }
        for key in (
            "account_equity",
            "current_position_market_values",
            "target_notionals",
            "delta_notionals",
        ):
            if key in plan:
                result[key] = plan[key]
        return result


    def _load_execution_prices(self, now=None):
        batch_loader = getattr(
            self.market_data,
            "get_execution_prices",
            None,
        )
        if callable(batch_loader):
            return batch_loader(self.symbols, now=now)
        return {
            symbol: self.market_data.get_latest_price(symbol)
            for symbol in self.symbols
        }


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

        history = self.load_history(
            as_of=getattr(clock, "timestamp", None),
        )

        if history.empty:

            raise ValueError(
                "market history is empty"
            )

        if self._uses_notional_orders():
            account, broker_positions = self.sync_portfolio()
            notional_state = self._notional_account_state(
                account,
                broker_positions,
            )
            execution_prices = None
        else:
            notional_state = None
            execution_prices = self._load_execution_prices(
                now=getattr(clock, "timestamp", None),
            )


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

            if notional_state is not None:
                result = self.trading_engine.run_notional_broker_cycle(
                    history=history,
                    cycle_key=cycle_key,
                    **notional_state,
                )
            else:
                result = self.trading_engine.run_broker_cycle(
                    history,
                    cycle_key=cycle_key,
                    execution_prices=execution_prices
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

        cycle_result = {
            "status": "COMPLETED",
            "cycle_key": cycle_key,
            "target_weights":
                result["target_weights"],
            "orders":
                result["orders"],
            "accepted_orders":
                result.get("accepted_orders", []),
            "rejected_orders":
                result.get("rejected_orders", []),
            "order_updates":
                order_updates
        }
        for key in (
            "account_equity",
            "current_position_market_values",
            "target_notionals",
            "delta_notionals",
        ):
            if key in result:
                cycle_result[key] = result[key]
        return cycle_result

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

        if self.execution_window == "after_open":
            timestamp = pd.Timestamp(getattr(clock, "timestamp", None))
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise ValueError(
                    "broker clock timestamp must be timezone-aware"
                )
            current_new_york = timestamp.tz_convert(_NEW_YORK)
            session_open = (
                current_new_york.normalize()
                + pd.Timedelta(hours=9, minutes=30)
            )
            seconds_after_open = (
                current_new_york - session_open
            ).total_seconds()
            return (
                0
                <= seconds_after_open
                <= self.minutes_after_open * 60
            )

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

                if result["status"] == "COMPLETED":
                    print("Proposed orders:", len(result["orders"]))
                    print(
                        "Accepted orders:",
                        len(result.get("accepted_orders", []))
                    )
                    print(
                        "Risk rejections:",
                        len(result.get("rejected_orders", []))
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
