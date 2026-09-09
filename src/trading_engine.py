from .portfolio import Portfolio
from .risk_manager import RiskManager
from .execution import ExecutionHandler
from .rebalancer import Rebalancer
from .strategy import MomentumStrategy
from .events import SignalEvent, OrderEvent, MarketEvent, FillEvent
from collections import deque
import logging
import hashlib
import time
logger = logging.getLogger(__name__)

class TradingEngine:

    def __init__(self, portfolio, risk_manager, execution, rebalancer, strategy, broker=None):

        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.execution = execution
        self.rebalancer = rebalancer
        self.strategy = strategy
        self.broker = broker
        self.events = deque()

    def rebalance(self, target_weights, prices):

        orders = self.rebalancer.generate_orders(target_weights, self.portfolio, prices)

        requested_turnover = self.rebalancer.calculate_turnover(orders, self.portfolio, prices)

        for order in orders:

            if self.risk_manager.check_order(order, self.portfolio, prices):

                fill = self.execution.execute_order(order, prices[order.symbol])

                self.portfolio.process_fill(fill)

            else:
                    logger.warning(
                        "%s %s order rejected by risk manager",
                        order.symbol,
                        order.side)

        return {"orders": orders, "requested_turnover": requested_turnover}

    def run_broker_cycle(
        self,
        history,
        cycle_key=None
    ):

        if self.broker is None:
            raise ValueError("broker is required for broker cycle")

        if history.empty:
            raise ValueError("history cannot be empty")

        target_weights = (
            self.strategy.generate_target_weights(history)
        )

        prices = history.iloc[-1].to_dict()

        orders = self.rebalancer.generate_orders(
            target_weights,
            self.portfolio,
            prices
        )

        if cycle_key is not None:

            for order in orders:

                order.client_order_id = (
                    self._build_client_order_id(
                        cycle_key,
                        order
                    )
                )

        submitted_orders = []

        for order in orders:

            if not self.risk_manager.check_order(
                order,
                self.portfolio,
                prices
            ):

                logger.warning(
                    "%s %s rejected by risk manager",
                    order.symbol,
                    order.side
                )

                continue


            if not self._broker_allows_order(
                order
            ):

                logger.warning(
                    "%s %s rejected: "
                    "asset/account cannot short",
                    order.symbol,
                    order.side
                )

                continue


            broker_order = (
                self.broker.submit_order(
                    order
                )
            )

            submitted_orders.append(
                broker_order
            )

        return {
            "target_weights": target_weights,
            "orders": orders,
            "submitted_orders": submitted_orders
        }

    def run(self, prices):

        while self.events:

            event = self.events.popleft()

            if event.type == "MARKET":

                logger.debug(
                    "Processing MARKET event for %s",
                    event.symbol)

                signal_event = self.strategy.on_market_event(event)

                if signal_event is not None:
                    self.events.append(signal_event)

            elif event.type == "SIGNAL":

                logger.info(
                    "Signal generated: %s %s",
                    event.signal,
                    event.symbol)

                order_events = self.rebalancer.on_signal_event(event, self.portfolio, prices)

                for order_event in order_events:
                    self.events.append(order_event)

            elif event.type == "ORDER":

                order = event.order

                approved = self.risk_manager.check_order(order, self.portfolio, prices)

                if approved:
                    logger.info(
                        "Order approved: %s %s %s",
                        order.side,
                        order.quantity,
                        order.symbol)

                    fill_event = self.execution.on_order_event(event, prices)

                    if fill_event is not None:
                        self.events.append(fill_event)

                    else:
                        logger.info(
                            "Order not filled: %s %s",
                            order.side,
                            order.symbol)

                else:
                    logger.warning(
                        "%s %s order rejected by risk manager",
                        order.symbol,
                        order.side)

            elif event.type == "FILL":

                fill = event.fill

                logger.info(
                    "Fill processed: %s %s %s @ %.2f",
                    fill.side,
                    fill.quantity,
                    fill.symbol,
                    fill.price)

                self.portfolio.process_fill(fill)

            else:
                raise ValueError(f"Unknown event type: {event.type}")

    def add_event(self, event):

        if event is None:
            raise ValueError("event cannot be None")

        if not hasattr(event, "type"):
            raise TypeError("event must have a type attribute")

        self.events.append(event)

    def run_strategy_cycle(self, history):

        if history.empty:
            raise ValueError("history cannot be empty")

        target_weights = self.strategy.generate_target_weights(history)

        prices = history.iloc[-1].to_dict()

        result = self.rebalance(
            target_weights,
            prices
        )

        return {
            "target_weights": target_weights,
            "orders": result["orders"],
            "requested_turnover": result["requested_turnover"]
        }

    def reconcile_broker_orders(self, order_ids):

        if self.broker is None:
            raise ValueError("broker is required")

        order_updates = []

        for order_id in order_ids:

            broker_order = self.broker.get_order(
                order_id
            )

            status = broker_order.status.value

            filled_qty = float(
                broker_order.filled_qty or 0
            )

            if broker_order.filled_avg_price is None:
                filled_avg_price = None
            else:
                filled_avg_price = float(
                    broker_order.filled_avg_price
                )

            order_updates.append({
                "order_id": order_id,
                "symbol": broker_order.symbol,
                "status": status,
                "filled_qty": filled_qty,
                "filled_avg_price": filled_avg_price
            })

        # Broker is the source of truth
        account = self.broker.get_account()
        broker_positions = self.broker.get_positions()

        self.portfolio.sync_from_broker(
            account,
            broker_positions
        )

        return order_updates

    def wait_for_orders(
        self,
        order_ids,
        timeout=30,
        poll_interval=1
    ):

        if self.broker is None:
            raise ValueError("broker is required")

        terminal_statuses = {
            "filled",
            "canceled",
            "cancelled",
            "rejected",
            "expired",
            "done_for_day"
        }

        start_time = time.time()

        while True:

            all_terminal = True

            for order_id in order_ids:

                broker_order = self.broker.get_order(
                    order_id
                )

                status = broker_order.status.value

                if status not in terminal_statuses:
                    all_terminal = False

            if all_terminal:
                break

            if time.time() - start_time >= timeout:

                # Even on timeout, sync broker truth
                self.reconcile_broker_orders(
                    order_ids
                )

                raise TimeoutError(
                    "Timed out waiting for broker orders"
                )

            time.sleep(poll_interval)

        return self.reconcile_broker_orders(
            order_ids
        )

    def _build_client_order_id(
        self,
        cycle_key,
        order
    ):

        strategy_name = getattr(
            self.strategy,
            "name",
            "strategy"
        )

        raw_id = (
            f"{cycle_key}|"
            f"{strategy_name}|"
            f"{order.symbol}"
        )

        digest = hashlib.sha256(
            raw_id.encode()
        ).hexdigest()[:16]

        date_key = cycle_key.replace(
            "-",
            ""
        )

        return (
            f"qt-{date_key}-{digest}"
        )

    def _requires_shorting(
        self,
        order
    ):

        if order.side != "SELL":
            return False

        position = self.portfolio.get_position(
            order.symbol
        )

        current_quantity = (
            0
            if position is None
            else position.quantity
        )

        projected_quantity = (
            current_quantity
            - order.quantity
        )

        return projected_quantity < 0

    def _broker_allows_order(
        self,
        order
    ):

        if not self._requires_shorting(
            order
        ):
            return True

        can_short = getattr(
            self.broker,
            "can_short",
            None
        )

        # Fail closed:
        # if broker cannot verify shortability,
        # do not open a short.
        if can_short is None:
            return False

        return can_short(
            order.symbol
        )

