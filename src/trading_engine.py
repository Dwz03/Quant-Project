from .portfolio import Portfolio
from .risk_manager import RiskManager
from .execution import ExecutionHandler
from .rebalancer import Rebalancer
from .strategy import MomentumStrategy
from .events import SignalEvent, OrderEvent, MarketEvent, FillEvent
from .fill import Fill
from collections import deque
import copy
import logging
import hashlib
import time
logger = logging.getLogger(__name__)


def _cycle_client_order_id_prefix(cycle_key):

    date_key = cycle_key.replace(
        "-",
        ""
    )

    return f"qt-{date_key}-"


class _ConservativeRiskProjection:

    def __init__(self, portfolio, prices):

        self.confirmed_quantities = {
            symbol: position.quantity
            for symbol, position
            in portfolio.positions.items()
        }
        self.pending_buy_quantities = {}
        self.pending_sell_quantities = {}
        self.available_cash = portfolio.cash
        self.equity = portfolio.total_value(prices)

    def with_order(self, order, price):

        projection = copy.deepcopy(self)

        if order.side == "BUY":
            quantities = (
                projection.pending_buy_quantities
            )
            projection.available_cash -= (
                order.quantity * price
            )
        else:
            quantities = (
                projection.pending_sell_quantities
            )

        quantities[order.symbol] = (
            quantities.get(order.symbol, 0)
            + order.quantity
        )

        return projection

    def conservative_quantities(self):

        symbols = (
            set(self.confirmed_quantities)
            | set(self.pending_buy_quantities)
            | set(self.pending_sell_quantities)
        )

        quantities = {}

        for symbol in symbols:

            confirmed = self.confirmed_quantities.get(
                symbol,
                0
            )
            buy_endpoint = (
                confirmed
                + self.pending_buy_quantities.get(
                    symbol,
                    0
                )
            )
            sell_endpoint = (
                confirmed
                - self.pending_sell_quantities.get(
                    symbol,
                    0
                )
            )

            quantities[symbol] = max(
                (
                    confirmed,
                    buy_endpoint,
                    sell_endpoint
                ),
                key=abs
            )

        return quantities


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
        cycle_key=None,
        execution_prices=None
    ):

        if self.broker is None:
            raise ValueError("broker is required for broker cycle")

        plan = self.preview_broker_cycle(
            history,
            execution_prices=execution_prices,
        )

        if cycle_key is not None:

            for order in plan["accepted_orders"]:

                order.client_order_id = (
                    self._build_client_order_id(
                        cycle_key,
                        order
                    )
                )

        for rejection in plan["rejected_orders"]:
            order = rejection["order"]
            logger.warning(
                "%s %s rejected: %s",
                order.symbol,
                order.side,
                rejection["reason"],
            )

        submitted_orders = []
        for order in plan["accepted_orders"]:
            broker_order = self.broker.submit_order(order)
            submitted_orders.append(broker_order)

        return {
            "target_weights": plan["target_weights"],
            "orders": plan["orders"],
            "accepted_orders": plan["accepted_orders"],
            "rejected_orders": plan["rejected_orders"],
            "submitted_orders": submitted_orders,
        }

    def preview_broker_cycle(
        self,
        history,
        execution_prices=None,
    ):
        """Plan and risk-check a broker cycle without submitting orders."""
        if history.empty:
            raise ValueError("history cannot be empty")

        target_weights = self.strategy.generate_target_weights(history)
        prices = (
            history.iloc[-1].to_dict()
            if execution_prices is None
            else dict(execution_prices)
        )
        orders = self.rebalancer.generate_orders(
            target_weights,
            self.portfolio,
            prices,
        )
        order_symbols = [order.symbol for order in orders]
        if len(order_symbols) != len(set(order_symbols)):
            raise ValueError(
                "broker batch cannot contain duplicate symbols"
            )

        accepted_orders = []
        rejected_orders = []
        cumulative_fill_projection = copy.deepcopy(
            self.portfolio
        )
        conservative_risk_projection = (
            _ConservativeRiskProjection(
                self.portfolio,
                prices
            )
        )

        for order in orders:

            conservative_candidate = (
                conservative_risk_projection
                .with_order(
                    order,
                    prices[order.symbol]
                )
            )

            cumulative_risk_ok = (
                self.risk_manager.check_order(
                    order,
                    cumulative_fill_projection,
                    prices
                )
            )

            conservative_risk_ok = (
                self._check_conservative_projection(
                    conservative_candidate,
                    prices
                )
            )

            if not (
                cumulative_risk_ok
                and conservative_risk_ok
            ):
                rejected_orders.append({
                    "order": order,
                    "reason": "risk manager",
                })
                continue


            if not self._broker_allows_order(
                order,
                portfolio=cumulative_fill_projection
            ):
                rejected_orders.append({
                    "order": order,
                    "reason": "asset/account cannot short",
                })
                continue

            self._apply_projected_fill(
                cumulative_fill_projection,
                order,
                prices[order.symbol]
            )
            conservative_risk_projection = (
                conservative_candidate
            )
            accepted_orders.append(order)

        return {
            "target_weights": target_weights,
            "orders": orders,
            "accepted_orders": accepted_orders,
            "rejected_orders": rejected_orders,
            "execution_prices": prices,
        }

    def run_notional_broker_cycle(
        self,
        history,
        account_equity,
        current_position_market_values,
        available_buying_power,
        cycle_key=None,
    ):
        if self.broker is None:
            raise ValueError("broker is required for broker cycle")

        plan = self.preview_notional_broker_cycle(
            history=history,
            account_equity=account_equity,
            current_position_market_values=current_position_market_values,
            available_buying_power=available_buying_power,
        )
        if cycle_key is not None:
            for order in plan["accepted_orders"]:
                order.client_order_id = self._build_client_order_id(
                    cycle_key,
                    order,
                )

        for rejection in plan["rejected_orders"]:
            order = rejection["order"]
            logger.warning(
                "%s %s notional order rejected: %s",
                order.symbol,
                order.side,
                rejection["reason"],
            )

        submitted_orders = [
            self.broker.submit_order(order)
            for order in plan["accepted_orders"]
        ]
        return {
            **plan,
            "submitted_orders": submitted_orders,
        }

    def preview_notional_broker_cycle(
        self,
        history,
        account_equity,
        current_position_market_values,
        available_buying_power,
    ):
        """Plan a long-only broker rebalance without external prices."""
        if history.empty:
            raise ValueError("history cannot be empty")

        target_weights = self.strategy.generate_target_weights(history)
        current_values = {
            symbol: float(value)
            for symbol, value in current_position_market_values.items()
        }
        all_symbols = sorted(set(target_weights) | set(current_values))
        target_notionals = {
            symbol: account_equity * float(target_weights.get(symbol, 0.0))
            for symbol in all_symbols
        }
        delta_notionals = {
            symbol: target_notionals[symbol] - current_values.get(symbol, 0.0)
            for symbol in all_symbols
        }
        orders = self.rebalancer.generate_notional_orders(
            target_weights=target_weights,
            account_equity=account_equity,
            current_market_values=current_values,
        )
        order_symbols = [order.symbol for order in orders]
        if len(order_symbols) != len(set(order_symbols)):
            raise ValueError("broker batch cannot contain duplicate symbols")

        required_symbols = {
            symbol
            for symbol, weight in target_weights.items()
            if weight > 0
        } | set(order_symbols)
        self._validate_notional_assets(required_symbols)

        accepted_orders = []
        rejected_orders = []
        projected_values = dict(current_values)
        remaining_buying_power = float(available_buying_power)
        for order in orders:
            risk_ok = self.risk_manager.check_notional_order(
                order=order,
                account_equity=account_equity,
                current_position_market_values=projected_values,
                available_buying_power=remaining_buying_power,
            )
            if not risk_ok:
                rejected_orders.append({
                    "order": order,
                    "reason": "risk manager",
                })
                continue

            current_value = projected_values.get(order.symbol, 0.0)
            if order.side == "BUY":
                projected_values[order.symbol] = current_value + order.notional
                remaining_buying_power -= order.notional
            else:
                projected_values[order.symbol] = max(
                    0.0,
                    current_value - order.notional,
                )
            accepted_orders.append(order)

        return {
            "target_weights": target_weights,
            "account_equity": account_equity,
            "current_position_market_values": current_values,
            "target_notionals": target_notionals,
            "delta_notionals": delta_notionals,
            "orders": orders,
            "accepted_orders": accepted_orders,
            "rejected_orders": rejected_orders,
            "execution_prices": None,
        }

    def _validate_notional_assets(self, symbols):
        supports_notional = getattr(
            self.broker,
            "supports_notional_order",
            None,
        )
        if not callable(supports_notional):
            raise RuntimeError(
                "broker cannot verify tradable/fractionable notional eligibility"
            )

        unsupported = [
            symbol
            for symbol in sorted(symbols)
            if supports_notional(symbol) is not True
        ]
        if unsupported:
            raise RuntimeError(
                "required symbols are not tradable and fractionable for "
                f"notional orders: {', '.join(unsupported)}"
            )

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

        return (
            f"{_cycle_client_order_id_prefix(cycle_key)}"
            f"{digest}"
        )

    def _requires_shorting(
        self,
        order,
        portfolio=None
    ):

        if order.side != "SELL":
            return False

        if portfolio is None:
            portfolio = self.portfolio

        position = portfolio.get_position(
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
        order,
        portfolio=None
    ):

        if not self._requires_shorting(
            order,
            portfolio=portfolio
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

    def _check_conservative_projection(
        self,
        projection,
        prices
    ):

        if projection.available_cash < 0:
            return False

        if projection.equity <= 0:
            return False

        quantities = (
            projection.conservative_quantities()
        )

        gross_exposure = 0

        for symbol, quantity in quantities.items():

            exposure = abs(
                quantity * prices[symbol]
            )

            if exposure > (
                self.risk_manager.max_position_pct
                * projection.equity
            ):
                return False

            gross_exposure += exposure

        leverage = (
            gross_exposure
            / projection.equity
        )

        return (
            leverage
            <= self.risk_manager.max_leverage
        )

    def _apply_projected_fill(
        self,
        portfolio,
        order,
        price
    ):

        fill = Fill(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.side,
            price=price,
            commission=0.0
        )

        portfolio.process_fill(fill)
