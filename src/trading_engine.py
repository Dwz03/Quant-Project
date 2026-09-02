from .portfolio import Portfolio
from .risk_manager import RiskManager
from .execution import ExecutionHandler
from .rebalancer import Rebalancer
from .strategy import MomentumStrategy
from .events import SignalEvent, OrderEvent, MarketEvent, FillEvent
from collections import deque

class TradingEngine:

    def __init__(self, portfolio, risk_manager, execution, rebalancer, strategy):

        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.execution = execution
        self.rebalancer = rebalancer
        self.strategy = strategy
        self.events = deque()

    def rebalance(self, target_weights, prices):

        orders = self.rebalancer.generate_orders(target_weights, self.portfolio, prices)

        requested_turnover = self.rebalancer.calculate_turnover(orders, self.portfolio, prices)

        for order in orders:

            if self.risk_manager.check_order(order, self.portfolio, prices):

                fill = self.execution.execute_order(order, prices[order.symbol])

                self.portfolio.process_fill(fill)

            else:
                print(f"{order.symbol} {order.side} rejected")

        return {"orders": orders, "requested_turnover": requested_turnover}

    def run(self, prices):

        while self.events:

            event = self.events.popleft()

            if event.type == "MARKET":

                signal_event = self.strategy.on_market_event(event)

                if signal_event is not None:
                    self.events.append(signal_event)

            elif event.type == "SIGNAL":

                order_events = self.rebalancer.on_signal_event(event, self.portfolio, prices)

                for order_event in order_events:
                    self.events.append(order_event)

            elif event.type == "ORDER":

                order = event.order

                approved = self.risk_manager.check_order(order, self.portfolio, prices)

                if approved:

                    fill_event = self.execution.on_order_event(event, prices)

                    self.events.append(fill_event)

                else:
                    print(f"{order.symbol} {order.side} rejected")

            elif event.type == "FILL":

                fill = event.fill

                self.portfolio.process_fill(fill)

    def add_event(self, event):

        self.events.append(event)