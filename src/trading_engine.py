from .portfolio import Portfolio
from .risk_manager import RiskManager
from .execution import ExecutionHandler
from .rebalancer import Rebalancer

class TradingEngine:

    def __init__(self, portfolio, risk_manager, execution, rebalancer):

        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.execution = execution
        self.rebalancer = rebalancer

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