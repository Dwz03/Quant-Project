
class Backtester:

    def __init__(self, data):
        self.data = data.copy()

    def add_strategy_returns(self):

        self.data["Position"] = self.data["Signal"].shift(1).fillna(0)

        self.data["Strategy_Return"] = (self.data["Position"] * self.data["Return"])

    def add_buy_hold_equity(self):

        self.data["Buy_Hold_Equity"] = (1 + self.data["Return"]).cumprod()

    def add_equity_curve(self):

        self.data["equity"] = (1 + self.data["Strategy_Return"]).cumprod()

    def add_turnover(self):

        self.data["Turnover"] = self.data["Position"].diff().abs().fillna(0)

    def add_transaction_cost(self, cost_rate):

        self.data["Transaction_Cost"] = self.data["Turnover"] * cost_rate

    def add_net_equity_curve(self):

        self.data["Net_Equity"] = (1 + self.data["Net_Strategy_Return"]).cumprod()

    def add_net_strategy_returns(self):

        self.data["Net_Strategy_Return"] = self.data["Strategy_Return"] - self.data["Transaction_Cost"] - self.data["Slippage"]

    def add_slippage(self, slippage_rate):

        self.data["Slippage"] = self.data["Turnover"] * slippage_rate

    def run(self, cost_rate, slippage_rate):
       
        self.add_strategy_returns()
        self.add_equity_curve()
        self.add_buy_hold_equity()

        self.add_turnover()
        self.add_transaction_cost(cost_rate)
        self.add_slippage(slippage_rate)
        self.add_net_strategy_returns()
        self.add_net_equity_curve()

        return self.data



