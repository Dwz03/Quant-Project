
class Backtester:

    def __init__(self, data):
        self.data = data.copy()

    def add_strategy_returns(self):

        self.data["Position"] = self.data["Signal"].shift(1)

        self.data["Strategy_Return"] = self.data["Position"] * self.data["Return"]

    def add_buy_hold_equity(self):

        self.data["Buy_Hold_Equity"] = (1 + self.data["Return"]).cumprod()

    def add_equity_curve(self):

        self.data["equity"] = (1 + self.data["Strategy_Return"]).cumprod()

    def run(self):

        self.add_strategy_returns()
        self.add_equity_curve()
        self.add_buy_hold_equity()

        return self.data

