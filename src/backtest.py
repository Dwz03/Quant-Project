def add_signal(data):

    data["Signal"] = 0

    data.loc[data["Return"] > 0, "Signal"] = 1
    data.loc[data["Return"] < 0, "Signal"] = -1

    return data

def add_strategy_returns(data):

    data["Position"] = data["Signal"].shift(1)

    data["Strategy_Return"] = data["Position"] * data["Return"]

    return data

def add_equity_curve(data):

    data["equity"] = (1 + data["Strategy_Return"]).cumprod()

    return data

def add_buy_hold_equity(data):

    data["Buy_Hold_Equity"] = (1 + data["Return"]).cumprod()

    return data
