import pandas as pd
import yfinance as yf
from pathlib import Path

def get_data_path(symbol, start, end):

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / f"{symbol}_{start}_{end}.csv"

    return file_path

def clean_market_data(data):

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.sort_index()

    data = data[~data.index.duplicated(keep="first")]

    data = data.dropna(subset=["Close"])

    return data

def save_market_data(data, symbol, start, end):

    file_path = get_data_path(symbol, start, end)

    data.to_csv(file_path)

def load_local_market_data(symbol, start, end):

    file_path = get_data_path(symbol, start, end)

    if not file_path.exists():
        raise FileNotFoundError(
            f"No cached data found for {symbol}"
        )

    data = pd.read_csv(
        file_path,
        index_col="Date",
        parse_dates=["Date"]
    )

    return data


def load_market_data(symbol, start, end):

    file_path = get_data_path(symbol, start, end)

    if file_path.exists():

        data = load_local_market_data(symbol, start, end)

        return data

    print("Downloading data from Yahoo Finance...")

    data = yf.download(symbol, start = start, end = end, interval = "1d", auto_adjust = True)

    if data.empty:
        raise ValueError(f"No market data found for this symbol : {symbol}")

    clean_data = clean_market_data(data)

    save_market_data(clean_data, symbol, start, end)

    return clean_data

def add_returns(data):

    data["Return"] = data["Close"] / data["Close"].shift(1) - 1

    return data

if __name__ == "__main__":

    load_market_data(
    "AAPL",
    "2020-01-01",
    "2025-01-01")
