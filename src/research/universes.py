"""Versioned research universes with explicit historical limitations.

The large universe is a fixed present-day list assembled for liquid US-equity
research. It is not point-in-time constituent data. Historical use can therefore
contain survivorship, availability, and present-day sector-classification bias.
"""


LARGE_LIQUID_US_EQUITIES_V1_BY_SECTOR = {
    "Communication Services": (
        "GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ", "T", "CHTR",
    ),
    "Consumer Discretionary": (
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG",
    ),
    "Consumer Staples": (
        "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ",
    ),
    "Energy": (
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY",
    ),
    "Financials": (
        "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "AXP",
    ),
    "Health Care": (
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN",
    ),
    "Industrials": (
        "GE", "CAT", "RTX", "HON", "UNP", "UPS", "BA", "DE",
    ),
    "Information Technology": (
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE",
    ),
    "Materials": (
        "LIN", "SHW", "ECL", "APD", "FCX", "NEM", "NUE", "DOW",
    ),
    "Real Estate": (
        "PLD", "AMT", "EQIX", "WELL", "SPG", "O", "PSA", "CCI",
    ),
    "Utilities": (
        "NEE", "SO", "DUK", "CEG", "AEP", "SRE", "EXC", "XEL",
    ),
}

LARGE_LIQUID_US_EQUITIES_V1 = tuple(
    symbol
    for symbols in LARGE_LIQUID_US_EQUITIES_V1_BY_SECTOR.values()
    for symbol in symbols
)

UNIVERSES = {
    "large": LARGE_LIQUID_US_EQUITIES_V1,
}

UNIVERSE_METADATA = {
    "large": {
        "version": "large-liquid-us-equities-v1",
        "as_of": "2026-09-11",
        "point_in_time": False,
        "contains_survivorship_bias": True,
        "sector_count": len(LARGE_LIQUID_US_EQUITIES_V1_BY_SECTOR),
        "number_of_symbols": len(LARGE_LIQUID_US_EQUITIES_V1),
    },
}
