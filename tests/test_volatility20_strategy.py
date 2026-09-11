import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from alpaca.data.enums import DataFeed

from src.execution import ExecutionHandler
from src.market_data import AlpacaMarketDataAdapter, MarketDataError
from src.paper_trading_engine import (
    PaperTradingEngine,
    filter_completed_daily_history,
)
from src.portfolio import Portfolio
from src.rebalancer import Rebalancer
from src.risk_manager import RiskManager
from src.strategy import Volatility20Strategy
from src.strategy_factory import build_strategy
from src.trading_engine import TradingEngine
from src.research.universes import LARGE_LIQUID_US_EQUITIES_V1
from scripts import run_paper_strategy as paper_runner


SYMBOLS = tuple(LARGE_LIQUID_US_EQUITIES_V1)


def close_history(end="2026-09-10", periods=35):
    dates = pd.bdate_range(end=end, periods=periods)
    steps = np.arange(periods, dtype=float)
    values = {}
    for number, symbol in enumerate(SYMBOLS, start=1):
        amplitude = number * 0.00002
        returns = amplitude * np.where(steps % 2 == 0, 1.0, -1.0)
        values[symbol] = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame(values, index=dates)


def test_volatility_20_matches_frozen_research_formula():
    history = close_history()
    strategy = Volatility20Strategy()

    scores = strategy.calculate_scores(history)
    expected = (
        history[SYMBOLS[10]]
        .pct_change(fill_method=None)
        .rolling(20)
        .std()
        .iloc[-1]
    )

    assert scores[SYMBOLS[10]] == pytest.approx(expected)
    assert strategy.feature_window == 20
    assert strategy.selection_fraction == pytest.approx(0.20)


def test_full_universe_selects_exactly_17_highest_volatility_symbols():
    strategy = Volatility20Strategy()
    weights = strategy.generate_target_weights(close_history())
    selected = {symbol for symbol, weight in weights.items() if weight > 0}

    assert len(weights) == 88
    assert len(selected) == 17
    assert selected == set(SYMBOLS[-17:])
    assert all(weights[symbol] == pytest.approx(1 / 17) for symbol in selected)
    assert all(weights[symbol] == 0.0 for symbol in set(SYMBOLS) - selected)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert strategy.last_ranking.iloc[0]["symbol"] == SYMBOLS[-1]


def test_future_bars_cannot_change_signal_date_weights():
    history = close_history(periods=40)
    signal_date = history.index[-6]
    changed = history.copy()
    changed.loc[changed.index > signal_date] *= np.linspace(
        2.0,
        20.0,
        len(SYMBOLS),
    )

    original_weights = Volatility20Strategy().generate_target_weights(
        history.loc[:signal_date]
    )
    changed_weights = Volatility20Strategy().generate_target_weights(
        changed.loc[:signal_date]
    )

    assert original_weights == changed_weights


def test_insufficient_or_missing_completed_history_is_ineligible_without_fill():
    history = close_history()
    insufficient_symbol = SYMBOLS[-1]
    missing_current_symbol = SYMBOLS[-2]
    history.loc[history.index[10]:, insufficient_symbol] = np.nan
    history.loc[history.index[-1], missing_current_symbol] = np.nan

    strategy = Volatility20Strategy()
    weights = strategy.generate_target_weights(history)

    assert insufficient_symbol not in strategy.last_scores
    assert missing_current_symbol not in strategy.last_scores
    assert weights[insufficient_symbol] == 0.0
    assert weights[missing_current_symbol] == 0.0
    assert len(strategy.last_scores) == 86


def test_factory_selects_frozen_strategy_without_changing_existing_factory():
    strategy = build_strategy("volatility_20", symbols=list(SYMBOLS))

    assert isinstance(strategy, Volatility20Strategy)
    assert build_strategy("moving_average").name == "Moving Average"
    with pytest.raises(ValueError, match="exact frozen 88-stock universe"):
        build_strategy("volatility_20", symbols=list(SYMBOLS[:-1]))


def test_current_new_york_session_bar_is_excluded_before_signal():
    completed = close_history(end="2026-09-10")
    current_bar = completed.iloc[[-1]].copy() * np.linspace(
        10.0,
        100.0,
        len(SYMBOLS),
    )
    current_bar.index = pd.DatetimeIndex(["2026-09-11"])
    returned_history = pd.concat([completed, current_bar])

    filtered = filter_completed_daily_history(
        returned_history,
        as_of=pd.Timestamp("2026-09-11 10:00", tz="America/New_York"),
    )
    expected_weights = Volatility20Strategy().generate_target_weights(completed)
    actual_weights = Volatility20Strategy().generate_target_weights(filtered)

    assert filtered.index.max() == pd.Timestamp("2026-09-10")
    assert actual_weights == expected_weights


def test_completed_bar_filter_fails_closed_without_aware_current_time():
    with pytest.raises(ValueError, match="as_of must be timezone-aware"):
        filter_completed_daily_history(
            close_history(),
            as_of=pd.Timestamp("2026-09-11 10:00"),
        )


class _FakeClock:
    is_open = True
    timestamp = pd.Timestamp("2026-09-11 10:00", tz="America/New_York")


class _FakeAccount:
    cash = "100000"
    equity = "100000"
    buying_power = "100000"


class _NoSubmitBroker:
    is_paper = True

    def __init__(self):
        self.submit_count = 0
        self.notional_eligibility_checks = []

    def get_clock(self):
        return _FakeClock()

    def get_account(self):
        return _FakeAccount()

    def get_positions(self):
        return []

    def can_short(self, symbol):
        return False

    def supports_notional_order(self, symbol):
        self.notional_eligibility_checks.append(symbol)
        return True

    def submit_order(self, order):
        self.submit_count += 1
        raise AssertionError("preview must not submit broker orders")


class _FakeMarketData:
    def __init__(self, execution_price, execution_feed=DataFeed.IEX):
        self.execution_price = execution_price
        self.execution_feed = execution_feed
        self.execution_price_requests = 0
        self.history = close_history(end="2026-09-10")
        current = self.history.iloc[[-1]].copy() * 50.0
        current.index = pd.DatetimeIndex(["2026-09-11"])
        self.returned_history = pd.concat([self.history, current])

    def get_history(self, symbols, start, end):
        self.requested_end = end
        return self.returned_history

    def get_latest_price(self, symbol):
        self.execution_price_requests += 1
        raise AssertionError("notional preview must not request latest trades")


class _BatchMarketData(_FakeMarketData):

    def __init__(self, execution_price):
        super().__init__(execution_price)
        self.batch_calls = []

    def get_execution_prices(self, symbols, now=None):
        self.batch_calls.append((list(symbols), now))
        raise AssertionError("notional preview must not request quotes")

    def get_latest_price(self, symbol):
        raise AssertionError("batch execution-price loading was expected")


def _preview(execution_price, execution_feed=DataFeed.IEX):
    broker = _NoSubmitBroker()
    strategy = Volatility20Strategy()
    trading_engine = TradingEngine(
        Portfolio(100000),
        RiskManager(max_position_pct=0.06, max_leverage=1.0),
        ExecutionHandler(commission_rate=0.0, slippage_rate=0.0),
        Rebalancer(),
        strategy,
        broker=broker,
    )
    engine = PaperTradingEngine(
        broker=broker,
        market_data=_FakeMarketData(execution_price, execution_feed),
        trading_engine=trading_engine,
        symbols=list(SYMBOLS),
        lookback_days=60,
        state_file=None,
    )
    return broker, engine.preview_cycle()


def test_empty_volatility_preview_uses_notional_orders_and_never_submits():
    first_broker, first = _preview(100.0, DataFeed.IEX)
    second_broker, second = _preview(200.0, DataFeed.SIP)

    assert first["status"] == "PREVIEW"
    assert first["signal_date"] == pd.Timestamp("2026-09-10")
    assert first["eligible_symbols"] == 88
    assert len(first["orders"]) == 17
    assert len(first["accepted_orders"]) == 17
    assert first["rejected_orders"] == []
    assert first["target_weights"] == second["target_weights"]
    pd.testing.assert_frame_equal(
        first["ranking"],
        second["ranking"],
    )
    assert all(order.quantity is None for order in first["orders"])
    assert all(
        order.notional == pytest.approx(100000 / 17)
        for order in first["orders"]
    )
    assert [order.notional for order in first["orders"]] == pytest.approx(
        [order.notional for order in second["orders"]]
    )
    assert first["account_equity"] == pytest.approx(100000)
    assert first["current_position_market_values"] == {}
    assert first["execution_prices"] is None
    assert first_broker.submit_count == 0
    assert second_broker.submit_count == 0
    assert len(first_broker.notional_eligibility_checks) == 17
    assert len(second_broker.notional_eligibility_checks) == 17
    assert paper_runner.VOLATILITY_20_EXECUTION_FEED == DataFeed.SIP


def test_volatility_notional_preview_never_loads_execution_prices():
    broker = _NoSubmitBroker()
    market_data = _BatchMarketData(100.0)
    trading_engine = TradingEngine(
        Portfolio(100000),
        RiskManager(max_position_pct=0.06, max_leverage=1.0),
        ExecutionHandler(commission_rate=0.0, slippage_rate=0.0),
        Rebalancer(),
        Volatility20Strategy(),
        broker=broker,
    )
    engine = PaperTradingEngine(
        broker=broker,
        market_data=market_data,
        trading_engine=trading_engine,
        symbols=list(SYMBOLS),
        lookback_days=60,
        state_file=None,
    )

    preview = engine.preview_cycle()

    assert preview["status"] == "PREVIEW"
    assert market_data.batch_calls == []
    assert market_data.execution_price_requests == 0
    assert preview["execution_prices"] is None
    assert broker.submit_count == 0


def test_notional_preview_uses_broker_market_values_for_exact_deltas():
    target_notional = 100000 / 17
    underweight_symbol = SYMBOLS[-1]
    overweight_symbol = SYMBOLS[-2]
    dropped_symbol = SYMBOLS[0]

    class PositionBroker(_NoSubmitBroker):
        def get_positions(self):
            return [
                SimpleNamespace(
                    symbol=underweight_symbol,
                    qty="10",
                    avg_entry_price="100",
                    market_value="1000",
                ),
                SimpleNamespace(
                    symbol=overweight_symbol,
                    qty="70",
                    avg_entry_price="100",
                    market_value="7000",
                ),
                SimpleNamespace(
                    symbol=dropped_symbol,
                    qty="25",
                    avg_entry_price="100",
                    market_value="2500",
                ),
            ]

    broker = PositionBroker()
    market_data = _FakeMarketData(100.0)
    trading_engine = TradingEngine(
        Portfolio(100000),
        RiskManager(max_position_pct=0.06, max_leverage=1.0),
        ExecutionHandler(commission_rate=0.0, slippage_rate=0.0),
        Rebalancer(),
        Volatility20Strategy(),
        broker=broker,
    )
    preview = PaperTradingEngine(
        broker=broker,
        market_data=market_data,
        trading_engine=trading_engine,
        symbols=list(SYMBOLS),
        lookback_days=60,
        state_file=None,
    ).preview_cycle()
    orders = {order.symbol: order for order in preview["orders"]}

    assert orders[underweight_symbol].side == "BUY"
    assert orders[underweight_symbol].notional == pytest.approx(
        target_notional - 1000
    )
    assert orders[overweight_symbol].side == "SELL"
    assert orders[overweight_symbol].notional == pytest.approx(
        7000 - target_notional
    )
    assert orders[dropped_symbol].side == "SELL"
    assert orders[dropped_symbol].notional == pytest.approx(2500)
    assert preview["current_position_market_values"] == {
        underweight_symbol: 1000.0,
        overweight_symbol: 7000.0,
        dropped_symbol: 2500.0,
    }
    assert market_data.execution_price_requests == 0
    assert broker.submit_count == 0


def test_volatility_paper_risk_profile_can_express_frozen_equal_weights(capsys):
    assert paper_runner.DEFAULT_MAX_POSITION_PCT == pytest.approx(0.02)
    assert paper_runner.VOLATILITY_20_MAX_POSITION_PCT == pytest.approx(0.06)
    assert paper_runner.VOLATILITY_20_MAX_POSITION_PCT >= 1 / 17

    paper_runner._print_volatility_risk_profile()
    output = capsys.readouterr().out
    assert "approximately 5.88%" in output
    assert "max_position_pct: 6.00%" in output


def test_paper_freshness_profile_does_not_change_signal_scores_or_weights():
    history = close_history()
    strategy = Volatility20Strategy()
    expected_weights = strategy.generate_target_weights(history)
    expected_scores = strategy.last_scores.copy()
    now = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)
    trade = SimpleNamespace(
        price=123.0,
        timestamp=now - timedelta(seconds=90),
    )

    class TradeClient:
        def get_stock_latest_trade(self, request):
            return {"SBUX": trade}

    strict = AlpacaMarketDataAdapter(TradeClient())
    paper = AlpacaMarketDataAdapter(
        TradeClient(),
        max_staleness_seconds=paper_runner.PAPER_MAX_STALENESS_SECONDS,
    )
    with pytest.raises(MarketDataError):
        strict.get_latest_price("SBUX", now=now)
    assert paper.get_latest_price("SBUX", now=now) == 123.0

    actual_weights = strategy.generate_target_weights(history)
    pd.testing.assert_series_equal(strategy.last_scores, expected_scores)
    assert actual_weights == expected_weights
    assert paper_runner.PAPER_MAX_STALENESS_SECONDS == 300
    assert paper_runner.PAPER_QUOTE_MAX_STALENESS_SECONDS == 300


def test_notional_preview_does_not_touch_failing_execution_price_methods():
    broker = _NoSubmitBroker()
    strategy = Volatility20Strategy()
    market_data = _FakeMarketData(100.0)

    def stale_prices(symbols, now=None):
        raise MarketDataError(
            "Execution quote for HON is stale: "
            "quote_age=301.0s, max_allowed=300s"
        )

    market_data.get_execution_prices = stale_prices
    market_data.get_latest_price = lambda symbol: pytest.fail(
        "stale quote must not fall back to legacy single-symbol loading"
    )
    trading_engine = TradingEngine(
        Portfolio(100000),
        RiskManager(max_position_pct=0.06, max_leverage=1.0),
        ExecutionHandler(commission_rate=0.0, slippage_rate=0.0),
        Rebalancer(),
        strategy,
        broker=broker,
    )
    engine = PaperTradingEngine(
        broker=broker,
        market_data=market_data,
        trading_engine=trading_engine,
        symbols=list(SYMBOLS),
        lookback_days=60,
        state_file=None,
    )

    preview = engine.preview_cycle()

    assert preview["status"] == "PREVIEW"
    assert len(preview["orders"]) == 17
    assert broker.submit_count == 0


def test_preview_output_reports_targets_orders_and_rejections(capsys):
    _, preview = _preview(100.0)

    paper_runner._print_preview(preview)

    output = capsys.readouterr().out
    assert "PAPER PREVIEW — NO ORDERS SUBMITTED" in output
    assert "Completed signal date: 2026-09-10" in output
    assert "Eligible symbols: 88" in output
    assert "Selected symbols: 17" in output
    assert "Current portfolio weights:" in output
    assert "Broker account equity:" in output
    assert "Current broker position market values:" in output
    assert "Target notionals:" in output
    assert "Delta notionals:" in output
    assert "Proposed orders:" in output
    assert "Accepted orders:" in output
    assert "Risk rejections:" in output


def test_volatility_execution_window_is_next_session_near_open_only():
    engine = PaperTradingEngine(
        broker=object(),
        market_data=object(),
        trading_engine=object(),
        symbols=list(SYMBOLS),
        state_file=None,
        execution_window="after_open",
        minutes_after_open=5,
    )

    class OpenClock:
        is_open = True

        def __init__(self, timestamp):
            self.timestamp = pd.Timestamp(timestamp, tz="America/New_York")

    assert engine._should_run_now(OpenClock("2026-09-11 09:32"))
    assert not engine._should_run_now(OpenClock("2026-09-11 09:36"))
