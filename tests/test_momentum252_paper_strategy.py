from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from alpaca.data.enums import DataFeed

from scripts import run_paper_strategy as paper_runner
from src.execution import ExecutionHandler
from src.market_data import AlpacaMarketDataAdapter, MarketDataError
from src.paper_trading_engine import PaperTradingEngine, filter_completed_daily_history
from src.portfolio import Portfolio
from src.rebalancer import Rebalancer
from src.research.features import build_momentum_features
from src.research.universes import LARGE_LIQUID_US_EQUITIES_V1
from src.risk_manager import RiskManager
from src.strategy import Momentum252Strategy
from src.strategy_factory import build_strategy
from src.trading_engine import TradingEngine


SYMBOLS = tuple(LARGE_LIQUID_US_EQUITIES_V1)


def close_history(end='2026-09-10', periods=270):
    dates = pd.bdate_range(end=end, periods=periods)
    progress = np.linspace(0.0, 1.0, periods)
    return pd.DataFrame({
        symbol: 100.0 * (1.0 + number * 0.001 * progress)
        for number, symbol in enumerate(SYMBOLS, start=1)
    }, index=dates)


class _Clock:
    is_open = True
    timestamp = pd.Timestamp('2026-09-11 10:00', tz='America/New_York')


class _NoSubmitBroker:
    is_paper = True

    def __init__(self, positions=None):
        self.positions = positions or []
        self.submit_count = 0

    def get_clock(self):
        return _Clock()

    def get_account(self):
        return SimpleNamespace(cash='100000', equity='100000', buying_power='100000')

    def get_positions(self):
        return self.positions

    def can_short(self, symbol):
        return False

    def supports_notional_order(self, symbol):
        return True

    def submit_order(self, order):
        self.submit_count += 1
        raise AssertionError('preview must not submit orders')


class _HistoryOnlyMarketData:
    execution_feed = DataFeed.SIP

    def __init__(self):
        completed = close_history()
        current = completed.iloc[[-1]].copy() * 100.0
        current.index = pd.DatetimeIndex(['2026-09-11'])
        self.returned_history = pd.concat([completed, current])
        self.requested_start = None

    def get_history(self, symbols, start, end):
        self.requested_start = start
        self.requested_end = end
        return self.returned_history

    def get_latest_price(self, symbol):
        raise AssertionError('notional preview must not load execution trades')


def test_score_matches_frozen_research_formula_without_mutating_input():
    history = close_history()
    original = history.copy(deep=True)
    strategy = Momentum252Strategy()

    scores = strategy.calculate_scores(history)
    symbol = SYMBOLS[10]
    expected = build_momentum_features(
        history[symbol].to_frame(name='Close')
    )['momentum_252'].iloc[-1]

    assert scores[symbol] == pytest.approx(expected)
    assert strategy.feature_horizon == 252
    assert strategy.selection_fraction == pytest.approx(0.20)
    pd.testing.assert_frame_equal(history, original)


def test_full_universe_selects_exactly_top_17_equal_weight_symbols():
    strategy = Momentum252Strategy()
    weights = strategy.generate_target_weights(close_history())
    selected = {symbol for symbol, weight in weights.items() if weight > 0}

    assert len(weights) == 88
    assert selected == set(SYMBOLS[-17:])
    assert len(selected) == 17
    assert all(weights[symbol] == pytest.approx(1 / 17) for symbol in selected)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert strategy.last_ranking.iloc[0]['symbol'] == SYMBOLS[-1]


def test_future_prices_cannot_change_prior_signal_or_selection():
    history = close_history(periods=280)
    signal_date = history.index[-6]
    changed = history.copy()
    changed.loc[changed.index > signal_date] *= np.linspace(
        2.0, 100.0, len(SYMBOLS)
    )

    original = Momentum252Strategy().generate_target_weights(
        history.loc[:signal_date]
    )
    recalculated = Momentum252Strategy().generate_target_weights(
        changed.loc[:signal_date]
    )

    assert recalculated == original


def test_incomplete_or_stale_symbol_history_is_excluded_without_fill():
    history = close_history()
    insufficient = SYMBOLS[-1]
    missing_current = SYMBOLS[-2]
    history.loc[history.index[:20], insufficient] = np.nan
    history.loc[history.index[-1], missing_current] = np.nan

    strategy = Momentum252Strategy()
    weights = strategy.generate_target_weights(history)

    assert insufficient not in strategy.last_scores
    assert missing_current not in strategy.last_scores
    assert weights[insufficient] == 0.0
    assert weights[missing_current] == 0.0
    with pytest.raises(ValueError, match='insufficient completed history'):
        Momentum252Strategy().generate_target_weights(history.iloc[:252])


def test_factory_enforces_frozen_universe_and_preserves_existing_strategies():
    strategy = build_strategy('momentum_252', symbols=list(SYMBOLS))

    assert isinstance(strategy, Momentum252Strategy)
    assert strategy.name == 'Momentum 252 V1'
    assert build_strategy('volatility_20').name == 'Volatility 20 V1'
    assert build_strategy('momentum').name == 'Momentum'
    with pytest.raises(ValueError, match='exact frozen 88-stock universe'):
        build_strategy('momentum_252', symbols=list(SYMBOLS[:-1]))


def _preview(positions=None):
    broker = _NoSubmitBroker(positions)
    market_data = _HistoryOnlyMarketData()
    strategy = Momentum252Strategy()
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
        lookback_days=paper_runner.MOMENTUM_252_LOOKBACK_DAYS,
        state_file=None,
        execution_window='after_open',
        minutes_after_open=5,
    )
    return broker, market_data, engine.preview_cycle()


def test_preview_uses_completed_close_and_never_submits_orders():
    broker, market_data, preview = _preview()

    assert preview['status'] == 'PREVIEW'
    assert preview['signal_date'] == pd.Timestamp('2026-09-10')
    assert preview['eligible_symbols'] == 88
    assert len(preview['orders']) == 17
    assert len(preview['accepted_orders']) == 17
    assert preview['rejected_orders'] == []
    assert preview['execution_prices'] is None
    assert all(order.quantity is None for order in preview['orders'])
    assert all(
        order.notional == pytest.approx(100000 / 17)
        for order in preview['orders']
    )
    assert market_data.requested_end == _Clock.timestamp.normalize()
    assert broker.submit_count == 0


def test_preview_reconciles_existing_positions_and_builds_delta_orders():
    selected_symbol = SYMBOLS[-1]
    dropped_symbol = SYMBOLS[0]
    positions = [
        SimpleNamespace(
            symbol=selected_symbol,
            qty='10',
            avg_entry_price='100',
            market_value='1000',
        ),
        SimpleNamespace(
            symbol=dropped_symbol,
            qty='20',
            avg_entry_price='100',
            market_value='2000',
        ),
    ]

    broker, _, preview = _preview(positions)
    orders = {order.symbol: order for order in preview['orders']}

    assert orders[selected_symbol].side == 'BUY'
    assert orders[selected_symbol].notional == pytest.approx(100000 / 17 - 1000)
    assert orders[dropped_symbol].side == 'SELL'
    assert orders[dropped_symbol].notional == pytest.approx(2000)
    assert preview['current_position_market_values'] == {
        selected_symbol: 1000.0,
        dropped_symbol: 2000.0,
    }
    assert broker.submit_count == 0


def test_frozen_runner_profile_matches_risk_history_feed_and_execution_rules():
    strategy = Momentum252Strategy()

    assert paper_runner._max_position_pct_for_strategy('momentum_252') == 0.06
    assert paper_runner._lookback_days_for_strategy('momentum_252', strategy) == 400
    assert paper_runner._execution_feed_for_strategy('momentum_252') == DataFeed.SIP
    assert paper_runner._execution_window_for_strategy('momentum_252') == 'after_open'

    engine = PaperTradingEngine(
        broker=object(),
        market_data=object(),
        trading_engine=object(),
        symbols=list(SYMBOLS),
        state_file=None,
        execution_window='after_open',
        minutes_after_open=5,
    )
    open_clock = lambda value: SimpleNamespace(
        is_open=True,
        timestamp=pd.Timestamp(value, tz='America/New_York'),
    )
    assert engine._should_run_now(open_clock('2026-09-11 09:32'))
    assert not engine._should_run_now(open_clock('2026-09-11 09:36'))


def test_current_session_bar_is_excluded_before_momentum_signal():
    market_data = _HistoryOnlyMarketData()
    filtered = filter_completed_daily_history(
        market_data.returned_history,
        as_of=_Clock.timestamp,
    )

    assert filtered.index.max() == pd.Timestamp('2026-09-10')
    expected = Momentum252Strategy().generate_target_weights(
        market_data.returned_history.loc[:'2026-09-10']
    )
    actual = Momentum252Strategy().generate_target_weights(filtered)
    assert actual == expected


def test_stale_sip_execution_mark_fails_closed():
    now = datetime(2026, 9, 11, 14, 32, tzinfo=timezone.utc)
    stale_trade = SimpleNamespace(
        price=100.0,
        timestamp=now - timedelta(seconds=301),
    )

    class TradeClient:
        def get_stock_latest_trade(self, request):
            return {SYMBOLS[0]: stale_trade}

    adapter = AlpacaMarketDataAdapter(
        TradeClient(),
        max_staleness_seconds=300,
        execution_feed=DataFeed.SIP,
    )
    with pytest.raises(MarketDataError, match='stale'):
        adapter.get_latest_price(SYMBOLS[0], now=now)


def test_preview_output_includes_momentum_scores_and_safety_sections(capsys):
    _, _, preview = _preview()

    paper_runner._print_preview(preview)
    paper_runner._print_momentum_252_risk_profile()
    output = capsys.readouterr().out

    assert 'NO ORDERS SUBMITTED' in output
    assert 'Completed signal date: 2026-09-10' in output
    assert 'Eligible symbols: 88' in output
    assert 'Selected symbols: 17' in output
    assert 'momentum_252' in output
    assert 'Current portfolio weights:' in output
    assert 'Target notionals:' in output
    assert 'Delta notionals:' in output
    assert 'Proposed orders:' in output
    assert 'Accepted orders:' in output
    assert 'Risk rejections:' in output
    assert 'max_position_pct: 6.00%' in output
    assert 'max_leverage: 1.00' in output
