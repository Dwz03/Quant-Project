import numpy as np
import pandas as pd

import src.research.momentum_experiment as momentum_experiment
from src.research.momentum_experiment import (
    EQUAL_WEIGHT_BENCHMARK_NAME,
    MOMENTUM_NAME,
    SPY_BENCHMARK_NAME,
    run_momentum_walk_forward,
)
from src.research.walk_forward import WalkForwardConfig


def multi_asset_prices(length=24):
    index = pd.RangeIndex(length)
    time = np.arange(length)
    return pd.DataFrame(
        {
            "SPY": 100.0 + 0.30 * time + np.sin(time / 3.0),
            "QQQ": 110.0 + 0.40 * time + np.sin(time / 2.0),
            "AAPL": 120.0 + 0.50 * time + np.cos(time / 3.0),
            "MSFT": 130.0 + 0.35 * time + np.cos(time / 2.0),
        },
        index=index,
    )


def walk_forward_config():
    return WalkForwardConfig(
        initial_train_size=8,
        validation_size=4,
        test_size=4,
    )


def test_momentum_walk_forward_reuses_existing_parameter_selection(monkeypatch):
    calls = []
    original_selector = momentum_experiment.select_momentum_parameters

    def recording_selector(*args, **kwargs):
        calls.append((kwargs["train_data"].index, kwargs["validation_data"].index))
        return original_selector(*args, **kwargs)

    monkeypatch.setattr(
        momentum_experiment,
        "select_momentum_parameters",
        recording_selector,
    )

    result = run_momentum_walk_forward(
        data=multi_asset_prices(),
        config=walk_forward_config(),
        transaction_cost_rate=0.001,
        lookbacks=[2, 4],
    )

    assert len(calls) == len(result.momentum.folds)
    assert "selected_lookback" in result.momentum.fold_table.columns


def test_future_final_test_prices_do_not_change_earlier_parameter_selection():
    original = multi_asset_prices()
    changed = original.copy()
    changed.iloc[20:] = changed.iloc[20:] * np.array([0.5, 1.5, 0.6, 1.4])

    original_result = run_momentum_walk_forward(
        data=original,
        config=walk_forward_config(),
        transaction_cost_rate=0.001,
        lookbacks=[2, 4],
    )
    changed_result = run_momentum_walk_forward(
        data=changed,
        config=walk_forward_config(),
        transaction_cost_rate=0.001,
        lookbacks=[2, 4],
    )

    original_parameters = [
        fold.selected_parameters for fold in original_result.momentum.folds
    ]
    changed_parameters = [
        fold.selected_parameters for fold in changed_result.momentum.folds
    ]
    assert original_parameters == changed_parameters
    assert not original_result.momentum.net_returns.equals(
        changed_result.momentum.net_returns
    )


def test_walk_forward_benchmarks_share_dates_and_position_continuity():
    result = run_momentum_walk_forward(
        data=multi_asset_prices(),
        config=walk_forward_config(),
        transaction_cost_rate=0.001,
        lookbacks=[2, 4],
    )

    for benchmark in result.benchmarks.values():
        assert benchmark.net_returns.index.equals(result.momentum.net_returns.index)
        assert benchmark.folds[1].test.turnover.iloc[0] == 0.0

    equal_weight = result.benchmarks[EQUAL_WEIGHT_BENCHMARK_NAME]
    assert equal_weight.positions.eq(0.25).all().all()

    spy = result.benchmarks[SPY_BENCHMARK_NAME]
    assert spy.positions["SPY"].eq(1.0).all()
    assert spy.positions.drop(columns="SPY").eq(0.0).all().all()
    assert result.comparison.index.tolist() == [
        MOMENTUM_NAME,
        EQUAL_WEIGHT_BENCHMARK_NAME,
        SPY_BENCHMARK_NAME,
    ]
