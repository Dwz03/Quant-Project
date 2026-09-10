import numpy as np
import pandas as pd
import pytest

from src.research.experiment import ExperimentConfig
from src.research.momentum_experiment import (
    EQUAL_WEIGHT_BENCHMARK_NAME,
    MOMENTUM_NAME,
    SPY_BENCHMARK_NAME,
    compare_momentum_with_benchmarks,
    run_momentum_experiment,
)


def momentum_result():
    index = pd.bdate_range("2021-01-01", periods=40)
    time = np.arange(len(index))
    prices = pd.DataFrame(
        {
            "SPY": 100.0 + 0.30 * time + np.sin(time / 3.0),
            "QQQ": 110.0 + 0.40 * time + np.sin(time / 2.0),
            "AAPL": 120.0 + 0.50 * time + np.cos(time / 3.0),
            "MSFT": 130.0 + 0.35 * time + np.cos(time / 2.0),
        },
        index=index,
    )
    return run_momentum_experiment(
        data=prices,
        config=ExperimentConfig(
            name="momentum-benchmark-test",
            train_ratio=0.5,
            validation_ratio=0.25,
            transaction_cost_rate=0.001,
        ),
        lookbacks=[2, 4],
        target_weights=[0.25],
        allow_short=False,
    )


def test_benchmarks_use_exact_momentum_validation_and_test_indexes():
    result = momentum_result()
    benchmarks = compare_momentum_with_benchmarks(result)

    for evaluation in benchmarks.validation.values():
        assert evaluation.net_returns.index.equals(
            result.experiment.validation.net_returns.index
        )

    for evaluation in benchmarks.test.values():
        assert evaluation.net_returns.index.equals(
            result.experiment.test.net_returns.index
        )


def test_equal_weight_benchmark_is_fully_invested_at_constant_equal_weights():
    result = momentum_result()
    benchmarks = compare_momentum_with_benchmarks(result)

    for split_results in (benchmarks.validation, benchmarks.test):
        evaluation = split_results[EQUAL_WEIGHT_BENCHMARK_NAME]
        assert evaluation.positions.eq(0.25).all().all()
        assert evaluation.positions.sum(axis=1).eq(1.0).all()


def test_spy_buy_and_hold_uses_only_aligned_spy_prices():
    result = momentum_result()
    benchmarks = compare_momentum_with_benchmarks(result)
    splits = result.experiment.splits

    for evaluation in (
        benchmarks.validation[SPY_BENCHMARK_NAME],
        benchmarks.test[SPY_BENCHMARK_NAME],
    ):
        assert evaluation.positions["SPY"].eq(1.0).all()
        assert evaluation.positions.drop(columns="SPY").eq(0.0).all().all()

    validation_prices = pd.concat([splits.train.iloc[-1:], splits.validation])
    expected_validation_returns = validation_prices["SPY"].pct_change().iloc[1:]
    pd.testing.assert_series_equal(
        benchmarks.validation[SPY_BENCHMARK_NAME].gross_returns,
        expected_validation_returns.rename("strategy_return"),
    )


def test_benchmark_transaction_costs_use_experiment_rate_and_restart_each_split():
    result = momentum_result()
    benchmarks = compare_momentum_with_benchmarks(result)

    for split_results in (benchmarks.validation, benchmarks.test):
        for name in (EQUAL_WEIGHT_BENCHMARK_NAME, SPY_BENCHMARK_NAME):
            evaluation = split_results[name]
            assert evaluation.turnover.iloc[0] == pytest.approx(1.0)
            assert evaluation.turnover.iloc[1:].eq(0.0).all()
            assert evaluation.costs.iloc[0] == pytest.approx(0.001)
            assert evaluation.costs.iloc[1:].eq(0.0).all()


def test_benchmark_comparison_does_not_change_momentum_selection():
    result = momentum_result()
    selected_before = dict(result.experiment.parameters)
    candidates_before = result.candidate_results.copy(deep=True)

    benchmarks = compare_momentum_with_benchmarks(result)

    assert result.experiment.parameters == selected_before
    pd.testing.assert_frame_equal(result.candidate_results, candidates_before)
    assert set(benchmarks.validation) == {
        MOMENTUM_NAME,
        EQUAL_WEIGHT_BENCHMARK_NAME,
        SPY_BENCHMARK_NAME,
    }
    assert set(benchmarks.test) == {
        MOMENTUM_NAME,
        EQUAL_WEIGHT_BENCHMARK_NAME,
        SPY_BENCHMARK_NAME,
    }


def test_comparison_table_contains_all_strategies_for_both_splits():
    benchmarks = compare_momentum_with_benchmarks(momentum_result())

    assert benchmarks.comparison.index.names == ["split", "strategy"]
    for split in ("validation", "test"):
        assert benchmarks.comparison.xs(split, level="split").index.tolist() == [
            MOMENTUM_NAME,
            EQUAL_WEIGHT_BENCHMARK_NAME,
            SPY_BENCHMARK_NAME,
        ]
    assert benchmarks.comparison.columns.tolist() == [
        "gross_return",
        "net_return",
        "volatility",
        "sharpe",
        "max_drawdown",
        "turnover",
        "transaction_cost",
    ]


def test_spy_benchmark_requires_spy_in_existing_aligned_data():
    result = momentum_result()
    for split in (
        result.experiment.splits.train,
        result.experiment.splits.validation,
        result.experiment.splits.test,
    ):
        split.drop(columns="SPY", inplace=True)

    with pytest.raises(ValueError, match="SPY is not in the experiment data"):
        compare_momentum_with_benchmarks(result)
