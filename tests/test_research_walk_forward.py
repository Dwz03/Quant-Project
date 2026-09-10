import numpy as np
import pandas as pd
import pytest

from src.research.experiment import StrategyEvaluation
from src.research.validation import apply_transaction_costs
from src.research.walk_forward import (
    WalkForwardConfig,
    generate_walk_forward_folds,
    run_walk_forward,
)


def price_data(length=12):
    return pd.DataFrame({"A": np.arange(100.0, 100.0 + length)})


def constant_evaluator(historical_data, evaluation_data, parameters):
    return StrategyEvaluation(
        returns=pd.Series(0.01, index=evaluation_data.index),
        positions=pd.Series(parameters["position"], index=evaluation_data.index),
    )


def test_generate_walk_forward_folds_expands_train_and_orders_all_windows():
    folds = generate_walk_forward_folds(
        price_data(),
        WalkForwardConfig(
            initial_train_size=4,
            validation_size=2,
            test_size=2,
        ),
    )

    assert [len(fold.train) for fold in folds] == [4, 6, 8]
    assert [fold.validation.index.tolist() for fold in folds] == [
        [4, 5],
        [6, 7],
        [8, 9],
    ]
    assert [fold.test.index.tolist() for fold in folds] == [
        [6, 7],
        [8, 9],
        [10, 11],
    ]

    for fold in folds:
        assert fold.train.index[-1] < fold.validation.index[0]
        assert fold.validation.index[-1] < fold.test.index[0]

    stitched_test_index = pd.Index([
        index
        for fold in folds
        for index in fold.test.index
    ])
    assert stitched_test_index.is_unique


@pytest.mark.parametrize(
    "config",
    [
        WalkForwardConfig(initial_train_size=4, validation_size=2, test_size=2),
    ],
)
def test_generate_walk_forward_folds_rejects_insufficient_data(config):
    with pytest.raises(ValueError, match="not enough data"):
        generate_walk_forward_folds(price_data(length=7), config)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"initial_train_size": 0, "validation_size": 2, "test_size": 2}, "positive"),
        ({"initial_train_size": 4, "validation_size": -1, "test_size": 2}, "positive"),
        ({"initial_train_size": 4, "validation_size": 2, "test_size": 0}, "positive"),
        ({
            "initial_train_size": 4,
            "validation_size": 2,
            "test_size": 2,
            "step_size": 1,
        }, "must equal test_size"),
    ],
)
def test_walk_forward_config_rejects_invalid_sizes(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        WalkForwardConfig(**kwargs)


def test_run_walk_forward_selects_without_test_and_stitches_only_test_dates():
    data = price_data()
    config = WalkForwardConfig(4, 2, 2)
    folds = generate_walk_forward_folds(data, config)
    selector_calls = []

    def selector(train, validation):
        selector_calls.append((train.index.copy(), validation.index.copy()))
        return {"position": 0.25}

    result = run_walk_forward(
        data=data,
        config=config,
        evaluator=constant_evaluator,
        parameter_selector=selector,
        transaction_cost_rate=0.001,
    )

    assert len(selector_calls) == len(folds)
    for (train_index, validation_index), fold in zip(selector_calls, folds):
        assert train_index.equals(fold.train.index)
        assert validation_index.equals(fold.validation.index)
        assert not train_index.isin(fold.test.index).any()
        assert not validation_index.isin(fold.test.index).any()

    assert result.net_returns.index.tolist() == [6, 7, 8, 9, 10, 11]
    assert result.net_returns.index.is_monotonic_increasing
    assert result.net_returns.index.is_unique
    assert not result.net_returns.index.isin([4, 5]).any()
    assert len(result.fold_table) == 3


def test_walk_forward_preserves_unchanged_position_across_fold_boundary():
    result = run_walk_forward(
        data=price_data(length=10),
        config=WalkForwardConfig(4, 2, 2),
        evaluator=constant_evaluator,
        parameter_selector=lambda train, validation: {"position": 0.25},
        transaction_cost_rate=0.001,
    )

    assert result.folds[0].test.turnover.iloc[0] == pytest.approx(0.25)
    assert result.folds[1].test.turnover.iloc[0] == pytest.approx(0.0)


def test_walk_forward_charges_actual_position_change_across_fold_boundary():
    def selector(train, validation):
        return {"position": 0.25 if len(train) == 4 else 0.0}

    result = run_walk_forward(
        data=price_data(length=10),
        config=WalkForwardConfig(4, 2, 2),
        evaluator=constant_evaluator,
        parameter_selector=selector,
        transaction_cost_rate=0.001,
    )

    assert result.folds[0].test.positions.iloc[-1] == pytest.approx(0.25)
    assert result.folds[1].test.positions.iloc[0] == pytest.approx(0.0)
    assert result.folds[1].test.turnover.iloc[0] == pytest.approx(0.25)


def test_fixed_split_cost_default_still_starts_from_zero_position():
    returns = pd.Series([0.01, 0.01])
    positions = pd.Series([0.25, 0.25])

    result = apply_transaction_costs(returns, positions, cost_rate=0.001)

    assert result["turnover"].tolist() == pytest.approx([0.25, 0.0])
    assert result["cost"].tolist() == pytest.approx([0.00025, 0.0])
