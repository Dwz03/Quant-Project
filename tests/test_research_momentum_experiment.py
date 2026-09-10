import numpy as np
import pandas as pd
import pytest

import src.research.momentum_experiment as momentum_experiment
from src.research.experiment import ExperimentConfig, ExperimentResult
from src.research.momentum_experiment import (
    MomentumExperimentResult,
    run_momentum_experiment,
    select_momentum_parameters,
    summarize_momentum_experiment,
)


def synthetic_prices():
    return pd.DataFrame({
        "A": [
            100.0, 102.0, 101.0, 104.0, 103.0,
            106.0, 105.0, 108.0, 107.0, 110.0,
            109.0, 112.0, 111.0, 114.0, 113.0,
            116.0, 115.0, 118.0, 117.0, 120.0,
        ]
    })


def experiment_config():
    return ExperimentConfig(
        name="standard-momentum",
        train_ratio=0.5,
        validation_ratio=0.3,
        transaction_cost_rate=0.001,
    )


def test_select_momentum_parameters_returns_required_candidate_metrics():
    prices = synthetic_prices()

    selected, candidates = select_momentum_parameters(
        train_data=prices.iloc[:10],
        validation_data=prices.iloc[10:16],
        lookbacks=[1, 3],
        target_weights=[0.2, 0.4],
        allow_short=False,
        transaction_cost_rate=0.001,
    )

    assert len(candidates) == 4
    assert {
        "lookback",
        "target_weight",
        "validation_net_return",
        "validation_net_volatility",
        "validation_net_sharpe",
        "validation_max_drawdown",
        "validation_turnover",
    }.issubset(candidates.columns)

    best = candidates.loc[candidates["validation_net_sharpe"].idxmax()]
    assert selected == {
        "lookback": int(best["lookback"]),
        "target_weight": float(best["target_weight"]),
        "allow_short": False,
    }


def test_selection_constructs_a_fresh_existing_strategy_for_each_candidate(
    monkeypatch,
):
    prices = synthetic_prices()
    created_strategies = []
    original_builder = momentum_experiment._build_momentum_strategy

    def recording_builder(parameters):
        strategy = original_builder(parameters)
        created_strategies.append(strategy)
        return strategy

    monkeypatch.setattr(
        momentum_experiment,
        "_build_momentum_strategy",
        recording_builder,
    )

    select_momentum_parameters(
        train_data=prices.iloc[:10],
        validation_data=prices.iloc[10:16],
        lookbacks=[1, 3],
        target_weights=[0.2, 0.4],
        allow_short=True,
        transaction_cost_rate=0.001,
    )

    assert len(created_strategies) == 4
    assert len({id(strategy) for strategy in created_strategies}) == 4


def test_candidate_selection_metrics_include_transaction_costs():
    prices = synthetic_prices()
    selection_arguments = {
        "train_data": prices.iloc[:10],
        "validation_data": prices.iloc[10:16],
        "lookbacks": [1],
        "target_weights": [0.4],
        "allow_short": True,
    }

    _, without_costs = select_momentum_parameters(
        **selection_arguments,
        transaction_cost_rate=0.0,
    )
    _, with_costs = select_momentum_parameters(
        **selection_arguments,
        transaction_cost_rate=0.01,
    )

    assert with_costs.loc[0, "validation_transaction_cost"] > 0
    assert (
        with_costs.loc[0, "validation_net_return"]
        < without_costs.loc[0, "validation_net_return"]
    )


def test_run_momentum_experiment_keeps_test_data_out_of_selection():
    original = synthetic_prices()
    changed_test = original.copy()
    changed_test.iloc[16:, 0] = [80.0, 130.0, 70.0, 140.0]

    original_result = run_momentum_experiment(
        data=original,
        config=experiment_config(),
        lookbacks=[1, 3],
        target_weights=[0.2, 0.4],
        allow_short=True,
    )
    changed_result = run_momentum_experiment(
        data=changed_test,
        config=experiment_config(),
        lookbacks=[1, 3],
        target_weights=[0.2, 0.4],
        allow_short=True,
    )

    assert original_result.experiment.parameters == changed_result.experiment.parameters
    pd.testing.assert_frame_equal(
        original_result.candidate_results,
        changed_result.candidate_results,
    )
    assert not original_result.experiment.test.net_returns.equals(
        changed_result.experiment.test.net_returns
    )


def test_standardized_momentum_workflow_returns_held_out_experiment_result():
    result = run_momentum_experiment(
        data=synthetic_prices(),
        config=experiment_config(),
        lookbacks=[1, 3],
        target_weights=[0.2, 0.4],
        allow_short=False,
    )

    assert isinstance(result, MomentumExperimentResult)
    assert isinstance(result.experiment, ExperimentResult)
    assert result.experiment.test.net_returns.index.tolist() == [16, 17, 18, 19]
    assert result.experiment.parameters["allow_short"] is False
    assert np.isfinite(result.experiment.test.net_returns).all()


def test_summarize_momentum_experiment_reports_required_fields():
    result = run_momentum_experiment(
        data=synthetic_prices(),
        config=experiment_config(),
        lookbacks=[1, 3],
        target_weights=[0.2, 0.4],
        allow_short=True,
    )

    summary = summarize_momentum_experiment(result)

    assert summary["experiment_name"] == "standard-momentum"
    assert summary["selected_parameters"] == result.experiment.parameters
    assert summary["performance"].index.tolist() == ["validation", "test"]
    assert summary["performance"].columns.tolist() == [
        "gross_return",
        "net_return",
        "volatility",
        "sharpe",
        "max_drawdown",
        "turnover",
        "transaction_cost",
    ]
    assert summary["performance"].loc["validation", "net_return"] == pytest.approx(
        result.experiment.validation.net_metrics["Total Return"]
    )
    assert summary["performance"].loc["test", "transaction_cost"] == pytest.approx(
        result.experiment.test.costs.sum()
    )


@pytest.mark.parametrize(
    ("lookbacks", "target_weights", "error"),
    [
        ([], [0.2], "lookbacks"),
        ([0], [0.2], "lookback"),
        ([1], [], "target_weights"),
        ([1], [np.nan], "target_weight"),
    ],
)
def test_select_momentum_parameters_validates_candidate_grid(
    lookbacks,
    target_weights,
    error,
):
    prices = synthetic_prices()

    with pytest.raises((TypeError, ValueError), match=error):
        select_momentum_parameters(
            train_data=prices.iloc[:10],
            validation_data=prices.iloc[10:16],
            lookbacks=lookbacks,
            target_weights=target_weights,
            allow_short=True,
            transaction_cost_rate=0.001,
        )
