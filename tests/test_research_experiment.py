import numpy as np
import pandas as pd
import pytest

from src.metrics import performance_summary
from src.research.experiment import (
    ExperimentConfig,
    ExperimentResult,
    StrategyEvaluation,
    TargetWeightStrategyEvaluator,
    create_data_splits,
    evaluate_split,
    evaluate_target_weight_strategy,
    run_experiment,
)
from src.research.validation import apply_transaction_costs
from src.strategy import MomentumTradingStrategy


def test_create_data_splits_reuses_chronological_split():
    data = pd.DataFrame({"price": range(10)})
    config = ExperimentConfig(
        name="split-test",
        train_ratio=0.5,
        validation_ratio=0.3,
    )

    splits = create_data_splits(data, config)

    assert splits.train.index.tolist() == [0, 1, 2, 3, 4]
    assert splits.validation.index.tolist() == [5, 6, 7]
    assert splits.test.index.tolist() == [8, 9]


def test_create_data_splits_rejects_empty_partition():
    data = pd.DataFrame({"price": [1, 2]})
    config = ExperimentConfig(
        name="too-small",
        train_ratio=0.5,
        validation_ratio=0.25,
    )

    with pytest.raises(ValueError, match="must all contain data"):
        create_data_splits(data, config)


def test_apply_transaction_costs_aggregates_multi_asset_turnover():
    returns = pd.Series([0.00, 0.02, -0.01])
    positions = pd.DataFrame({
        "A": [0.0, 0.5, -0.5],
        "B": [0.0, -0.5, 0.5],
    })

    result = apply_transaction_costs(returns, positions, cost_rate=0.001)

    assert result["turnover"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert result["cost"].tolist() == pytest.approx([0.0, 0.001, 0.002])
    assert result["net_strategy_return"].tolist() == pytest.approx([
        0.00,
        0.019,
        -0.012,
    ])


def test_evaluate_split_adds_costs_metrics_and_artifacts():
    index = pd.Index([2, 3])

    def evaluator(historical_data, evaluation_data, parameters):
        return StrategyEvaluation(
            returns=pd.Series([0.02, -0.01], index=index),
            positions=pd.Series([1.0, -1.0], index=index),
            artifacts={"model": parameters["model"]},
        )

    result = evaluate_split(
        evaluator=evaluator,
        historical_data=pd.DataFrame({"price": [1, 2]}),
        evaluation_data=pd.DataFrame({"price": [3, 4]}, index=index),
        parameters={"model": "demo"},
        transaction_cost_rate=0.001,
    )

    assert result.turnover.tolist() == pytest.approx([1.0, 2.0])
    assert result.net_returns.tolist() == pytest.approx([0.019, -0.012])
    assert result.gross_metrics == performance_summary(result.gross_returns)
    assert result.net_metrics == performance_summary(result.net_returns)
    assert result.artifacts == {"model": "demo"}


def test_evaluate_split_rejects_results_outside_evaluation_dates():
    def evaluator(historical_data, evaluation_data, parameters):
        wrong_index = pd.Index([1, 2])
        return StrategyEvaluation(
            returns=pd.Series([0.01, 0.02], index=wrong_index),
            positions=pd.Series([1.0, 1.0], index=wrong_index),
        )

    with pytest.raises(ValueError, match="evaluation data index"):
        evaluate_split(
            evaluator=evaluator,
            historical_data=pd.DataFrame({"price": [1, 2]}, index=[0, 1]),
            evaluation_data=pd.DataFrame({"price": [3, 4]}, index=[2, 3]),
            parameters={},
            transaction_cost_rate=0.0,
        )


def test_run_experiment_keeps_test_data_out_of_selection_and_validation():
    data = pd.DataFrame({"price": range(10)})
    calls = []

    def select_parameters(train, validation):
        assert train.index.tolist() == [0, 1, 2, 3, 4]
        assert validation.index.tolist() == [5, 6, 7]
        return {"weight": 0.5}

    def evaluator(historical_data, evaluation_data, parameters):
        calls.append((historical_data.index.tolist(), evaluation_data.index.tolist()))
        return StrategyEvaluation(
            returns=pd.Series(0.01, index=evaluation_data.index),
            positions=pd.Series(parameters["weight"], index=evaluation_data.index),
        )

    result = run_experiment(
        data=data,
        config=ExperimentConfig(
            name="research-v2",
            train_ratio=0.5,
            validation_ratio=0.3,
            transaction_cost_rate=0.001,
        ),
        evaluator=evaluator,
        parameter_selector=select_parameters,
    )

    assert isinstance(result, ExperimentResult)
    assert result.parameters == {"weight": 0.5}
    assert calls == [
        ([0, 1, 2, 3, 4], [5, 6, 7]),
        ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]),
    ]
    assert result.validation.net_returns.index.tolist() == [5, 6, 7]
    assert result.test.net_returns.index.tolist() == [8, 9]


def test_target_weight_strategy_evaluation_uses_only_prior_prices():
    class RecordingStrategy:
        def __init__(self):
            self.last_seen = []

        def generate_target_weights(self, history):
            self.last_seen.append(history.index[-1])
            weight = 1.0 if history["A"].iloc[-1] < 105 else 0.0
            return {"A": weight}

    strategy = RecordingStrategy()
    historical = pd.DataFrame({"A": [90.0, 100.0]}, index=[0, 1])
    evaluation = pd.DataFrame({"A": [110.0, 121.0]}, index=[2, 3])

    result = evaluate_target_weight_strategy(strategy, historical, evaluation)

    assert strategy.last_seen == [1, 2]
    assert result.positions["A"].tolist() == [1.0, 0.0]
    assert result.returns.tolist() == pytest.approx([0.10, 0.0])
    assert result.artifacts["asset_returns"]["A"].tolist() == pytest.approx([
        0.10,
        0.10,
    ])


@pytest.mark.parametrize(
    ("historical_prices", "evaluation_prices"),
    [
        ([90.0, 100.0], [np.nan, 110.0]),
        ([90.0, 0.0], [100.0, 110.0]),
        ([90.0, 100.0], [np.inf, 110.0]),
    ],
)
def test_target_weight_strategy_rejects_non_finite_asset_returns(
    historical_prices,
    evaluation_prices,
):
    class FixedWeightStrategy:
        def generate_target_weights(self, history):
            return {"A": 1.0}

    historical = pd.DataFrame({"A": historical_prices}, index=[0, 1])
    evaluation = pd.DataFrame({"A": evaluation_prices}, index=[2, 3])

    with pytest.raises(ValueError, match="evaluation asset returns must be finite"):
        evaluate_target_weight_strategy(
            FixedWeightStrategy(),
            historical,
            evaluation,
        )


def test_target_weight_evaluator_builds_fresh_strategy_per_split():
    built_with = []

    class FixedWeightStrategy:
        def __init__(self, weight):
            self.weight = weight

        def generate_target_weights(self, history):
            return {"A": self.weight}

    def strategy_factory(parameters):
        built_with.append(dict(parameters))
        return FixedWeightStrategy(parameters["weight"])

    evaluator = TargetWeightStrategyEvaluator(strategy_factory)
    parameters = {"weight": 0.25}
    historical = pd.DataFrame({"A": [100.0, 101.0]}, index=[0, 1])
    evaluation = pd.DataFrame({"A": [102.0]}, index=[2])

    first = evaluator(historical, evaluation, parameters)
    second = evaluator(historical, evaluation, parameters)

    assert built_with == [parameters, parameters]
    assert first.positions["A"].tolist() == [0.25]
    assert second.positions["A"].tolist() == [0.25]


def test_existing_trading_strategy_runs_through_experiment_layer():
    prices = pd.DataFrame({"A": np.arange(100.0, 112.0)})
    evaluator = TargetWeightStrategyEvaluator(
        lambda parameters: MomentumTradingStrategy(
            lookback=2,
            target_weight=0.4,
            allow_short=False,
        )
    )

    result = run_experiment(
        data=prices,
        config=ExperimentConfig(
            name="momentum-integration",
            train_ratio=0.5,
            validation_ratio=0.25,
            transaction_cost_rate=0.001,
        ),
        evaluator=evaluator,
    )

    assert isinstance(result, ExperimentResult)
    assert result.validation.positions["A"].tolist() == [0.4, 0.4, 0.4]
    assert result.test.positions["A"].tolist() == [0.4, 0.4, 0.4]
    assert result.validation.gross_returns.gt(0).all()
    assert result.test.gross_returns.gt(0).all()
    assert result.validation.costs.tolist() == pytest.approx([0.0004, 0.0, 0.0])
    assert result.test.costs.tolist() == pytest.approx([0.0004, 0.0, 0.0])
