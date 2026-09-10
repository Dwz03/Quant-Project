"""Composable train/validation/test research experiment primitives."""

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import pandas as pd

from ..metrics import performance_summary
from .common import split_data
from .validation import apply_transaction_costs


Positions = pd.Series | pd.DataFrame
Parameters = Mapping[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    train_ratio: float
    validation_ratio: float
    transaction_cost_rate: float = 0.0


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class StrategyEvaluation:
    returns: pd.Series
    positions: Positions
    artifacts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    positions: Positions
    turnover: pd.Series
    costs: pd.Series
    gross_metrics: Mapping[str, float]
    net_metrics: Mapping[str, float]
    artifacts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    splits: DataSplits
    parameters: Mapping[str, Any]
    validation: EvaluationResult
    test: EvaluationResult


class StrategyEvaluator(Protocol):
    def __call__(
        self,
        historical_data: pd.DataFrame,
        evaluation_data: pd.DataFrame,
        parameters: Parameters,
    ) -> StrategyEvaluation:
        ...


class TargetWeightGenerator(Protocol):
    def generate_target_weights(self, history: pd.DataFrame) -> Mapping[str, float]:
        ...


ParameterSelector = Callable[[pd.DataFrame, pd.DataFrame], Parameters]


def create_data_splits(data: pd.DataFrame, config: ExperimentConfig) -> DataSplits:
    """Create non-empty chronological splits through the existing split utility."""
    train, validation, test = split_data(
        data,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
    )

    if train.empty or validation.empty or test.empty:
        raise ValueError("train, validation, and test splits must all contain data")

    return DataSplits(
        train=train.copy(),
        validation=validation.copy(),
        test=test.copy(),
    )


def evaluate_split(
    evaluator: StrategyEvaluator,
    historical_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    parameters: Parameters,
    transaction_cost_rate: float,
    initial_positions: Any | None = None,
) -> EvaluationResult:
    """Evaluate one out-of-sample split and attach costs and existing metrics."""
    evaluation = evaluator(
        historical_data.copy(),
        evaluation_data.copy(),
        dict(parameters),
    )

    if not isinstance(evaluation, StrategyEvaluation):
        raise TypeError("evaluator must return StrategyEvaluation")

    if evaluation.returns.empty:
        raise ValueError("strategy evaluation returns cannot be empty")

    if not evaluation.returns.index.equals(evaluation_data.index):
        raise ValueError("strategy returns must match the evaluation data index")

    if not evaluation.returns.index.equals(evaluation.positions.index):
        raise ValueError("strategy returns and positions must have identical indexes")

    cost_result = apply_transaction_costs(
        evaluation.returns,
        evaluation.positions,
        transaction_cost_rate,
        initial_positions=initial_positions,
    )

    gross_returns = cost_result["strategy_return"].copy()
    net_returns = cost_result["net_strategy_return"].copy()

    return EvaluationResult(
        gross_returns=gross_returns,
        net_returns=net_returns,
        positions=evaluation.positions.copy(),
        turnover=cost_result["turnover"].copy(),
        costs=cost_result["cost"].copy(),
        gross_metrics=performance_summary(gross_returns),
        net_metrics=performance_summary(net_returns),
        artifacts=dict(evaluation.artifacts),
    )


def run_experiment(
    data: pd.DataFrame,
    config: ExperimentConfig,
    evaluator: StrategyEvaluator,
    parameter_selector: ParameterSelector | None = None,
) -> ExperimentResult:
    """Select parameters without test data, then evaluate validation and test.

    The parameter selector may inspect both the training and validation splits.
    Consequently, validation metrics describe model-selection performance and
    are not an unbiased final out-of-sample estimate. The test split is never
    passed to parameter selection and its metrics are the final held-out
    evaluation.
    """
    splits = create_data_splits(data, config)

    if parameter_selector is None:
        parameters: Parameters = {}
    else:
        parameters = dict(
            parameter_selector(splits.train.copy(), splits.validation.copy())
        )

    validation_result = evaluate_split(
        evaluator=evaluator,
        historical_data=splits.train,
        evaluation_data=splits.validation,
        parameters=parameters,
        transaction_cost_rate=config.transaction_cost_rate,
    )

    test_history = pd.concat([splits.train, splits.validation])
    test_result = evaluate_split(
        evaluator=evaluator,
        historical_data=test_history,
        evaluation_data=splits.test,
        parameters=parameters,
        transaction_cost_rate=config.transaction_cost_rate,
    )

    return ExperimentResult(
        config=config,
        splits=splits,
        parameters=dict(parameters),
        validation=validation_result,
        test=test_result,
    )


def evaluate_target_weight_strategy(
    strategy: TargetWeightGenerator,
    historical_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
) -> StrategyEvaluation:
    """Evaluate close-to-close returns using weights decided one bar earlier.

    Every asset return in the supplied evaluation universe must be finite,
    regardless of its target weight. Missing or infinite returns fail the
    evaluation instead of being converted to zero contribution by aggregation.
    """
    if historical_data.empty:
        raise ValueError("historical_data cannot be empty")

    if evaluation_data.empty:
        raise ValueError("evaluation_data cannot be empty")

    if not historical_data.columns.equals(evaluation_data.columns):
        raise ValueError("historical and evaluation data must have identical columns")

    symbols = list(evaluation_data.columns)
    combined = pd.concat([historical_data, evaluation_data])
    evaluation_returns = combined.pct_change(fill_method=None).iloc[
        len(historical_data):
    ]

    if not np.isfinite(evaluation_returns.to_numpy(dtype=float)).all():
        raise ValueError(
            "evaluation asset returns must be finite; "
            "check for missing prices or zero prior prices"
        )

    position_rows = []

    for offset in range(len(evaluation_data)):
        decision_history = combined.iloc[:len(historical_data) + offset]
        target_weights = strategy.generate_target_weights(decision_history.copy())

        unknown_symbols = set(target_weights) - set(symbols)
        if unknown_symbols:
            raise ValueError("strategy returned weights for unknown symbols")

        row = {symbol: float(target_weights.get(symbol, 0.0)) for symbol in symbols}
        if not all(np.isfinite(weight) for weight in row.values()):
            raise ValueError("strategy target weights must be finite")

        position_rows.append(row)

    positions = pd.DataFrame(position_rows, index=evaluation_data.index, columns=symbols)
    strategy_returns = (positions * evaluation_returns).sum(axis=1, skipna=False)
    strategy_returns.name = "strategy_return"

    return StrategyEvaluation(
        returns=strategy_returns,
        positions=positions,
        artifacts={"asset_returns": evaluation_returns},
    )


@dataclass(frozen=True)
class TargetWeightStrategyEvaluator:
    """Build a fresh target-weight strategy for each evaluated data split."""

    strategy_factory: Callable[[Parameters], TargetWeightGenerator]

    def __call__(
        self,
        historical_data: pd.DataFrame,
        evaluation_data: pd.DataFrame,
        parameters: Parameters,
    ) -> StrategyEvaluation:
        strategy = self.strategy_factory(parameters)
        return evaluate_target_weight_strategy(
            strategy,
            historical_data,
            evaluation_data,
        )
