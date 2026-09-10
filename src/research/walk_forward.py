"""Generic expanding-window walk-forward research evaluation."""

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ..metrics import performance_summary
from .experiment import (
    EvaluationResult,
    ParameterSelector,
    Positions,
    StrategyEvaluator,
    evaluate_split,
)


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train_size: int
    validation_size: int
    test_size: int
    step_size: int | None = None

    def __post_init__(self):
        sizes = {
            "initial_train_size": self.initial_train_size,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
        }
        for name, value in sizes.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.step_size is None:
            object.__setattr__(self, "step_size", self.test_size)
        elif isinstance(self.step_size, bool) or not isinstance(self.step_size, int):
            raise TypeError("step_size must be an integer")
        elif self.step_size <= 0:
            raise ValueError("step_size must be positive")

        if self.step_size != self.test_size:
            raise ValueError(
                "step_size must equal test_size for contiguous, non-overlapping tests"
            )


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold: WalkForwardFold
    selected_parameters: Mapping[str, Any]
    test: EvaluationResult


@dataclass(frozen=True)
class WalkForwardResult:
    config: WalkForwardConfig
    folds: tuple[WalkForwardFoldResult, ...]
    gross_returns: pd.Series
    net_returns: pd.Series
    positions: Positions
    turnover: pd.Series
    costs: pd.Series
    gross_metrics: Mapping[str, float]
    net_metrics: Mapping[str, float]
    total_turnover: float
    total_transaction_cost: float
    fold_table: pd.DataFrame


def generate_walk_forward_folds(
    data: pd.DataFrame,
    config: WalkForwardConfig,
) -> tuple[WalkForwardFold, ...]:
    """Build expanding train windows with contiguous, non-overlapping tests."""
    if data.empty:
        raise ValueError("data cannot be empty")
    if not data.index.is_monotonic_increasing:
        raise ValueError("data index must be chronological")
    if not data.index.is_unique:
        raise ValueError("data index must be unique")

    required = (
        config.initial_train_size
        + config.validation_size
        + config.test_size
    )
    if required > len(data):
        raise ValueError("not enough data for the first walk-forward fold")

    folds = []
    train_end = config.initial_train_size
    fold_number = 1

    while train_end + config.validation_size + config.test_size <= len(data):
        validation_end = train_end + config.validation_size
        test_end = validation_end + config.test_size
        train = data.iloc[:train_end].copy()
        validation = data.iloc[train_end:validation_end].copy()
        test = data.iloc[validation_end:test_end].copy()

        if train.empty or validation.empty or test.empty:
            raise ValueError("walk-forward folds cannot contain empty partitions")

        folds.append(WalkForwardFold(
            fold=fold_number,
            train=train,
            validation=validation,
            test=test,
        ))
        train_end += config.step_size
        fold_number += 1

    return tuple(folds)


def _ending_positions(positions: Positions):
    ending = positions.iloc[-1]
    return ending.copy() if isinstance(ending, pd.Series) else float(ending)


def _build_fold_table(
    fold_results: tuple[WalkForwardFoldResult, ...],
) -> pd.DataFrame:
    rows = []

    for fold_result in fold_results:
        fold = fold_result.fold
        evaluation = fold_result.test
        row = {
            "fold": fold.fold,
            "train_start": fold.train.index[0],
            "train_end": fold.train.index[-1],
            "validation_start": fold.validation.index[0],
            "validation_end": fold.validation.index[-1],
            "test_start": fold.test.index[0],
            "test_end": fold.test.index[-1],
            "selected_parameters": dict(fold_result.selected_parameters),
            "test_net_return": evaluation.net_metrics["Total Return"],
            "test_sharpe": evaluation.net_metrics["Sharpe Ratio"],
            "test_max_drawdown": evaluation.net_metrics["Max Drawdown"],
            "turnover": evaluation.turnover.sum(),
            "transaction_cost": evaluation.costs.sum(),
        }
        row.update({
            f"selected_{name}": value
            for name, value in fold_result.selected_parameters.items()
        })
        rows.append(row)

    return pd.DataFrame(rows)


def run_walk_forward(
    data: pd.DataFrame,
    config: WalkForwardConfig,
    evaluator: StrategyEvaluator,
    parameter_selector: ParameterSelector,
    transaction_cost_rate: float,
) -> WalkForwardResult:
    """Select per fold, evaluate unseen tests, and stitch continuous OOS output."""
    folds = generate_walk_forward_folds(data, config)
    fold_results = []
    previous_positions = None

    for fold in folds:
        selected_parameters = dict(
            parameter_selector(fold.train.copy(), fold.validation.copy())
        )
        test_history = pd.concat([fold.train, fold.validation])
        test_result = evaluate_split(
            evaluator=evaluator,
            historical_data=test_history,
            evaluation_data=fold.test,
            parameters=selected_parameters,
            transaction_cost_rate=transaction_cost_rate,
            initial_positions=previous_positions,
        )
        fold_results.append(WalkForwardFoldResult(
            fold=fold,
            selected_parameters=selected_parameters,
            test=test_result,
        ))
        previous_positions = _ending_positions(test_result.positions)

    fold_results_used = tuple(fold_results)
    gross_returns = pd.concat([
        fold_result.test.gross_returns for fold_result in fold_results_used
    ])
    net_returns = pd.concat([
        fold_result.test.net_returns for fold_result in fold_results_used
    ])
    positions = pd.concat([
        fold_result.test.positions for fold_result in fold_results_used
    ])
    turnover = pd.concat([
        fold_result.test.turnover for fold_result in fold_results_used
    ])
    costs = pd.concat([
        fold_result.test.costs for fold_result in fold_results_used
    ])

    if not gross_returns.index.is_unique:
        raise ValueError("walk-forward test indexes must not overlap")
    if not gross_returns.index.is_monotonic_increasing:
        raise ValueError("stitched walk-forward index must be chronological")

    for values in (net_returns, positions, turnover, costs):
        if not values.index.equals(gross_returns.index):
            raise ValueError("all stitched walk-forward outputs must share one index")

    return WalkForwardResult(
        config=config,
        folds=fold_results_used,
        gross_returns=gross_returns,
        net_returns=net_returns,
        positions=positions,
        turnover=turnover,
        costs=costs,
        gross_metrics=performance_summary(gross_returns),
        net_metrics=performance_summary(net_returns),
        total_turnover=float(turnover.sum()),
        total_transaction_cost=float(costs.sum()),
        fold_table=_build_fold_table(fold_results_used),
    )
