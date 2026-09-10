"""Standardized research workflow for the existing momentum strategy."""

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .experiment import (
    EvaluationResult,
    ExperimentConfig,
    ExperimentResult,
    TargetWeightStrategyEvaluator,
    evaluate_split,
    run_experiment,
)
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)


@dataclass(frozen=True)
class MomentumExperimentResult:
    experiment: ExperimentResult
    candidate_results: pd.DataFrame


MOMENTUM_NAME = "Momentum"
EQUAL_WEIGHT_BENCHMARK_NAME = "Equal-Weight Constant Allocation"
SPY_BENCHMARK_NAME = "SPY Buy & Hold"


@dataclass(frozen=True)
class MomentumBenchmarkResult:
    validation: Mapping[str, EvaluationResult]
    test: Mapping[str, EvaluationResult]
    comparison: pd.DataFrame


@dataclass(frozen=True)
class MomentumWalkForwardResult:
    momentum: WalkForwardResult
    benchmarks: Mapping[str, WalkForwardResult]
    comparison: pd.DataFrame


@dataclass(frozen=True)
class _ConstantWeightStrategy:
    weights: Mapping[str, float]

    def generate_target_weights(self, history: pd.DataFrame) -> dict[str, float]:
        return dict(self.weights)


def _build_momentum_strategy(parameters: Mapping[str, Any]):
    # Imported lazily because src.strategy also depends on research utilities.
    from ..strategy import MomentumTradingStrategy

    return MomentumTradingStrategy(
        lookback=parameters["lookback"],
        target_weight=parameters["target_weight"],
        allow_short=parameters["allow_short"],
    )


def _validate_candidates(
    lookbacks: Sequence[int],
    target_weights: Sequence[float],
    allow_short: bool,
) -> tuple[list[int], list[float]]:
    lookbacks_used = list(lookbacks)
    target_weights_used = list(target_weights)

    if not lookbacks_used:
        raise ValueError("lookbacks must contain at least one candidate")

    if not target_weights_used:
        raise ValueError("target_weights must contain at least one candidate")

    if not isinstance(allow_short, bool):
        raise TypeError("allow_short must be a boolean")

    for lookback in lookbacks_used:
        if isinstance(lookback, bool) or not isinstance(lookback, int):
            raise TypeError("each lookback must be an integer")
        if lookback <= 0:
            raise ValueError("each lookback must be positive")

    normalized_weights = []
    for target_weight in target_weights_used:
        weight = float(target_weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("each target_weight must be finite and positive")
        normalized_weights.append(weight)

    return lookbacks_used, normalized_weights


def select_momentum_parameters(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    lookbacks: Sequence[int],
    target_weights: Sequence[float],
    allow_short: bool,
    transaction_cost_rate: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select momentum parameters using net validation Sharpe only.

    Candidates are evaluated in input order, so the first candidate is retained
    when multiple candidates share the maximum net validation Sharpe.
    """
    lookbacks_used, target_weights_used = _validate_candidates(
        lookbacks,
        target_weights,
        allow_short,
    )
    evaluator = TargetWeightStrategyEvaluator(_build_momentum_strategy)
    rows = []

    for lookback, target_weight in product(lookbacks_used, target_weights_used):
        parameters = {
            "lookback": lookback,
            "target_weight": target_weight,
            "allow_short": allow_short,
        }
        validation_result = evaluate_split(
            evaluator=evaluator,
            historical_data=train_data,
            evaluation_data=validation_data,
            parameters=parameters,
            transaction_cost_rate=transaction_cost_rate,
        )
        net_metrics = validation_result.net_metrics

        rows.append({
            "lookback": lookback,
            "target_weight": target_weight,
            "validation_net_return": net_metrics["Total Return"],
            "validation_net_volatility": net_metrics["Annualized Volatility"],
            "validation_net_sharpe": net_metrics["Sharpe Ratio"],
            "validation_max_drawdown": net_metrics["Max Drawdown"],
            "validation_turnover": validation_result.turnover.sum(),
            "validation_transaction_cost": validation_result.costs.sum(),
        })

    candidate_results = pd.DataFrame(rows)
    sharpe_values = candidate_results["validation_net_sharpe"]

    if not sharpe_values.notna().any():
        raise ValueError("all candidate validation Sharpe ratios are undefined")

    best = candidate_results.loc[sharpe_values.idxmax()]
    selected_parameters = {
        "lookback": int(best["lookback"]),
        "target_weight": float(best["target_weight"]),
        "allow_short": allow_short,
    }

    return selected_parameters, candidate_results


def run_momentum_experiment(
    data: pd.DataFrame,
    config: ExperimentConfig,
    lookbacks: Sequence[int],
    target_weights: Sequence[float],
    allow_short: bool = True,
) -> MomentumExperimentResult:
    """Select momentum parameters, then run the standard held-out experiment."""
    selection = {}

    def parameter_selector(train_data, validation_data):
        parameters, candidate_results = select_momentum_parameters(
            train_data=train_data,
            validation_data=validation_data,
            lookbacks=lookbacks,
            target_weights=target_weights,
            allow_short=allow_short,
            transaction_cost_rate=config.transaction_cost_rate,
        )
        selection["candidate_results"] = candidate_results
        return parameters

    experiment = run_experiment(
        data=data,
        config=config,
        evaluator=TargetWeightStrategyEvaluator(_build_momentum_strategy),
        parameter_selector=parameter_selector,
    )

    return MomentumExperimentResult(
        experiment=experiment,
        candidate_results=selection["candidate_results"].copy(),
    )


def _summary_row(result: EvaluationResult) -> dict[str, float]:
    return {
        "gross_return": result.gross_metrics["Total Return"],
        "net_return": result.net_metrics["Total Return"],
        "volatility": result.net_metrics["Annualized Volatility"],
        "sharpe": result.net_metrics["Sharpe Ratio"],
        "max_drawdown": result.net_metrics["Max Drawdown"],
        "turnover": result.turnover.sum(),
        "transaction_cost": result.costs.sum(),
    }


def _evaluate_constant_weights(
    historical_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    weights: Mapping[str, float],
    transaction_cost_rate: float,
) -> EvaluationResult:
    evaluator = TargetWeightStrategyEvaluator(
        lambda parameters: _ConstantWeightStrategy(parameters["weights"])
    )
    return evaluate_split(
        evaluator=evaluator,
        historical_data=historical_data,
        evaluation_data=evaluation_data,
        parameters={"weights": dict(weights)},
        transaction_cost_rate=transaction_cost_rate,
    )


def compare_momentum_with_benchmarks(
    result: MomentumExperimentResult,
    spy_symbol: str = "SPY",
) -> MomentumBenchmarkResult:
    """Evaluate passive benchmarks on the Momentum experiment's exact splits."""
    experiment = result.experiment
    splits = experiment.splits
    symbols = list(splits.train.columns)

    if spy_symbol not in symbols:
        raise ValueError(f"benchmark symbol {spy_symbol} is not in the experiment data")

    equal_weight = 1.0 / len(symbols)
    equal_weights = {symbol: equal_weight for symbol in symbols}
    spy_weights = {
        symbol: 1.0 if symbol == spy_symbol else 0.0
        for symbol in symbols
    }
    test_history = pd.concat([splits.train, splits.validation])
    cost_rate = experiment.config.transaction_cost_rate

    validation_results = {
        MOMENTUM_NAME: experiment.validation,
        EQUAL_WEIGHT_BENCHMARK_NAME: _evaluate_constant_weights(
            historical_data=splits.train,
            evaluation_data=splits.validation,
            weights=equal_weights,
            transaction_cost_rate=cost_rate,
        ),
        SPY_BENCHMARK_NAME: _evaluate_constant_weights(
            historical_data=splits.train,
            evaluation_data=splits.validation,
            weights=spy_weights,
            transaction_cost_rate=cost_rate,
        ),
    }
    test_results = {
        MOMENTUM_NAME: experiment.test,
        EQUAL_WEIGHT_BENCHMARK_NAME: _evaluate_constant_weights(
            historical_data=test_history,
            evaluation_data=splits.test,
            weights=equal_weights,
            transaction_cost_rate=cost_rate,
        ),
        SPY_BENCHMARK_NAME: _evaluate_constant_weights(
            historical_data=test_history,
            evaluation_data=splits.test,
            weights=spy_weights,
            transaction_cost_rate=cost_rate,
        ),
    }

    rows = {
        (split_name, strategy_name): _summary_row(evaluation)
        for split_name, evaluations in (
            ("validation", validation_results),
            ("test", test_results),
        )
        for strategy_name, evaluation in evaluations.items()
    }
    comparison = pd.DataFrame.from_dict(rows, orient="index")
    comparison.index = pd.MultiIndex.from_tuples(
        comparison.index,
        names=["split", "strategy"],
    )

    return MomentumBenchmarkResult(
        validation=validation_results,
        test=test_results,
        comparison=comparison,
    )


def _walk_forward_summary_row(result: WalkForwardResult) -> dict[str, float]:
    return {
        "gross_return": result.gross_metrics["Total Return"],
        "net_return": result.net_metrics["Total Return"],
        "volatility": result.net_metrics["Annualized Volatility"],
        "sharpe": result.net_metrics["Sharpe Ratio"],
        "max_drawdown": result.net_metrics["Max Drawdown"],
        "turnover": result.total_turnover,
        "transaction_cost": result.total_transaction_cost,
    }


def run_momentum_walk_forward(
    data: pd.DataFrame,
    config: WalkForwardConfig,
    transaction_cost_rate: float,
    lookbacks: Sequence[int] = (5, 10, 20, 60),
    target_weight: float = 0.25,
    allow_short: bool = False,
    spy_symbol: str = "SPY",
) -> MomentumWalkForwardResult:
    """Run expanding Momentum selection and same-date passive benchmarks."""
    symbols = list(data.columns)
    if spy_symbol not in symbols:
        raise ValueError(f"benchmark symbol {spy_symbol} is not in the data")

    def momentum_selector(train_data, validation_data):
        selected, _ = select_momentum_parameters(
            train_data=train_data,
            validation_data=validation_data,
            lookbacks=lookbacks,
            target_weights=[target_weight],
            allow_short=allow_short,
            transaction_cost_rate=transaction_cost_rate,
        )
        return selected

    momentum = run_walk_forward(
        data=data,
        config=config,
        evaluator=TargetWeightStrategyEvaluator(_build_momentum_strategy),
        parameter_selector=momentum_selector,
        transaction_cost_rate=transaction_cost_rate,
    )

    equal_weight = 1.0 / len(symbols)
    benchmark_weights = {
        EQUAL_WEIGHT_BENCHMARK_NAME: {
            symbol: equal_weight for symbol in symbols
        },
        SPY_BENCHMARK_NAME: {
            symbol: 1.0 if symbol == spy_symbol else 0.0
            for symbol in symbols
        },
    }
    benchmark_evaluator = TargetWeightStrategyEvaluator(
        lambda parameters: _ConstantWeightStrategy(parameters["weights"])
    )
    benchmarks = {}

    for name, weights in benchmark_weights.items():
        benchmarks[name] = run_walk_forward(
            data=data,
            config=config,
            evaluator=benchmark_evaluator,
            parameter_selector=lambda train, validation, weights=weights: {
                "weights": weights
            },
            transaction_cost_rate=transaction_cost_rate,
        )

        if not benchmarks[name].net_returns.index.equals(momentum.net_returns.index):
            raise ValueError("benchmark and Momentum OOS indexes must match")

    comparison_rows = {
        MOMENTUM_NAME: _walk_forward_summary_row(momentum),
        **{
            name: _walk_forward_summary_row(benchmark)
            for name, benchmark in benchmarks.items()
        },
    }

    return MomentumWalkForwardResult(
        momentum=momentum,
        benchmarks=benchmarks,
        comparison=pd.DataFrame.from_dict(comparison_rows, orient="index"),
    )


def summarize_momentum_experiment(
    result: MomentumExperimentResult,
) -> dict[str, Any]:
    """Return a compact validation/test report for a momentum experiment."""
    experiment = result.experiment
    performance = pd.DataFrame.from_dict(
        {
            "validation": _summary_row(experiment.validation),
            "test": _summary_row(experiment.test),
        },
        orient="index",
    )
    performance.index.name = "split"

    return {
        "experiment_name": experiment.config.name,
        "selected_parameters": dict(experiment.parameters),
        "performance": performance,
    }
