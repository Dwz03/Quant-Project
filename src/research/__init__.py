# src/research/__init__.py

from .common import (
    split_data,
    calculate_zscore,
    calculate_zscore_with_history,
    calculate_equity_curve,
    normalize_positions,
)

from .mean_reversion import (
    run_mean_reversion,
    run_mean_reversion_with_history,
)

from .pairs import (
    run_pairs_trading,
    run_pairs_trading_with_history,
    screen_pairs,
    select_pairs,
    estimate_hedge_ratio,
    calculate_spread,
    check_spread_stationarity,
    check_cointegration
)

from .pca import (
    fit_pca,
    run_pca_stat_arb,
    calculate_half_life,
)

from .validation import run_validation_comparison

from .experiment import (
    DataSplits,
    EvaluationResult,
    ExperimentConfig,
    ExperimentResult,
    ParameterSelector,
    StrategyEvaluation,
    StrategyEvaluator,
    TargetWeightGenerator,
    TargetWeightStrategyEvaluator,
    create_data_splits,
    evaluate_split,
    evaluate_target_weight_strategy,
    run_experiment,
)

from .momentum_experiment import (
    EQUAL_WEIGHT_BENCHMARK_NAME,
    MOMENTUM_NAME,
    SPY_BENCHMARK_NAME,
    MomentumBenchmarkResult,
    MomentumExperimentResult,
    MomentumWalkForwardResult,
    compare_momentum_with_benchmarks,
    run_momentum_experiment,
    run_momentum_walk_forward,
    select_momentum_parameters,
    summarize_momentum_experiment,
)

from .walk_forward import (
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardFoldResult,
    WalkForwardResult,
    generate_walk_forward_folds,
    run_walk_forward,
)
