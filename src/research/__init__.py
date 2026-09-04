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