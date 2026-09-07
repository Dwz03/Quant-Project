# Research Report v1

## 1. Research Question

The objective of this research is to evaluate whether simple statistical
trading strategies can generate robust out-of-sample performance after
accounting for parameter selection, transaction costs, market regimes,
and common backtesting biases.

The strategies considered include:

- Mean reversion
- Pairs trading
- PCA statistical arbitrage


## 2. Validation Framework

The dataset is divided chronologically into:

- Training data
- Validation data
- Test data

Training data is used for model estimation and historical context.

Validation data is used for strategy comparison and hyperparameter tuning.

The test set remains untouched during parameter selection and is used only
for final out-of-sample evaluation.


## 3. Hyperparameter Selection

For the mean-reversion strategy, parameters such as:

- rolling window
- signal threshold

are evaluated using a parameter grid on validation data.

The parameter combination with the best validation performance is selected
before evaluating the final test set.


## 4. Parameter Robustness

The best parameter combination is not evaluated in isolation.

Performance around neighbouring parameter combinations is also examined.

A strategy is considered more robust when:

- neighbouring parameters produce similar performance
- performance does not collapse away from the optimum
- the parameter surface is relatively smooth


## 5. Bias Audit

The research process explicitly considers:

### Look-ahead bias
Models and signals use only information available at the time of prediction.

### Data leakage
Validation and test observations are not included in model fitting.

### Data snooping
The final test set is not repeatedly used for parameter selection.

### Survivorship bias
This is not yet fully controlled because the current asset universe may
contain securities selected using information available today.


## 6. Transaction Cost Sensitivity

Strategy performance is evaluated under multiple transaction-cost assumptions.

Costs are applied according to portfolio turnover.

The analysis measures how:

- total return
- Sharpe ratio
- maximum drawdown

change as transaction costs increase.

A grid-based break-even transaction cost is also estimated.


## 7. Regime Analysis

Market periods are classified into:

- calm
- volatile

using historical rolling volatility.

Strategy performance is then evaluated separately within each regime.

This helps determine whether performance depends heavily on a particular
market environment.


## 8. Performance Attribution

Strategy performance is decomposed by:

- position: long / short / flat
- time: monthly contribution
- asset: contribution from individual securities

This analysis helps identify where strategy profits and losses originate.


## 9. Main Findings

To be completed after running the research pipeline on real market data.

Key questions include:

- Does the strategy remain profitable out of sample?
- Is performance stable around the selected parameters?
- Does the strategy survive realistic transaction costs?
- Is performance concentrated in one market regime?
- Are profits dominated by a small number of assets or periods?


## 10. Limitations

Current limitations include:

- Survivorship bias is not fully controlled.
- Transaction-cost models remain simplified.
- Market impact and liquidity constraints are not fully modelled.
- Hyperparameter grids are relatively small.
- Historical regime behaviour may not represent future market conditions.
- Strong backtest performance does not guarantee live performance.


## 11. Why the Strategy May Fail Live

A strategy that performs well historically may fail in live trading because:

- market regimes change
- alpha decays after discovery
- execution prices differ from backtest assumptions
- transaction costs increase
- liquidity deteriorates
- parameter relationships are unstable
- historical patterns may be statistical noise


## 12. Next Steps

The next stage is to run the complete research pipeline on real historical
market data and compare the candidate strategies using the same validation
framework.