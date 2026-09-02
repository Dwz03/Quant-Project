Strategy:
Z-score Mean Reversion

Hypothesis:
Prices that deviate significantly from their recent mean
tend to revert toward the mean over a short horizon.

Data Split:
Train: 50%
Validation: 30%
Test: 20%

Rules:
- Parameters can be developed using train data.
- Parameters can be selected using validation data.
- Test data cannot be used for parameter tuning.
- Once test results are observed, that test set is considered used.

Transaction Costs:
To be included in final strategy evaluation.

Primary Metrics:
Sharpe Ratio
Total Return
Max Drawdown
Turnover