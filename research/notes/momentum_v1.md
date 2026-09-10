# Momentum v1 Research Note

## Hypothesis

A simple long-only time-series momentum filter may improve risk-adjusted performance relative to passive equity exposure.

## Experiment design

| Item | Value |
| --- | --- |
| Universe | SPY, QQQ, AAPL, MSFT |
| Data | Daily adjusted close prices |
| Aligned period | 2021-01-04 to 2025-12-31 |
| Candidate lookbacks | 5, 10, 20, 60 trading days |
| Target weight | 0.25 |
| Shorting | Disabled (`allow_short=False`) |
| Transaction cost | 0.001 per unit turnover |

The fixed-split parameter selection used train and validation data only. The held-out test period did not influence selection.

## Fixed-split result

The selected lookback was 60 trading days.

| Held-out test metric | Result |
| --- | ---: |
| Gross return | 0.161687 |
| Net return | 0.153292 |
| Net volatility | 0.107400 |
| Net Sharpe | 1.387153 |
| Maximum drawdown | -0.058789 |
| Turnover | 7.250000 |
| Transaction cost | 0.007250 |

The fixed-split result produced lower return than SPY alongside lower volatility and drawdown. It is not evidence of robust alpha by itself.

## Walk-forward robustness evaluation

The expanding-window evaluation used 252 training observations, 63 validation observations, 63 unseen test observations, and a 63-observation step. Target weight remained 0.25 and shorting remained disabled.

Across the 14 folds, the selected lookbacks were:

```text
5, 5, 20, 20, 10, 60, 10, 5, 60, 20, 5, 60, 60, 60
```

### Stitched out-of-sample results

| Strategy | Gross return | Net return | Net volatility | Net Sharpe | Maximum drawdown | Turnover | Transaction cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Momentum v1 | 0.354233 | 0.230336 | 0.133234 | 0.511217 | -0.139037 | 96.000000 | 0.096000 |
| Equal-Weight Constant Allocation | 0.666782 | 0.665144 | 0.221110 | 0.769074 | -0.256648 | 1.000000 | 0.001000 |
| SPY Buy & Hold | 0.561643 | 0.560095 | 0.180345 | 0.794578 | -0.212848 | 1.000000 | 0.001000 |

## Interpretation

- Momentum v1 did not outperform either passive benchmark on walk-forward risk-adjusted performance.
- The fixed-split Sharpe of approximately 1.39 did not remain robust across the broader stitched walk-forward out-of-sample periods.
- Momentum v1 reduced volatility and maximum drawdown relative to the passive benchmarks.
- The result is more promising as a trend or risk overlay than as standalone alpha.
- Performance was regime-dependent.
- Selected lookbacks varied materially across folds, indicating parameter instability.
- High turnover and transaction costs materially reduced gross performance.

## Decision

The strong standalone-alpha hypothesis is rejected for Momentum v1. The version must not be tuned using the observed out-of-sample evidence.

Momentum v1 is preserved as a negative-result baseline, a reference implementation for the standardized research workflow, and a possible risk-overlay candidate.

Future work must be framed as new hypotheses and new versions rather than revisions to Momentum v1. Candidate directions include volatility-targeted momentum, lower-turnover signals, a broader universe, cross-sectional momentum, and use as a portfolio risk overlay.
