# Week 7 Day 1 — Walk-Forward Validation Report

## 1. Objective

The objective of Day 1 was to implement and validate a walk-forward evaluation framework for the strategies developed in Week 6.

The main goals were:

1. Verify that the walk-forward implementation preserves time ordering and does not introduce look-ahead bias.
2. Check whether the simple mean-reversion strategy behaves consistently under continuous and walk-forward evaluation.
3. Evaluate the PCA statistical arbitrage strategy under repeated out-of-sample refitting.
4. Investigate why fixed-PCA and refitted-PCA strategies produce substantially different performance.

No parameter tuning was performed during Day 1.

Baseline parameters were kept fixed:

- Mean-reversion window: `20`
- Signal threshold: `2.0`
- PCA components: `2`
- Initial training period: `252` trading days
- Walk-forward test block: `21` trading days

---

## 2. Walk-Forward Framework

An expanding-window walk-forward scheme was implemented.

For each fold:

1. All observations available before the test period were used as training data.
2. The following 21 trading days formed the out-of-sample test period.
3. The training window expanded after every test block.
4. Only out-of-sample observations were retained in the final performance series.

Example:

```text
Fold 1:
Train: Day 1 ... Day 252
Test:  Day 253 ... Day 273

Fold 2:
Train: Day 1 ... Day 273
Test:  Day 274 ... Day 294

Fold 3:
Train: Day 1 ... Day 294
Test:  Day 295 ... Day 315
```

The dataset contained:

- Total observations: `1255`
- Walk-forward folds: `47`
- Out-of-sample observations: `987`

Only complete 21-day test blocks were included.

---

## 3. Mean-Reversion Sanity Check

The mean-reversion strategy does not require estimation of a fitted statistical model when the rolling window and signal threshold are fixed.

Therefore, continuous evaluation and walk-forward evaluation should produce identical strategy returns over identical dates, provided that historical information is handled correctly.

The comparison produced:

```text
Maximum return difference: 0.0
```

This confirms that the walk-forward implementation did not introduce artificial discontinuities at fold boundaries.

### Walk-Forward Mean-Reversion Performance

```text
Total Return:            -15.97%
Annualized Volatility:    11.79%
Sharpe Ratio:             -0.32
Max Drawdown:             -23.64%
```

The performance was considerably weaker than the shorter Week 6 validation result.

However, this does not mean that walk-forward validation itself reduced strategy performance.

The Week 6 result was measured over a shorter validation period, whereas the walk-forward experiment evaluated approximately four years of out-of-sample observations.

Therefore, the result suggests that the current mean-reversion strategy is not robust across the longer sample.

---

## 4. PCA Walk-Forward Evaluation

Unlike the simple mean-reversion strategy, PCA requires model estimation.

For every walk-forward fold:

1. Training prices were converted into returns.
2. PCA was fitted using training returns only.
3. The fitted PCA model was applied to the following test period.
4. PCA residuals were calculated.
5. Historical residuals were used to calculate test-period residual z-scores.
6. Trading signals and normalized portfolio positions were generated.
7. Only out-of-sample strategy returns were retained.

Therefore, the PCA model was re-estimated at the beginning of every 21-day test block.

---

## 5. Fixed PCA vs Refitted PCA

To isolate the impact of repeated PCA refitting, a direct comparison was performed.

Both strategies were evaluated over exactly the same 987 out-of-sample observations.

### Fixed PCA

The PCA model was fitted once using the initial 252-day training sample and was then kept unchanged.

```text
Total Return:             +14.77%
Annualized Volatility:     11.58%
Sharpe Ratio:               0.36
Max Drawdown:             -16.19%
```

### Refitted PCA

The PCA model was re-estimated before every 21-day walk-forward test block.

```text
Total Return:              -1.85%
Annualized Volatility:     12.18%
Sharpe Ratio:               0.02
Max Drawdown:             -32.53%
```

Under the current strategy design, repeated PCA refitting produced substantially weaker performance than keeping the initial PCA model fixed.

This result does not imply that PCA should never be refitted.

It only shows that, under the current universe, parameters, expanding-window design, and refitting frequency, monthly refitting did not improve performance.

---

## 6. PCA Loading Stability

A possible explanation for the performance difference was that PCA components changed substantially after every refit.

To investigate this, absolute cosine similarity between consecutive PCA loading vectors was calculated.

Absolute similarity was used because the sign of a PCA component is arbitrary.

### Adjacent-Fold Similarity

```text
PC1 mean similarity:    0.999919
PC1 minimum similarity: 0.999215

PC2 mean similarity:    0.999149
PC2 minimum similarity: 0.982504
```

The PCA loading vectors were therefore extremely stable between adjacent folds.

There is no evidence of large month-to-month instability in the PCA factors.

---

## 7. PCA Drift Relative to the Initial Model

Although adjacent PCA models were highly similar, small changes could accumulate over time.

Each refitted PCA model was therefore compared with the PCA model from the first walk-forward fold.

### Similarity to Initial PCA

```text
PC1 mean similarity:    0.991466
PC1 minimum similarity: 0.985679

PC2 mean similarity:    0.946801
PC2 minimum similarity: 0.905622
```

PC1 remained highly stable throughout the sample.

PC2 experienced more noticeable long-term drift.

Therefore:

- PCA factors did not change abruptly.
- Some cumulative factor drift existed, particularly in PC2.
- The performance difference cannot be explained purely by PCA loading instability.

---

## 8. Signal Disagreement

The next hypothesis was that small PCA changes could cause residual z-scores to cross the hard trading threshold.

Signals from fixed PCA and refitted PCA were compared.

### Overall Results

```text
Overall stock-day signal disagreement: 3.51%

Fraction of days with at least one
signal disagreement:                   13.17%
```

### Per-Symbol Signal Disagreement

```text
AAPL: 2.63%
MSFT: 2.03%
GOOG: 2.23%
AMZN: 3.75%
NVDA: 6.89%
```

Approximately 96.5% of individual stock-day trading signals were identical.

Therefore, repeated PCA refitting did not cause widespread signal instability.

However, even a relatively small number of signal differences may affect portfolio weights after normalization.

---

## 9. Position Differences

The L1 distance between fixed-PCA and refitted-PCA portfolio weights was calculated.

Results:

```text
Mean position distance:  0.1299
Median:                  0.0000
75th percentile:         0.0000
Maximum:                 2.0000
```

The fraction of trading days with different portfolio positions was:

```text
13.17%
```

Therefore, most trading days produced exactly the same portfolio.

However, on a relatively small number of days, portfolio allocations differed substantially.

A maximum L1 distance of `2.0` represents a very large difference in portfolio positioning.

---

## 10. Daily PnL Difference

Daily strategy returns from fixed PCA and refitted PCA were compared.

The fixed-minus-refitted daily return difference had the following distribution:

```text
Mean:      0.000156
Std:       0.007371
25%:       0.000000
Median:    0.000000
75%:       0.000000
Minimum:  -0.054125
Maximum:   0.068328
```

The 25th, 50th, and 75th percentiles were all zero.

This means that the two strategies produced identical PnL on most trading days.

However, on a small number of days, their daily returns differed substantially.

---

## 11. PnL Attribution

Because the strategy uses lagged positions, a signal difference on day `t` affects strategy PnL on day `t+1`.

Days were therefore divided into:

- **Affected days:** days following a signal / position disagreement
- **Normal days:** all remaining days

Results:

```text
Affected days: 130

Mean PnL difference on affected days:
0.001181

Arithmetic sum of PnL differences
on affected days:
0.1535

Mean PnL difference on normal days:
0.0000

Arithmetic sum of PnL differences
on normal days:
0.0000
```

All observed daily PnL divergence between fixed and refitted PCA occurred on the relatively small set of affected days.

The arithmetic sum of daily return differences is not equivalent to the difference between compounded total returns, but it provides a useful attribution measure.

---

## 12. Interpretation

The evidence does not support the hypothesis that repeated PCA refitting creates large or unstable changes in the principal components.

Instead, the evidence is more consistent with the following mechanism:

```text
Small PCA estimation changes
        ↓
Small residual changes
        ↓
Small z-score changes
        ↓
Occasional crossing of the hard ±2 threshold
        ↓
BUY / SELL / HOLD signal changes
        ↓
Portfolio weights are renormalized
        ↓
Portfolio composition changes
        ↓
PnL differs on a relatively small number of days
        ↓
Long-run compounded performance diverges
```

The main issue is therefore not large PCA instability.

Instead, the current PCA strategy appears sensitive to small changes in model estimation because continuous residual z-scores are converted into discrete trading decisions using a hard threshold.

A small movement such as:

```text
Fixed PCA z-score:    2.01 → SELL
Refitted PCA z-score: 1.99 → HOLD
```

may therefore produce a large change in the resulting portfolio.

---

## 13. Main Findings

### Walk-Forward Framework

- The expanding-window walk-forward framework was successfully implemented.
- The mean-reversion sanity check produced a maximum return difference of exactly zero.
- The implementation therefore preserves strategy behaviour across fold boundaries.

### Mean Reversion

- Long-horizon mean-reversion performance was substantially weaker than the shorter Week 6 validation result.
- The difference is primarily related to the longer evaluation period rather than to the walk-forward implementation.

### PCA

- Fixed PCA significantly outperformed repeatedly refitted PCA over the same OOS dates.
- PCA loadings were extremely stable between adjacent folds.
- Some cumulative drift existed, especially in PC2.
- Only around 3.5% of individual stock-day signals differed.
- Portfolio positions differed on approximately 13.2% of trading days.
- The entire observed daily PnL divergence occurred on those affected trading days.
- The strategy therefore appears sensitive to threshold crossings and portfolio normalization.

---

## 14. Research Implication

The large difference between fixed and refitted PCA performance should not immediately be interpreted as evidence that fixed PCA is superior.

Instead, the results indicate that the strategy is sensitive to small model-estimation changes.

The key research question becomes:

> Is the strategy profitable across a stable region of model and trading parameters, or does profitability depend on a small number of fragile parameter choices and threshold crossings?

This question will be investigated in Day 2.

---

## 15. Implications for Day 2

Week 7 Day 2 will focus on **hyperparameter discipline and robustness analysis**.

Parameters to investigate include:

- Z-score rolling window
- Signal threshold
- Number of PCA components

The objective is not to identify the single parameter combination with the highest Sharpe ratio.

Instead, the objective is to identify stable regions of parameter space where nearby parameter combinations produce similar performance.

A robust parameter region is preferable to an isolated performance peak.

For example:

```text
Robust region:

               threshold
window       1.5    2.0    2.5

10           0.4    0.6    0.3
20           0.7    1.0    0.8
30           0.6    0.9    0.7
```

is more convincing than:

```text
Potential overfit:

               threshold
window       1.5    2.0    2.5

10          -0.2    0.1   -0.3
20           0.0    2.5   -0.1
30          -0.3    0.2    0.0
```

The Day 1 results make threshold robustness particularly important because relatively small changes around the decision boundary were capable of producing economically meaningful differences in portfolio PnL.

---

## 16. Research Status

**Week 7 Day 1: Completed**

Completed tasks:

- Expanding walk-forward splitter
- Historical continuity handling
- Mean-reversion walk-forward validation
- Mean-reversion sanity check
- PCA walk-forward refitting
- Fixed vs refitted PCA comparison
- PCA adjacent loading stability analysis
- PCA long-term drift analysis
- Signal disagreement analysis
- Position disagreement analysis
- PnL attribution

Next:

**Week 7 Day 2 — Hyperparameter Discipline and Robustness Analysis**