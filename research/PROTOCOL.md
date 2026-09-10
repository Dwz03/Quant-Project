# Research Protocol

This protocol defines the minimum evidence standard for standardized strategy research in this repository. It keeps research conclusions versioned, reproducible, and separate from broker execution.

## Data roles and selection

- Split data chronologically into train, validation, and held-out test periods.
- Use training data to fit models, estimate parameters, and provide required historical context.
- Use validation data to select strategies and hyperparameters. Validation results are model-selection performance, not an unbiased out-of-sample estimate.
- Keep held-out test data completely outside parameter selection. Test performance is the final untouched evaluation for that hypothesis version.
- Prevent look-ahead bias and state leakage: every signal, model state, normalization, and feature at a timestamp must use only information available by that timestamp.

## Robustness and comparisons

- Treat a single train/validation/test split as necessary but insufficient evidence of robustness.
- Use chronological walk-forward evaluation to measure performance across multiple unseen periods and market regimes.
- Examine parameter stability. A strategy whose selected parameters or nearby-candidate results change materially across folds requires caution.
- Compare strategies with relevant passive benchmarks on the same dates and under consistent return and cost conventions.
- Report turnover and transaction costs alongside gross and net performance. High gross performance is not sufficient when trading costs materially reduce it.

For stitched walk-forward results, preserve continuous positions and cost accounting across fold boundaries. Do not erase turnover or transaction costs by independently resetting each test fold.

## Versioning and decision discipline

- Never tune the same strategy version after observing its held-out or walk-forward out-of-sample results.
- Once out-of-sample evidence has been observed, retain it as evidence for that version.
- A materially changed hypothesis, signal, universe, sizing rule, or selection process must become a new version with a newly defined evaluation.
- Preserve negative results. Rejected or inconclusive experiments remain useful baselines and protect against repeating data-mined searches.

## Research lifecycle

```text
Hypothesis
  -> Experiment design
  -> Train/validation development and selection
  -> Held-out test
  -> Passive benchmark comparison
  -> Walk-forward robustness evaluation
  -> Evidence record
  -> Conclusion
  -> Decision: reject, retain as baseline, or advance as a new version
```

Research reports should record the data, universe, strategy version, candidate parameters, cost assumptions, selected parameters, validation and test metrics, benchmark results, walk-forward configuration, limitations, conclusion, and next decision.
