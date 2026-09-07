# Bias Audit

## 1. Look-Ahead Bias

Status: PASS

- Signals only use historical/current information.
- Positions are applied to subsequent returns where required.
- Walk-forward testing preserves chronological order.
- PCA is fitted only on training data.

## 2. Data Leakage

Status: PASS

- Training data is used for model fitting.
- Validation data is used for hyperparameter selection.
- Test data is not used during tuning.
- Final test evaluation uses parameters selected before seeing test performance.

## 3. Data Snooping

Status: CONTROLLED

- Hyperparameter search is performed only on validation data.
- Parameter robustness is checked around the selected optimum.
- Test performance must not be used to retune parameters.

## 4. Survivorship Bias

Status: NOT YET CONTROLLED

- Current asset selection may rely on securities that exist today.
- A historical point-in-time universe is not yet implemented.
- Results involving multi-stock universes should therefore be interpreted with caution.

## Conclusion

The current research pipeline controls look-ahead bias and data leakage reasonably well.
The main unresolved issue is survivorship bias.
The final test set should remain untouched until the research design is frozen.