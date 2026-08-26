# Calibration protocol

## Objective

Assess whether post-hoc probability calibration improves the probability scale of the fixed 300-tree XGBoost models at 1-month, 1-year, and 5-year horizons, without retraining the base models.

## Design

1. Build a patient-level table from the authorized source predictions and select a patient-disjoint calibration subset using stratification by outcome/follow-up pattern.
2. Search candidate splits using a fixed seed and select the split with the lowest maximum standardized mean difference across supplied static features. The reported analysis used 5,000 candidates, 392 calibration patients, and at least 20 positive patients at each horizon.
3. Fit Platt scaling on the logit of clipped source probabilities in the calibration subset. Fit isotonic regression only as a sensitivity analysis; do not serialize or distribute it.
4. Apply the fitted mappings unchanged to predictions for the retained patients.
5. Report AUROC, AUPRC, Brier score, calibration intercept/slope, ECE, binned calibration, decision-curve analysis, stage summaries, and patient-cluster bootstrap comparisons.

## Interpretation rules

- Platt scaling is monotonic, so AUROC and AUPRC should be unchanged apart from numerical precision.
- Brier score and calibration summaries assess probability quality; they do not establish clinical utility by themselves.
- Decision curves are descriptive study outputs. No threshold in this module is a deployment recommendation.
- The patient-disjoint retained test is internal. External validation and local recalibration remain necessary before clinical use.
