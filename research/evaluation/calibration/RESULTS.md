# Reported aggregate results

The released CSV files and figures were generated from the internal patient-disjoint calibration experiment described in `CALIBRATION_PROTOCOL.md`. The base XGBoost models were unchanged. Platt coefficients were fit using 392 calibration patients and evaluated on 789 different retained-test patients.

| Horizon | Original Brier score | Platt Brier score | Change |
|---|---:|---:|---:|
| 1 month | 0.0413 | 0.0056 | -0.0357 |
| 1 year | 0.0804 | 0.0196 | -0.0608 |
| 5 years | 0.1254 | 0.0495 | -0.0759 |

Platt scaling preserves rank order: retained-test AUROC was unchanged at 0.9083, 0.8182, and 0.7333 for the three horizons. The corresponding aggregation, calibration, decision-curve, bootstrap, stage, and split-balance summaries are in `results/tables/`; the three figures are in `results/figures/`.

These results describe one internal study cohort. They do not demonstrate external validity, and the published coefficients are not a clinical deployment setting.
