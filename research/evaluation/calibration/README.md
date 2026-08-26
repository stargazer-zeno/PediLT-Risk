# Probability calibration experiment

[中文说明](README.zh-CN.md)

This module reproduces a patient-disjoint probability-calibration experiment for the fixed XGBoost risk model. Platt scaling is the primary method and isotonic regression is a sensitivity analysis. The released material contains scripts, aggregate results, figures, and Platt coefficients only. It contains no EHR data, patient-level predictions, patient assignments, source manifests, model serializations, or clinical threshold configuration.

The reported analysis used 392 patients for calibration and 789 different patients for retained testing. It is an internal patient-disjoint evaluation, not an external validation. The model was not retrained.

## Public contents

- `run_calibration_analysis.py`: fits calibrators and creates private analysis outputs from authorized inputs.
- `verify_calibration_outputs.py`: independently checks a private run.
- `parameters/platt_parameters.json`: reported coefficients for research reproduction only; the application does not load them.
- `results/`: aggregate CSV tables and three figures with no patient-level records.
- `CALIBRATION_PROTOCOL.md` and `RESULTS.md`: prespecified workflow and interpretation of the released results.

## Authorized inputs

Create a private JSON manifest outside the repository. Paths may be relative to that manifest. The two prediction files must have `Patient_ID`, `Sample_ID`, `Target`, `True_Label`, and `Pred_Prob` columns. `Target` must be `1m`, `1y`, or `5y` (the `Label_` prefix is accepted). `Model` is used only when `model_name` is supplied.

```json
{
  "calibration_source_predictions": "private/saved_test_predictions.csv",
  "retained_evaluation_predictions": "private/common_cohort_predictions.csv",
  "model_name": "XGBoost",
  "sequence_test_npz": "private/sequence_test.npz",
  "schema_json": "private/schema.json",
  "training_patient_ids_csv": "private/training_patient_ids.csv",
  "model_paths": {
    "Label_1m": "private/xgb_1m.json",
    "Label_1y": "private/xgb_1y.json",
    "Label_5y": "private/xgb_5y.json"
  }
}
```

`sequence_test_npz` plus `schema_json`, or a one-row-per-patient numeric `patient_static_features_csv`, is optional but required to reproduce the static-feature balance selection. `training_patient_ids_csv` and `model_paths` enable additional verification checks. Do not commit the manifest or any input/output data.

## Run and verify

Install the dependencies listed in `research/requirements.txt`, then run from `research/` or set the Python path accordingly.

```bash
python evaluation/calibration/run_calibration_analysis.py \
  --input-manifest path/to/calibration.private.json \
  --output-dir path/to/private_calibration_output

python evaluation/calibration/verify_calibration_outputs.py \
  --input-manifest path/to/calibration.private.json \
  --output-dir path/to/private_calibration_output
```

The output directory must be outside this module. It contains patient-level intermediate files and must remain in an authorized location. The scripts use 5,000 candidate splits and 2,000 patient-cluster bootstrap replicates by default; smaller values are suitable only for smoke testing.

## Scope and limitations

The coefficients are tied to the fixed model, source predictions, outcome definition, and internal cohort used in this study. They are not integrated into `apps/risk-prediction-system`, do not replace a `null` LLM result, and are not a clinical deployment recommendation. Any prospective use requires independent validation, governance approval, and local recalibration.
