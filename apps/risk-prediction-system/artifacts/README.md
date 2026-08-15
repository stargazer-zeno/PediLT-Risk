# Bundled XGBoost artifacts

This directory contains the validated schema and three trained XGBoost models used by the prediction system:

```text
artifacts/
├── model_manifest.json
├── datasets/schema.json
└── xgboost/
    ├── xgb_sequence_Label_1m.json
    ├── xgb_sequence_Label_1y.json
    └── xgb_sequence_Label_5y.json
```

The public manifest records SHA-256 checksums, the 22,927-feature runtime contract, model targets, and delivery-validation AUROC values. Training records and patient-level predictions are private and are not included.

These models are released for research and software demonstration only. They are not validated for clinical decision-making.
