# Reproducibility boundary

The repository supports two levels of reproduction:

1. **Public smoke reproduction:** run parsers, feature-contract checks, bundled XGBoost inference, frontend tests, Docker builds, and synthetic examples without private data.
2. **Authorized full reproduction:** configure paths to an authorized clinical cohort and patient-level prediction files stored outside the repository, then run the training and statistical-analysis commands documented under `research/`.

Reported aggregate results cannot be reconstructed from the public repository alone because the underlying clinical records and patient-level predictions are intentionally excluded. Training scripts use patient-level splits to prevent follow-up nodes from the same patient crossing train and test sets.

Record the repository commit, Python environment, random seeds, input cohort version, model hashes, and Hugging Face model revision for every reproduction run.
