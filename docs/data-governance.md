# Data and model governance

## Public material

- Source code and configuration templates
- Bundled XGBoost model artifacts and their checksums
- Aggregate evaluation tables
- Explicitly synthetic examples without real identifiers, outcomes, or exact clinical dates

## Excluded material

- Raw or processed EHR records
- Patient and visit identifiers
- Patient-level model predictions and LLM logs
- SFT/GRPO datasets and training checkpoints
- Credentials, internal endpoints, job uploads, and runtime results

Clinical data remain in institutionally controlled storage. Access, processing, and model release require the applicable institutional, ethics, privacy, and data-use approvals. GitHub issues and CI artifacts must never be used to exchange clinical data.

The released system is a research demonstration, not a clinical service. Any new deployment is responsible for authentication, TLS, access control, audit, retention, incident response, and local regulatory review.
