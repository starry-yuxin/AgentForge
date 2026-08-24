# Real-world dataset validation

AgentForge includes an opt-in adapter for the **Iranian Churn Dataset** from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)
(DOI `10.24432/C5JW3Z`, CC BY 4.0). The repository does not commit the downloaded
archive, extracted CSV, processed CSV, or run artifacts.

## Reproduce

```bash
python scripts/prepare_uci_churn.py
python -m agentforge.cli run \
  --mode deterministic \
  --no-persist \
  --dataset data/external/uci_iranian_churn/processed/uci_iranian_churn.csv \
  --output-root outputs/uci-real-churn \
  --request "请为UCI真实电信客户流失数据比较Logistic Regression和Random Forest，以F1作为主要指标。"
```

Preparation uses the registered HTTPS download URL, a timeout, a temporary file
and atomic replacement. Existing archives are reused unless `--force` is given.
ZIP paths and the exact member name are validated before extraction. The raw
schema must match the official 14-column CSV exactly.

## Data contract and transformations

- 3,150 rows, 13 input features, target `Churn` (`0`/`1`), and no missing values.
- Original names are normalized through an explicit mapping; `Churn` becomes
  `churn`, matching the existing runtime contract.
- Official integer-coded categories (`Complains`, `Charge Amount`, `Age Group`,
  `Tariff Plan`, and `Status`) become stable string category labels so the existing
  sklearn pipeline handles them as categorical rather than continuous features.
- A stable `customer_id` is derived from source row order and excluded from model
  features by the existing runtime.
- Rows and feature values are retained, including duplicates. Preparation does
  not impute, scale, encode, fit, resample, or tune anything.
- Source, license, timestamps, class counts, transformations, leakage review, and
  SHA-256 values are written to the ignored
  `data/external/uci_iranian_churn/metadata/dataset_metadata.json` file and copied
  into workflow JSON reports.

## Leakage review

UCI documents the attributes as aggregation over the first nine months and the
churn outcome at month 12, separated by a three-month gap. The target is removed
from model features; `customer_id` is also excluded. AgentForge fits preprocessing
only on the training split, chooses thresholds and candidates on validation, and
uses test data only for final evaluation.

Two deployment-time risks remain and are reported rather than hidden: `Status`
may be an operational proxy whose availability must be confirmed, and `Customer
Value` is a derived business feature whose upstream calculation should be audited.
No evidence in the published schema identifies either field as a direct target
derivative, so this reproducibility run retains them.

## Interpretation boundary

Results on this public dataset demonstrate a reproducible end-to-end run, not
production performance, business generalization, causal validity, fairness, or
deployment readiness. The fixed stratified random split is appropriate for this
repository demo but is not a substitute for temporal or organization-specific
validation.

## Reproduced result (2026-08-24)

| Candidate | Validation F1 | Test F1 | Test ROC-AUC | Threshold |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.686192 | 0.669456 | 0.927790 | 0.625 |
| Random Forest | **0.861386** | **0.882353** | **0.980350** | 0.55 |

Random Forest was selected solely because its validation F1 was higher. Test
metrics were computed only after selection and were not used for threshold search
or ranking. Both candidates exceeded the default minimum validation F1 of 0.60.
The positive class is 495/3,150 (15.71%), so this dataset is materially imbalanced;
unlike the synthetic demo it has no missing values and consists of numeric or
integer-coded fields. These differences plausibly explain the different model
ranking and scores, but do not establish a causal explanation.
