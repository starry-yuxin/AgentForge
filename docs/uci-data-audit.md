# UCI Iranian Churn data credibility audit

This document records a deterministic pre-commit sensitivity analysis of the
[UCI Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)
(DOI `10.24432/C5JW3Z`, CC BY 4.0). It does not replace AgentForge's primary
workflow result and does not change the official feature set.

## Definitions and method

- A complete processed-row duplicate includes `customer_id` and `churn`.
- A source-record duplicate excludes only the generated `customer_id`.
- A `feature_signature` is the SHA-256 of all ordered predictors, explicitly
  excluding both `customer_id` and `churn`.
- The ordinary split is the existing fixed, stratified 60/20/20 split.
- The sensitivity split uses five-fold `StratifiedGroupKFold`, fixed seed 42:
  three folds train, one validation, and one test. Every feature signature is
  confined to one partition. Its realized proportions were 60.03%, 20.03%, and
  19.94%; positive rates were 15.76%, 15.69%, and 15.61%.
- Preprocessing and models are fitted on train only. Thresholds and best candidates
  use validation F1 only; test is accessed only for final metrics.

## Duplicate audit

| Measure | Result |
|---|---:|
| Complete duplicates including generated ID, beyond first | 0 |
| Duplicates excluding `customer_id`, beyond first | 300 |
| Duplicate feature signatures, beyond first | 314 |
| Duplicate feature groups | 162 |
| Samples involved in duplicate feature groups | 476 |
| Same-feature groups containing both labels | 14 groups / 64 samples |

Duplicate group sizes were: 37 groups of size 2, 117 of size 3, two each of
sizes 4, 5, and 6, and one each of sizes 10 and 11.

The ordinary split placed identical feature signatures across partitions:

| Crossing | Shared groups | Samples in the two partitions |
|---|---:|---:|
| Train ↔ validation | 59 | 168 (101 train, 67 validation) |
| Train ↔ test | 70 | 195 (114 train, 81 test) |
| Validation ↔ test | 35 | 84 (41 validation, 43 test) |

This can make ordinary random-split estimates optimistic, especially for a model
capable of learning local interactions. Conflicting labels in 14 identical-feature
groups also show that duplicates are not simply safe copies to delete without a
separate data-governance decision.

## Ordinary versus duplicate-group isolation

| Split | Algorithm | Validation F1 | Test F1 | Test ROC-AUC | Threshold |
|---|---|---:|---:|---:|---:|
| Ordinary | Logistic Regression | 0.686192 | 0.669456 | 0.927790 | 0.625 |
| Ordinary | Random Forest | **0.861386** | **0.882353** | **0.980350** | 0.550 |
| Group isolated | Logistic Regression | 0.632035 | 0.607930 | 0.914613 | 0.750 |
| Group isolated | Random Forest | **0.851282** | **0.833333** | **0.984193** | 0.700 |

Group isolation reduces Random Forest test F1 by about 0.049 and Logistic
Regression by about 0.062. Random Forest remains the validation-selected candidate
and retains strong ROC-AUC, so duplicate crossing explains part, but not all, of
the high ordinary-split result.

## Potential proxy-field sensitivity

All rows below use the same group-isolated indices and seed. `Status` and
`Customer Value` are official UCI features. They are treated as potential
operational-proxy and derived-variable risks, not asserted to be target leakage.

| Features | Algorithm | Validation F1 | Test F1 | Test ROC-AUC | Threshold |
|---|---|---:|---:|---:|---:|
| Full | Logistic Regression | 0.632035 | 0.607930 | 0.914613 | 0.750 |
| Full | Random Forest | **0.851282** | **0.833333** | **0.984193** | 0.700 |
| Without `status` | Logistic Regression | 0.603053 | 0.611570 | 0.916115 | 0.675 |
| Without `status` | Random Forest | **0.798030** | **0.790960** | **0.980169** | 0.675 |
| Without `status` and `customer_value` | Logistic Regression | 0.603053 | 0.611570 | 0.916269 | 0.675 |
| Without both | Random Forest | **0.807882** | **0.823529** | **0.982595** | 0.650 |

Removing `status` lowers Random Forest validation/test F1, which is consistent
with it carrying useful predictive information. Removing `customer_value` as well
does not cause a further collapse in this fixed split. These results do not justify
removing or retaining either field for production; availability, construction,
and governance must be assessed independently. The primary workflow continues to
use the complete official feature set.

## `customer_id` verification

`customer_id` is generated solely as a stable row identifier. Audit tests prove it
is removed before feature splitting, absent from `ColumnTransformer`, and changing
every ID leaves model input frames unchanged. It therefore cannot affect current
predictions. Retaining it provides row-level traceability; removal is not currently
recommended unless that tracing requirement is deliberately dropped.

## Conclusions and limits

The experiment supports that AgentForge can evaluate the public dataset without
feature-signature overlap and that Random Forest's advantage remains under the
tested sensitivities. It also supports reporting the ordinary split as potentially
optimistic rather than presenting its score alone.

It does not prove production generalization, causal validity, absence of every
proxy or leakage mechanism, temporal robustness, fairness, or that duplicate rows
are erroneous. Only one dataset, one fixed group assignment, and two algorithms
were studied. No LLM was called and no knowledge-graph data was persisted.
