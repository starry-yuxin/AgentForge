# Sanitized UCI workflow result

- Date (UTC): 2026-08-24
- Mode / LLM calls: `deterministic` / `0`
- Dataset: `data/external/uci_iranian_churn/processed/uci_iranian_churn.csv`
- Target / selection metric / minimum: `churn` / `f1` / `0.60`
- AST and interface checks: passed for both candidates
- Knowledge persistence: `false`

| Candidate | Validation F1 | Test F1 | Test ROC-AUC | Threshold | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.686192 | 0.669456 | 0.927790 | 0.625 | 0.9788 |
| Random Forest | **0.861386** | **0.882353** | **0.980350** | 0.55 | 1.2724 |

`random_forest` was selected by validation F1 only. Test data did not participate
in candidate selection or threshold optimization. This result is a reproducible
public-data experiment and is not evidence of production readiness or business
generalization.
