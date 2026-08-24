# AgentForge deterministic example report

> This committed example uses synthetic customer churn data and makes zero LLM calls.

- Mode: `deterministic`
- Dataset: `data/churn_sample.csv`
- Primary metric: `f1`
- Best candidate: `logistic_regression`
- Selection rule: validation metrics only
- Test role: final reporting only; never candidate selection

| Candidate | Validation F1 | Test F1 | Test ROC-AUC | Threshold |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.695652 | 0.673913 | 0.914502 | 0.675 |
| Random Forest | 0.631579 | 0.604651 | 0.885762 | 0.400 |

Both candidates passed the AST security policy, interface contract and controlled subprocess execution. These synthetic-data metrics do not demonstrate real-business generalization.
