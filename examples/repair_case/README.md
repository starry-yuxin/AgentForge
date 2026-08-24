# Missing-imputer repair example

The deterministic fault injection routes Logistic Regression attempt-0 to a training helper without imputers. Execution fails on real missing values and is classified as `MissingValueError`.

AgentForge retrieves the `missing_value_error` FailureExperience and its `IMPROVED_BY` links, regenerates attempt-1 with numerical and categorical imputation, and reruns AST, interface and subprocess validation. The sanitized `repair.diff` is copied from that real deterministic repair run.
