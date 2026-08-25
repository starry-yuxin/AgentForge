# Sanitized real LLM requirement validation

`deepseek_requirement_validation.json` records one successful, paid compatibility-endpoint
call used only to validate `RequirementAgent` structured parsing. It contains no API key,
authorization header, base URL, raw provider response, or local absolute path.

This evidence does **not** validate free-form LLM code generation or LLM code repair. The
default deterministic demo still makes zero API calls. Provider behavior and model output
can change over time, so the checked-in JSON is historical, sanitized evidence rather than
a guaranteed live-service result.
