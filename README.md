# AgentForge

A lightweight deterministic AI-Agent algorithm capability factory with knowledge retrieval,
code generation, automated validation, and iterative repair.

## Controlled execution safety boundary

Generated candidates are checked with a lightweight AST policy and interface contract, then
executed by the current virtual-environment interpreter in a timeout-bounded subprocess with
`shell=False`, a dedicated working directory, captured output, and a minimal environment.

**Subprocess isolation and AST checks reduce accidental risk but do not constitute a
production-grade security sandbox.** They do not provide kernel, container, filesystem,
network, memory, or syscall isolation and cannot detect every dynamic Python behavior. Only
project-generated deterministic templates are in scope for execution.
