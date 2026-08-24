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
project-generated candidates are in scope for execution.

# Configurable LLM integration (optional)

AgentForge remains deterministic by default. Stage five adds three explicit modes:

- `deterministic`: no `.env` loading and no network/API calls.
- `hybrid`: use the configured LLM for requirement parsing and planning, with an explicit deterministic fallback.
- `llm`: require a working provider configuration unless fallback is explicitly enabled.

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `OPENAI_MODEL` only when you
intend to make real calls. ChatGPT Plus and API billing are separate; a Plus subscription
does not provide API credits. The default OpenAI integration uses the Responses API.
OpenAI-compatible endpoints use Chat Completions and require `OPENAI_BASE_URL` plus
`OPENAI_API_MODE=chat_completions`.

```bash
python -m agentforge.cli demo
python -m agentforge.cli demo --mode hybrid --allow-llm-fallback
python -m agentforge.cli demo --mode llm
```

LLM code generation and repair are disabled by default and require their corresponding
CLI flags. Every generated program still passes the existing syntax, AST security,
interface and bounded-subprocess checks. Prompts, provider/model, call status, usage,
fallbacks and generation modes are recorded; API keys and private reasoning are not.
Real calls may incur cost, latency, provider rate limits, and data-handling implications.
