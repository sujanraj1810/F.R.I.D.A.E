# FRIDAE V2 — Upgrade Notes

## What was changed

- Integrated the existing `ArtifactManager` into generated-code execution.
- Added controlled artifact helpers:
  - `save_dataframe_artifact(...)`
  - `save_csv_artifact(...)`
  - `save_json_artifact(...)`
  - `save_text_artifact(...)`
- Added artifact metadata to LangGraph state and Telegram document delivery.
- Added startup artifact cleanup and `/clear` artifact cleanup.
- Protected `/run.jsonl` with `X-Eval-Token` instead of exposing the log publicly.
- Stopped returning raw internal exceptions to Telegram users.
- Fixed the V2 executor/node interface mismatch (`run_python` arguments and result fields).
- Added compatibility aliases in the executor for the earlier V2 call shape.
- Added explicit `verification_ok` and response-type state fields.
- Fixed the ML repair route so a repaired ML run can return to deterministic execution.
- Serialized generated-code execution and added protection against starting another generated-code run while a timed-out thread is still alive.
- Improved public retrieval defaults to 30s timeout, 10 MB response cap, and 5 redirects; all remain environment-configurable.
- Made dataframe artifact writes temporary/atomic and size-checked before promotion.
- Added `.gitignore` protection for `.env`, runtime logs, caches, and runtime directories.
- Added unit tests for artifacts, AST execution guardrails, artifact helper injection, web URL guardrails, and graph compilation (graph test is dependency-gated).

## Validation performed

- Python syntax compilation: passed for all project Python modules.
- Test suite: 7 passed, 1 skipped in the current tool environment.
- The graph compilation test was skipped because the current execution environment does not have `langgraph` installed. The project requirements still pin `langgraph` for normal deployment.

## Important security boundary

FRIDAE still uses application-level AST guardrails plus a timed daemon thread, not a true OS-level sandbox. The upgrade reduces attack surface and concurrency hazards but does not turn arbitrary Python execution into a security sandbox.
