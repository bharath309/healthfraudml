---
description: Orchestrator/worker mode — plan here, fan implementation out to parallel Sonnet workers, verify at the end. Usage - /orchestrate <task>
---

Run the requested task in orchestrator/worker mode. You (the main, expensive
model) only PLAN and VERIFY; cheap `worker` subagents do the building, so most
tokens are billed at the lower rate.

Task: $ARGUMENTS

1. **Plan (you).** Read enough of the codebase to decompose the task into
   INDEPENDENT units — no two workers may touch the same file. Typical unit:
   one module in `healthfraudml/`, one fraud detector, one test file. If units
   can't be made independent, do the shared part yourself first, then fan out
   the rest.
2. **Fan out.** Spawn one `worker` agent per unit, all in a single message so
   they run in parallel. Each prompt must be self-contained: exact files to
   touch, expected behavior, interfaces it must conform to, and the pytest
   `-k` selection that proves the unit works. Workers can't see your context —
   include everything they need.
3. **Handle reports.** If a worker returns a blocker, resolve the decision
   yourself and re-dispatch that unit; don't silently absorb its work into
   your own context.
4. **Verify (you).** After all workers return, run the full suite:
   `PYTHONPATH=. .venv/bin/pytest tests/ -v`. Fix integration seams yourself
   only if small; otherwise dispatch a fix-up worker. Then summarize per the
   CLAUDE.md style: units dispatched, files changed, test result.
