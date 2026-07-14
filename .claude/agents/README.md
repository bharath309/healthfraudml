# Agent setup: orchestrator/worker + executor/advisor

Two cost-tiering patterns for running Claude Code on this repo. Both keep the
expensive frontier model's token share small so most cost bills at the cheap
model's rate.

## Pattern 1 — Orchestrator/worker (`/orchestrate <task>`)

```
Main loop (frontier model) --plan--> fan out --> worker (Sonnet)  [own loop]
                                             --> worker (Sonnet)  [own loop]
                                             --> worker (Sonnet)  [own loop]
```

The main session only plans, dispatches, and verifies. Each `worker`
(`worker.md`, `model: sonnet`) implements one independent unit and self-tests
before reporting back. Use for tasks that split cleanly: new fraud detector +
its tests + docs, refactors across `preprocessing`/`models`/`fraud_types`, etc.

## Pattern 2 — Executor/advisor (`advisor` agent)

```
Main loop (cheap model, every turn) --tool call--> advisor (frontier, on-demand)
                                    <--advice-----
```

Inverse of pattern 1: run the session itself on a cheap model (`/model
sonnet`), and let it escalate to the read-only `advisor` (`advisor.md`,
`model: inherit` — set the session or Agent-call model to the frontier tier
when using this pattern) only when stuck. Budget ~one advisor call per task.
The advisor diagnoses and recommends; the executor implements.

## Choosing

| Task shape                              | Pattern                     |
|-----------------------------------------|-----------------------------|
| Big, decomposable build/refactor        | 1 — `/orchestrate`          |
| Routine work with occasional hard calls | 2 — cheap session + advisor |
