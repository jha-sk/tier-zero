# ADR 0004: Four-way token accounting against a versioned price table

**Status:** Accepted · **Date:** 2026-09-07

## Context

The system commits to a p95 of $0.06 per answered request. A cost SLO is only
meaningful if the cost figure is correct and remains reproducible after prices
move.

## Decision

Account for tokens in **four classes**, never collapsed: `input`, `output`,
`cache_creation` (cache write), `cache_read`. Price them from a **versioned,
dated table** in `config/prices.toml`, and stamp the table version onto every
cost figure the system emits.

Attribute cost to the **whole trace**, not to individual calls. An agent turn
that routes, generates and samples a judge is one user-visible request and one
budget.

## Rationale

**Why four classes.** Cache writes carry a premium (1.25× input at the 5-minute
TTL, 2.0× at the hour) and cache reads a deep discount (0.1×). Treating cache
creation as ordinary input understates cost by exactly 20% at the 5-minute TTL
and 50% at the hour. There is a test asserting that arithmetic, because it is
the kind of error that makes a cost SLO report passing while it is not.

**Why the budget forces the architecture.** At current prices, for a
20K-context / 1K-output request:

| Model | Cost | Verdict |
|---|---|---|
| Haiku 4.5 | $0.025 | Fits, with room for retries |
| Sonnet 5 | $0.050 | Fits with 17% headroom |
| Opus 5 | $0.125 | 2× over budget |

Opus is not affordable as a default. With prompt caching — 18K of 20K served
from cache — the same request drops back under budget. So the tier structure and
the caching strategy are not embellishments; they are what the number requires.
Both facts are pinned by tests.

**Why a versioned table rather than constants.** Prices change. A benchmark
published today must remain recomputable tomorrow, and a cost attributed to a
price table version is; a cost attributed to a hard-coded literal is not. An
unknown model is an **error**, never a zero — a silently free model is how a
cost SLO passes while being wrong.

## Consequences

Cost is reported as a distribution (p50/p95/p99), never as total spend divided
by request count. A mean hides exactly the tail the SLO is about.
