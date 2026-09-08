# ADR 0007: A router and one agent, not a multi-agent system

**Status:** Accepted · **Date:** 2026-09-07

## Context

The obvious architecture for an "agentic" system in 2026 is an orchestrator
delegating to specialist sub-agents. It demonstrates more machinery and is the
shape most portfolio projects reach for.

This project has a $0.06 p95 per-request budget, so the choice can be settled
with arithmetic rather than taste.

## The arithmetic

Anthropic's published figures for their own multi-agent research system:

| Workload shape | Token usage vs a chat turn |
|---|---|
| Chat | 1× |
| Single agent with tools | ~4× |
| Multi-agent | ~**15×** |

Against current rates, for a representative 20K-context / 1K-output request:

| Architecture | Effective tokens | Cost | Verdict |
|---|---|---|---|
| Single agent, Haiku 4.5 | ~1× | $0.025 | Fits, with retry headroom |
| Single agent, Sonnet 5 | ~1× | $0.050 | Fits, 17% headroom |
| Multi-agent, Haiku workers | ~15× | ~$0.375 | **6× over budget** |
| Multi-agent, Sonnet workers | ~15× | ~$0.750 | **12× over budget** |

There is no configuration in which a multi-agent system serves this workload
inside this budget. The decision is closed before any judgement about elegance
is required.

## What the multi-agent evidence actually supports

The +90.2% improvement Anthropic reports for multi-agent is real, and it is
reported on a **breadth-first research benchmark** with a large token budget —
work where the task genuinely decomposes into independent parallel searches and
where the answer is worth many dollars. Their own analysis attributes ~80% of
performance variance on that benchmark to token usage alone.

Read plainly: much of the multi-agent gain *is* the extra tokens. That
generalises to problems where more tokens help and the budget permits them. Ops
question-answering over a fixed corpus is not such a problem — the retrieval is
one search, not a research tree.

Also notable from the same source: upgrading the model beat doubling the token
budget on the weaker model. Model choice dominates orchestration cleverness,
which is the opposite of what a multi-agent architecture optimises.

## Decision

**Router → single agent with tools, with a bounded escalation path.**

```
request
  → PII redact → exact cache → semantic cache (guarded)
  → cheap non-LLM classifier
      ├─ tier 0: cached or trivial — no model call at all
      ├─ tier 1: single agent, Haiku 4.5, context capped at 12 chunks
      └─ tier 2: escalate to Sonnet 5, context capped at 8 chunks
  → output validation → sampled judge → stream
```

Two details are deliberate and counter-intuitive:

**The classifier is not a model call.** A model-based router adds 30–100ms and
its own tokens to *every* request, including the ones it routes to tier zero.
The cheapest path should not have to pay to be identified as cheap. A regex and
length classifier is a few microseconds and costs nothing.

**The expensive tier gets the *tighter* context cap** (8 chunks vs 12). Tier two
is the only tier that can actually breach the budget, so it is the one that must
be capped hardest. Capping context is also the main lever on time-to-first-token,
since prefill is linear in input length — so the same control serves the cost
SLO and the latency SLO at once.

## Enforcement, not observation

The budget is checked **before** each call, not reported after. `BudgetGuard`
projects cost from input size and `max_tokens` — the only output bound actually
enforced, rather than a hoped-for average — and refuses or trims. By the time a
p95 breach appears on a dashboard the money is spent; a budget that is only
charted is a report, not a budget.

When a call does not fit, the first response is to **trim context until it
does**, not to refuse the request. That is what makes the budget a control
rather than a tripwire.

## When to revisit

This decision should be reopened if: the per-request budget rises above roughly
$0.50; the workload shifts from question-answering to genuine multi-source
research where sub-tasks are independent; or measurement shows tier-two quality
failing in a way more context and a stronger single model do not fix. None of
those hold today, and each is checkable rather than a matter of opinion.
