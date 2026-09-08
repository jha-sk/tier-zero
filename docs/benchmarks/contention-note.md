# Ingest/query contention hits embedding, not the vector store

**Measured 2026-09-07, during the 10k index build.**

A search benchmark run *while the indexer was running* produced:

| Stage | p95 (under ingest) | p95 (baseline, idle) |
|---|---|---|
| Query embedding | **61.05 ms** | 8.70 ms |
| Qdrant search | 21.07 ms | not yet measured idle |
| **Retrieval total** | **76.86 ms** | — |

Query embedding degraded **7x**. This was not the failure mode the plan
anticipated.

## Why it matters

The published guidance on Qdrant ingest contention concerns the *vector store*:
during optimizer draining, search p50 has been measured going from ~4ms to
780ms and p95 to 2.0s, and the mitigations are all store-side
(`prevent_unoptimized`, `indexed_only`, an optimizer CPU budget). Those were
configured here from the start.

But this architecture put the encoder **in the same process as the query path**
— the decision that makes a 25ms budget reachable at all (ADR 0002), because it
removes a 15–40ms network hop. The consequence, which was not obvious in
advance, is that the encoder now competes for the same CPU as ingest. Embedding
is compute-bound and ingest embedding is *also* compute-bound, so they contend
directly, and no amount of Qdrant tuning touches it.

Put plainly: moving work in-process to remove a network hop also moved it into
contention with every other CPU-bound stage. That is a real trade, and it is
invisible until something else is running.

## Consequences

1. **The steady-state SLO must be measured with ingest quiescent**, and any
   figure taken under concurrent ingest reported separately and labelled. Both
   numbers are real; conflating them is what makes benchmarks dishonest.
2. **The mitigation is CPU isolation, not Qdrant configuration.** The ingest
   embedder needs a bounded thread pool leaving cores for the query path —
   the same argument as Qdrant's `optimizer_cpu_budget`, applied one layer up
   where the plan did not think to apply it.
3. A production deployment would separate ingest and serving entirely. On one
   machine, the honest statement is that the SLO holds at steady state and
   degrades ~7x during a full reindex, with the reindex being a scheduled,
   bounded event.

This is exactly the class of finding the project exists to surface: the
architecture's headline decision bought the latency budget, and the bill came
due somewhere the plan was not looking.
