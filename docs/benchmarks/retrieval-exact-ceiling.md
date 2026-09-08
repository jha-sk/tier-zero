# Retrieval: exact-search recall ceiling

**Measured 2026-09-07.** Index: 1,775 gold documents / 4,265 chunks, no
distractors. 150 held-out queries from the duplicate-linked golden set.

## What this measures

`indexed_vectors_count` was **0**: with 12 segments and an indexing threshold of
1,000 points per segment, a 4,265-point collection stays below the threshold and
Qdrant brute-forces. Confirmed by the sweep — recall is **identical at every
`hnsw_ef` value** (32 / 64 / 128 / 256), which only happens when the parameter
is not being used.

That makes this an **exact-search** result, and it is the more useful one to
have first: it is the embedding model's true recall ceiling on this task,
independent of any ANN approximation. The HNSW measurement can then be compared
against it to separate *embedding* loss from *approximation* loss — two failures
that a single end-to-end number cannot distinguish.

## Quality — passes the floor

| Metric | Value |
|---|---|
| recall@1 | 0.420 |
| recall@5 | 0.660 |
| recall@10 | 0.753 |
| **recall@50** | **0.867** — SLO floor is 0.85, **PASS** |
| recall@100 | 0.920 |
| nDCG@10 | 0.584 |
| MRR | 0.536 |
| Complete misses | 12 of 150 |

Dense-only retrieval, no reranking, no hybrid sparse. Against human-authored
duplicate judgements — a moderator decided these two questions have the same
answer — 86.7% of the time the canonical question is in the top 50 documents.

The recall curve is the argument for retrieving deep and reranking rather than
retrieving shallow: **0.420 → 0.867 going from k=1 to k=50**. A pipeline that
fetches five candidates has already discarded more than a third of the answers
it could have found, and no reranker can recover a document that was never
retrieved.

These are a **lower bound**: duplicate judgements name *a* relevant document,
not every one, so a genuinely good answer the set does not know about scores as
a miss.

## Latency — not a valid steady-state measurement

| Stage | p50 | p95 |
|---|---|---|
| embed | 40.93 | **80.13** |
| qdrant_search | 33.09 | 62.94 |
| decode | 3.05 | 6.70 |
| **total** | 82.50 | **115.39** |

Against the 25ms budget this reads as a 4.6x failure, and it is **not reportable
as one**. The distractor indexing phase was running throughout, saturating all
12 cores. Query embedding measured 80.13ms p95 here against **8.70ms idle** —
a 9x degradation, consistent with (and worse than) the 7x recorded in
[contention-note.md](contention-note.md).

Two separate effects are stacked in that number: CPU contention with ingest, and
brute-force search over 4,265 chunks instead of an HNSW traversal.

A steady-state measurement requires ingest quiescent and the HNSW graph built.
That measurement is pending; this file will not be updated to claim otherwise.

## What this run established

1. The retrieval path works end to end against real data.
2. The embedding model clears the quality floor on this task without hybrid
   retrieval or reranking — so those become measurable improvements rather than
   necessities.
3. Depth matters more than any single tuning knob: 0.420 at k=1 against 0.867
   at k=50.
4. The latency budget cannot be evaluated while ingest runs, which is itself a
   finding about this architecture rather than a limitation of the benchmark.
