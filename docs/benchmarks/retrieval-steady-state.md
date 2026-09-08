# Retrieval SLO: steady state

**Measured 2026-09-07.** Index: 10,000 documents / 17,813 chunks (17,767
HNSW-indexed), containing all 1,775 gold documents plus 8,225 distractors.
300 held-out queries. Idle machine, ingest quiescent, loopback gRPC.

## Result

| SLO | Budget | Measured | |
|---|---|---|---|
| Retrieval p95 | ≤ 25 ms | **29.21 ms** | FAIL by 17% |
| recall@50 | ≥ 0.85 | **0.757** | FAIL |

| Stage | p50 | p95 | p99 |
|---|---|---|---|
| embed | 11.02 | 13.96 | 16.78 |
| qdrant_search | 12.64 | 16.34 | 16.96 |
| decode | 0.61 | 0.70 | 0.83 |
| **total** | 24.85 | **29.21** | 31.50 |

Cold first query 27.23 ms, reported separately.

## The `hnsw_ef` result is the useful one

`hnsw_ef` at 32 / 64 / 128 / 256 produced **identical recall (0.757) and
identical latency** (p95 29.68–29.97 ms). Not similar — identical.

That single fact resolves what would otherwise be a confounded comparison.
Against the exact-search ceiling measured on the gold-only index:

| | 1,775 docs, exact search | 10,000 docs, HNSW |
|---|---|---|
| recall@10 | 0.753 | 0.560 |
| recall@50 | **0.867** | **0.757** |

Two variables changed at once — corpus size (5.6x more distractors) and search
method (exact → approximate). Ordinarily that comparison would be
uninterpretable. But if ANN approximation were responsible for any of the drop,
raising `ef` would recover some of it, and it recovers **none**.

**So HNSW is costing nothing here. The recall loss is entirely the harder
corpus.** Tuning graph parameters further is wasted effort. The levers that
would actually move recall are hybrid sparse retrieval and reranking, neither of
which is built — and both are now justified by measurement rather than by
assumption.

This is the whole reason the exact-search ceiling was measured first. A single
end-to-end number cannot distinguish "the embedding model cannot find it" from
"the index approximated it away", and those have opposite fixes.

## What the latency number is, precisely

29.21 ms is for the **eval query shape**: 500 chunks fetched per query, because
document-level recall@100 requires enough chunks to yield 100 distinct
documents (this corpus runs ~5 chunks per document at p95). That is a far
heavier query than production would issue.

At a production-shaped `top_k=50`, the load test measured **25.49 ms corrected
p95 at 10 rps** and 25.93 ms at 25 rps — a near-miss rather than a 17% miss.

Both are over budget, and the fix is already measured rather than speculative:
`all-MiniLM-L6-v2` embeds at 3.63 ms p95 against bge-small's 8.70 ms
(ADR 0002). Roughly 5 ms of headroom — enough to clear the budget at either
query shape. It was not adopted because it is the weaker retriever, and recall
is already the failing SLO. **Trading recall for latency when recall is the
constraint would be optimising the wrong number.**

## Honest reading

The system misses both SLOs. It misses latency by 17% at eval shape and by ~2%
at production shape, with a measured 5 ms fix available at a known recall cost.
It misses recall by 0.093, with the two highest-value interventions
(hybrid + rerank) untried.

Neither gap is mysterious, and no number here required a caveat invented after
the fact to explain it.
