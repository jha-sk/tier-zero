# TierZero benchmarks

Every number here is reproducible with the command beside it. Numbers that have
not been measured are marked as such rather than estimated.

Hardware for all runs: 12-core x86_64, 22 GB RAM, local NVMe, single machine.
Qdrant 1.19.1 on loopback gRPC, co-located with the client.

---

## 1. Corpus ingestion — streaming

`cargo run --release -p tz-ingest --bin tz-scan -- data/serverfault/Posts.xml`

| Rows | Peak RSS |
|---|---|
| 200,000 | 5 MB |
| 844,092 (full 1.21 GB) | 6 MB |

4.2x the input for 1.2x the memory. 65,841 rows/s, 94.5 MB/s.
Raw output: [corpus-scan.md](corpus-scan.md)

Corpus: 325,169 questions, 514,457 answers, 3,920 distinct tags. 47.1% of
questions have an accepted answer; 31.6% of posts contain code.

## 2. Chunking — the budget invariant holds

`cargo run --release -p tz-ingest --bin tz-chunkstats -- data/serverfault/Posts.xml`

| Metric | Value |
|---|---|
| Documents → chunks | 325,169 → **638,156** |
| Chunk tokens | p50 353 · p95 493 · p99 509 · **max 512** |
| Over the 512-token encoder window | **0 (0.000%)** |
| Under 50 tokens (fragment signature) | 14,568 (2.28%) |
| Code chunks | 360,862 (**56.5%**) |
| Throughput | 182s full corpus (1,786 docs/s) |

Reaching `max 512` took five separate fixes; see
[ADR 0003](../adr/0003-chunking.md). One of them — chunks up to 12,971 tokens
from unsplittable base64 lines — was invisible to every unit test and appeared
only on the full corpus.

## 3. Query embedding — the latency budget's first stage

`cargo run --release -p tz-embed --example encoder_sweep`

| Encoder | Threads | p50 | **p95** | p99 |
|---|---|---|---|---|
| bge-small fp32 | default | 7.68 | **8.70** | 9.45 |
| bge-small fp32 | 1 | 18.64 | **19.60** | 19.97 |
| bge-small int8 | default | 14.81 | **16.05** | 16.46 |
| all-MiniLM-L6 fp32 | default | 3.17 | **3.63** | 3.80 |

Two priors refuted: INT8 quantization was **2x slower**, and pinning to one
thread was **2.3x worse** than the default fan-out. See
[ADR 0002](../adr/0002-in-process-embedding.md).

## 4. Ingest embedding throughput — the binding constraint

`cargo run --release -p tz-embed --example ingest_throughput`

At this corpus's measured median chunk length (350 tokens):

| Encoder | chunks/s | Full 638k-chunk corpus |
|---|---|---|
| bge-small (12 layers) | 7.8 | 22.7 h |
| all-MiniLM-L6 (6 layers) | 14.5 | 12.3 h |

**Sequence length dominates; batch size is within noise.** 2.7x the tokens costs
3.3x the time — superlinear, consistent with attention being quadratic in
sequence length. Batch 16/32/64 differ by less than run-to-run variance.

Consequence: a full-corpus index is not a same-day job on CPU. The index is
therefore scoped as an IR benchmark subset — **every gold document plus a
controlled number of distractors** — and recall is reported against distractor
count rather than as one number from one corpus size. Gold documents are
mandatory; recall against an index that lacks the answer measures nothing.

## 5. Ingest memory — bounded batches

Peak RSS during indexing: **15.4 GB → 1.7 GB** after capping the embedding
batch. Attention is O(batch × heads × seq²), so batch size is a memory knob
before it is a throughput knob. See
[ADR 0005](../adr/0005-ingest-memory-and-batching.md).

## 6. A benchmark-validity trap: the indexing threshold

Qdrant builds an HNSW graph for a segment only once it holds more than
`indexing_threshold` points — **20,000 by default** — and brute-forces below
that. For a small collection that default is correct: exact search is both
faster and perfectly accurate at that size.

It is a trap for a benchmark. This project's index is a deliberately scoped
subset sitting near that threshold, so with the default in place the benchmark
would have measured **exact brute-force search while reporting it as approximate
search**. The latency would have been wrong in one direction, and the recall
figure would have been the recall of brute force — trivially near-perfect and
saying nothing about the HNSW parameters the report claims to be sweeping.

`indexing_threshold` is therefore set to 1,000, and a test asserts it stays
below the benchmark corpus size. Every run prints `indexed_vectors_count` beside
`points_count`; if the first is zero, the numbers below it are not ANN numbers.

The general form: **a benchmark on a scoped subset can silently exercise a
different code path than production.** Checking which path actually ran is part
of the measurement, not a detail beneath it.

## 7. Retrieval quality — exact-search ceiling

`cargo run --release -p tz-retrieve --bin tz-searchbench`

150 held-out queries, 1,775-document gold index, dense-only, no reranking:

| Metric | Value |
|---|---|
| recall@1 | 0.420 |
| recall@10 | 0.753 |
| **recall@50** | **0.867** — floor 0.85, **PASS** |
| recall@100 | 0.920 |
| nDCG@10 · MRR | 0.584 · 0.536 |

`hnsw_ef` had **no effect** at 32/64/128/256, which is how we know the
collection was below the per-segment indexing threshold and Qdrant was
brute-forcing. That makes this the embedding model's exact-search ceiling — the
baseline the ANN result gets differenced against, separating embedding loss from
approximation loss.

Full detail: [retrieval-exact-ceiling.md](retrieval-exact-ceiling.md)

## 8. Instrumentation traps found while building the dashboard

Two, both of which produce a **believable wrong number** rather than an error:

- **Default histogram buckets.** OTel's defaults are sized for milliseconds; the
  instrument records seconds, so an 80ms operation reported a p95 of 4,750ms.
  The cost SLO was unmeasurable for the same reason — no bucket boundary between
  $0 and $5. [histogram-buckets.md](histogram-buckets.md)
- **Invented metric names.** The first dashboard queried
  `qdrant_collection_points_count`, which does not exist; Qdrant exposes
  `collection_points`. A Grafana panel querying a non-existent metric renders
  **empty, not an error**.

## 9. Retrieval SLO

Pending — the index build is the gate. Will report per-stage p50/p95/p99
**paired with recall@k**, swept over `hnsw_ef`, with cold-start reported
separately.

A measurement taken during concurrent ingest is already recorded in
[contention-note.md](contention-note.md): query embedding degraded 7x under
ingest load, which is a property of running the encoder in-process and is not
fixable with Qdrant configuration.

## How to read any number here

Four rules the tooling enforces, each because the corresponding mistake was made
at least once during this project:

1. **Latency is reported with its recall.** Any retriever can be made
   arbitrarily fast by returning less, and any recall can be bought with an
   unbounded `ef`. Only the pair is a claim.
2. **Percentiles come with their conditions.** Corpus size, arrival rate,
   warm or cold, and whether ingest was running. A figure taken during
   concurrent ingest is reported separately, never blended in.
3. **Projections use the workload's own distribution.** The throughput tool
   originally headlined its fastest row — 128-token chunks — for a corpus whose
   median chunk is 353 tokens, a 3x optimistic error. It now projects from the
   measured median and prints the fast row explicitly labelled as not this
   corpus.
4. **Check which code path actually ran.** See the indexing-threshold trap
   above.

## Not yet measured

Stated explicitly so the absence is not mistaken for a passing result:

- Retrieval p95 and recall at steady state (index build in progress)
- Hybrid dense+sparse fusion (dense implemented; BM25 sparse vectors not built)
- Cost per request against the 6¢ SLO (router and budget guard implemented and
  tested against a mock; no live API call made)
- TTFT and end-to-end generation latency
- Load test ramp (generator built and unit-tested; not yet run against a
  populated index)
- Semantic cache false-hit rate (guards implemented and tested; rate not
  measured on real traffic)
- `pass^k` consistency (not implemented)
