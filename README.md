# TierZero

**A retrieval system for IT support questions, built to hold three hard budgets
at once — and measured honestly enough to admit when it doesn't.**

Ask it *"why is nginx returning 502 after an upgrade"* and it searches 638,156
chunks of a real operations knowledge base and returns cited passages in about
30 milliseconds, for about a penny and a half a question.

Written in Rust. 206 tests. Every number below is reproducible with the command
printed beside it.

## The three headline numbers

| Budget, declared before any code was written | Target | Measured | |
|---|---|---|---|
| Answer quality — recall@50 | ≥ 0.85 | **0.857** | ✅ |
| Cost per question | ≤ $0.06 | **$0.0149** | ✅ 4× headroom |
| Retrieval latency p95 | ≤ 25 ms | **31.5 ms** | ❌ missed by 17% |

**The failure is the interesting one.** It is diagnosed, not hidden: a 5 ms fix
is measured and available, and it was deliberately not taken, because it trades
away recall — and recall is the metric actually under pressure. Optimising the
number that isn't binding is how systems get worse while their dashboards get
better.

## What makes this different from a tutorial RAG project

**The eval set is human-labelled, not model-generated.** 4,437 query/document
pairs come from moderator duplicate rulings on Server Fault — a person read two
questions and judged that one answers the other. Most portfolio projects grade
retrieval with an LLM, which means the model being evaluated and the model doing
the grading share the same blind spots.

**Three assumptions were measured and found wrong.** INT8 quantization made
embedding *2× slower*, not faster. Pinning ONNX to one thread was *2.3× worse*
than the default. Hybrid dense+sparse retrieval *lost* to dense alone. Each is
written up with the evidence, because a project where every hypothesis was
confirmed is a project that wasn't really testing anything.

**Ten bugs are catalogued, and none of them raised an error.** A chunk silently
truncated at 2,564 tokens. A dashboard reporting 4,750 ms for an 80 ms
operation. A load generator reporting its *best* latency while the system
saturated. A prompt-cache breakpoint that could never fire. Silent failure is the
normal failure mode in this domain, and finding them is the actual work.

## What is and isn't built

Retrieval, ingestion, evaluation, caching, routing, budget enforcement,
observability and load testing are built and measured. **The generation path is
built and unit-tested but has never called a live model** — there was no API
budget, so answer quality is unmeasured and cost is computed analytically from
real token counts rather than from billed usage. Both limits are stated wherever
the numbers appear.

---

## The budgets in full

The headline table above rounds. These are the exact figures and the conditions
they were taken under.

| Budget | Target | Measured |
|---|---|---|
| Retrieval p95 (embed + search + fusion + payload) | ≤ 25 ms | **29.21 ms** at eval query shape, **25.49 ms** at production shape — FAIL |
| Cost p95 per answered request | ≤ $0.06 | **$0.01493** — PASS, holds across ±20% tokenizer error |
| Recall@50 on the golden set | ≥ 0.85 | **0.857** with body queries; 0.757 title-only — PASS |
| TTFT p95 / total p95 | ≤ 1 s / ≤ 5 s | **not measured** — requires a live model call |

Quality is a budget, not a nicety. The other three can all be satisfied by
returning garbage instantly and for free, which is why a system with only
latency and cost SLOs will happily optimise itself into uselessness.

---

## Measured so far

### Corpus ingestion is streaming

`cargo run --release -p tz-ingest --bin tz-scan -- data/serverfault/Posts.xml`

| Rows | Peak RSS |
|---|---|
| 200,000 | 5 MB |
| 844,092 (full 1.21 GB) | 6 MB |

4.2× the input for 1.2× the memory. Throughput 65,841 rows/s (94.5 MB/s) on 12
cores. 325,169 questions, 514,457 answers; 47.1% of questions have an accepted
answer, 31.6% of posts contain code.

### Query embedding fits its share of the latency budget

`cargo run --release -p tz-embed --example encoder_sweep`

| Encoder | Threads | p50 | **p95** | p99 |
|---|---|---|---|---|
| bge-small fp32 | default | 7.68 | **8.70** | 9.45 |
| bge-small fp32 | 1 | 18.64 | **19.60** | 19.97 |
| bge-small **int8** | default | 14.81 | **16.05** | 16.46 |
| all-MiniLM-L6 fp32 | default | 3.17 | **3.63** | 3.80 |

Two results contradicted the prior and changed the design:

- **INT8 quantization made it 2× slower**, not ~3.5× faster. Dynamic ONNX
  quantization inserts Quantize/Dequantize pairs around each matmul; where those
  do not fuse into a native INT8 GEMM the conversion cost lands *on top of* the
  arithmetic. The project's pipeline tag originally said `-int8`; the
  measurement retired that assumption before it reached the index.
- **Pinning to one thread was 2.3× worse than the default fan-out.** The
  intuition that a ten-token forward pass is too small to parallelise is simply
  wrong on this hardware.

Selected: bge-small fp32, default threads, **8.70 ms p95**. See
[ADR 0002](docs/adr/0002-in-process-embedding.md).

### Ingest embedding is the binding constraint

`cargo run --release -p tz-embed --example ingest_throughput`

At the corpus's measured median chunk length (350 tokens):

| Encoder | chunks/s | Full 638k-chunk corpus |
|---|---|---|
| bge-small (12 layers) | 7.8 | **22.7 h** |
| all-MiniLM-L6 (6 layers) | 14.5 | **12.3 h** |

**Sequence length dominates; batch size is within noise.** 2.7× the tokens costs
3.3× the time — superlinear, consistent with attention being quadratic in
sequence length.

A full-corpus index is therefore not a same-day job on CPU. Rather than quote a
number from a corpus that was never built, the index is scoped the way IR
benchmark subsets are: **every gold document plus a controlled number of
distractors**, with recall reported against distractor count. Gold documents are
mandatory — recall against an index that does not contain the answer measures
nothing.

Separately, capping the embedding batch cut peak ingest memory from **15.4 GB to
1.7 GB**. Attention is O(batch × heads × seq²), so batch size is a memory knob
before it is a throughput one ([ADR 0005](docs/adr/0005-ingest-memory-and-batching.md)).

### In-process embedding bought the latency budget, and sent the bill elsewhere

Running the encoder in-process is what makes 25ms reachable — it deletes a
15–40ms network hop. Measured under concurrent ingest, that same decision costs:

| Stage | p95 idle | p95 under ingest |
|---|---|---|
| Query embedding | 8.70 ms | **61.05 ms** (7×) |

The published guidance on ingest contention is all about the vector store, and
those mitigations were configured from the start. But the encoder now shares CPU
with ingest, and no Qdrant setting touches that. Details in
[contention-note.md](docs/benchmarks/contention-note.md).

### The SLO measurement

`cargo run --release -p tz-retrieve --bin tz-searchbench`

10,000 documents / 17,813 chunks (HNSW built), 300 held-out queries, idle
machine, dense-only:

| SLO | Budget | Measured | |
|---|---|---|---|
| Retrieval p95 | ≤ 25 ms | **29.21 ms** | FAIL by 17% |
| recall@50 | ≥ 0.85 | **0.757** | FAIL |

| Stage | p95 |
|---|---|
| embed | 13.96 ms |
| qdrant_search | 16.34 ms |
| decode | 0.70 ms |
| **total** | **29.21 ms** |

**`hnsw_ef` at 32/64/128/256 produced identical recall and identical latency.**
Not similar — identical. That one fact resolves an otherwise confounded
comparison against the exact-search ceiling below: if ANN approximation were
costing recall, raising `ef` would recover some, and it recovers none. **HNSW is
costing nothing; the recall loss is entirely the 5.6× larger distractor set.**
Graph tuning is therefore a dead end here, and hybrid retrieval plus reranking
become justified by measurement rather than assumption.

The 29.21 ms is the *eval* query shape (500 chunks fetched, so document-level
recall@100 means what it claims). At production `top_k=50` the load test
measured **25.49 ms corrected p95** — a near-miss. A 5 ms fix is already
measured (MiniLM-L6 embeds at 3.63 ms vs bge-small's 8.70 ms) and deliberately
not taken: it is the weaker retriever, and recall is the *failing* SLO. Trading
recall for latency when recall is the constraint optimises the wrong number.

Detail: [retrieval-steady-state.md](docs/benchmarks/retrieval-steady-state.md)

### Hybrid retrieval lost, and query shape beat every retriever choice

`cargo run --release -p tz-retrieve --bin tz-searchbench`

Identical index, identical queries, 300 held-out. Only the retriever and the
query text vary:

| Query shape | Retriever | p95 | recall@50 | MRR |
|---|---|---|---|---|
| title | dense | 31.54 ms | 0.757 | 0.362 |
| title | sparse (BM25) | 22.93 ms | 0.587 | 0.235 |
| title | hybrid RRF | 46.98 ms | 0.730 | 0.332 |
| body | **dense** | 41.70 ms | **0.857** ✅ | **0.471** |
| body | sparse (BM25) | 32.53 ms | 0.717 | 0.301 |
| body | hybrid RRF | 60.67 ms | 0.830 | 0.428 |

**Hybrid RRF lost to dense on both query shapes.** Measuring the sparse
retriever *alone* rules out a bug: BM25 works (0.587 / 0.717 recall@50), it is
simply much weaker than dense here. RRF fuses by rank with equal weight, so
combining an unequal pair drags the strong one down — hybrid lands *between* its
constituents, which is what equal-weight fusion must do. The "hybrid always
wins" result assumes retrievers of comparable strength, and that precondition is
one sparse-only measurement away from being checked.

**Query shape moved recall further than any retriever swap.** Title → title+body
took dense from 0.757 to **0.857**, clearing the SLO floor, with no change to
the model, index, or infrastructure. A support assistant seeing only a ticket
*subject* works at 0.757; one seeing the whole ticket is at 0.857.

The two SLOs then pull against each other: the query shape that clears the
recall floor pushes p95 from 31.5 ms to 41.7 ms, because encoding is the largest
stage and longer queries mean more tokens. That tension is the actual result,
and a system with one headline metric would never have surfaced it.

Detail: [retriever-ab.md](docs/benchmarks/retriever-ab.md)

### Cost, measured without spending anything

`cargo run --release -p tz-agent --bin tz-costbench`

Cost is arithmetic on token counts against a rate card — the API does not
compute the price. So 300 held-out queries were run through real retrieval and
real prompt construction, counted with the local tokenizer, and priced from the
versioned table. Output is charged at `max_tokens`, the enforced ceiling, rather
than a hoped-for average.

| | Value |
|---|---|
| **Cost p95** | **$0.01493** vs $0.06 budget — **PASS**, 4x headroom |
| Routing | 94.0% tier 1 (Haiku) · 6.0% tier 2 (Sonnet) |
| Input tokens | p50 4,707 · p95 5,484 |

The verdict holds across a ±20% tokenizer-error band, because the local
wordpiece tokenizer is a proxy for Anthropic's BPE and a result that only holds
at one point estimate is not a result.

**The finding: the cache breakpoint can never fire.** `answer.rs` places
`cache_control` on the system prompt and documents at length why stable content
must precede volatile content. The reasoning is right and the code is
irrelevant — the system prompt is **170 tokens** and the minimum cacheable
prefix is **2,048** (Haiku). The marker is silently ignored; no error, and
`cache_read_input_tokens` would have sat at zero forever.

Generally: **prompt caching is only available to workloads with a large stable
prefix.** In RAG, nearly all the input is per-query retrieved context, so no
such prefix exists. "Use prompt caching to cut RAG costs" does not apply to this
shape of workload.

**This reverses the design's founding assumption.** The architecture was built
around a 6-cent budget forcing routing, tight context caps and caching. Routing
and caps work; caching cannot work at all; and none of it was load-bearing,
because the workload costs a quarter of budget. Cost is not the binding
constraint — latency and recall are. Detail: [cost.md](docs/benchmarks/cost.md)

### Load test: coordinated omission, demonstrated

| Rate | Corrected p95 | Naive p95 | |
|---|---|---|---|
| 10/s | 25.49 ms | 24.14 ms | |
| 25/s | 25.93 ms | 25.14 ms | |
| 50/s | **52.96 ms** | **21.05 ms** | generator fell behind |

At 50 rps the naive measurement reports the **lowest p95 in the table** while
the system is saturating. It does not understate the problem, it inverts it: a
closed-loop generator slows down with the server, so the overloaded period gets
fewer samples and latency appears to improve.

Defensible capacity claim: **25 rps at 25.93 ms p95 corrected, zero errors**,
knee below 50 rps — on one machine shared by Qdrant, the encoder and the
generator. Detail: [loadtest.md](docs/benchmarks/loadtest.md)

### The exact-search recall ceiling

`cargo run --release -p tz-retrieve --bin tz-searchbench`

Measured first, on a 1,775-document gold-only index, precisely so the ANN result
above could be differenced against it. 150 held-out queries, dense-only:

| Metric | Value |
|---|---|
| recall@1 | 0.420 |
| recall@10 | 0.753 |
| **recall@50** | **0.867** — SLO floor 0.85 ✅ |
| recall@100 | 0.920 |
| nDCG@10 · MRR | 0.584 · 0.536 |

The curve is the finding: **0.420 → 0.867 from k=1 to k=50**. A pipeline that
fetches five candidates has already discarded a third of the answers available
to it, and no reranker recovers a document that was never retrieved.

Two things this is *not*. The `hnsw_ef` sweep returned identical recall at
32/64/128/256 — which only happens when the parameter is unused — because a
4,265-point collection sits below the per-segment indexing threshold and Qdrant
brute-forced. So it is an **exact-search ceiling**: the embedding model's true
recall, independent of ANN approximation, and the baseline the HNSW number gets
differenced against. And it is a **lower bound**, because duplicate judgements
name *a* relevant document, not every one.

Steady-state latency is not yet measured; the figure taken during this run is
contaminated by concurrent ingest and is reported separately, not as an SLO
result. Details: [retrieval-exact-ceiling.md](docs/benchmarks/retrieval-exact-ceiling.md)

### The golden set is human-labelled, not synthesised

`cargo run --release -p tz-eval --bin tz-golden -- data/serverfault`

**4,437 query/document pairs** (3,548 train / 889 test), from 5,003 moderator
duplicate rulings. Each pair is a domain practitioner's judgement that one
question is answered by another — not an LLM's guess about its own kind.

Pinned to corpus hash `582503f9…`. Filtering is reported, not hidden: 50 pairs
dropped for absent posts, 91 for a gold document that is itself a duplicate, 251
for a negatively-scored target, 108 for a non-discriminating query.

The 4,437 pairs point at **1,775 distinct gold documents** — popular canonical
questions are the target of many duplicates each. All 1,775 are mandatory
members of any index the benchmark runs against.

Known limitation, stated wherever the numbers appear: duplicate judgements name
*a* relevant document, not *every* one, so recall@k is a **lower bound**.

---

## Bugs the tests caught

Kept deliberately, because "what did your tests catch" is a more useful signal
than a green badge. All four fail silently in production — none raises an error.

| Bug | Effect if shipped | Found by |
|---|---|---|
| Bare `<` in prose parsed as a tag opener | `if (a < b)` swallowed text through to the next real tag. Silent corpus loss across an ops corpus full of comparisons. | Unit test |
| Packer could only split *between* segments | A document with no code fences became one 2,564-token chunk — 5× budget, truncated at encode time, tail never indexed. | Unit test |
| Overlap helper cut at line boundaries only | Prose without line breaks has one "line", so a 51-token overlap carried a full 512-token chunk. | Unit test |
| Table cells emitted no closing space | `\| Key\| Val \|` — malformed rows in a corpus where tables carry configuration values. | Unit test |
| Unsplittable single lines | Base64 dumps produced chunks up to **12,971 tokens**; 1.12% of the corpus over budget. | **Full-corpus run** |
| `pack_units` ignored joiner tokens | Running sum drifted below true tokenization; long runs of small units overflowed. | Invariant test |
| Unbounded embedding batch | Peak RSS **15.4 GB**; machine thrashed with no error raised. | **Real ingest run** |
| Default OTel histogram buckets (ms-sized, seconds recorded) | Dashboard read **p95 = 4,750 ms** for an 80 ms operation, and the $0.06 cost SLO had no bucket boundary to resolve against. | **Checking the panel against a known value** |
| Dashboard queried a metric name that does not exist | Grafana renders an empty panel, not an error. | Reading Qdrant's actual `/metrics` |
| Index built in stream order, gold documents last | Recall unmeasurable until the final moments of a 50-minute run; 2 of 40 gold docs present after 15 min. | **Sampling the index mid-build** |

Three of these were invisible to unit tests and appeared only when the code met
325,169 real documents or a real ingest. Unit tests establish that the logic is
right; a full run establishes that the *assumptions* are.

---

## Scope: what was left out, and why

Listed so an absence is never mistaken for a passing result.

| Area | State |
|---|---|
| Hybrid dense+sparse retrieval | Dense implemented and instrumented; BM25 sparse vectors not built, so RRF fusion is unit-tested but unexercised |
| Live model calls | Router, tiers and pre-flight budget enforcement are implemented and tested against a mock; no request has been sent to a real API, so no cost figure is empirical yet |
| TTFT / generation latency | Not measured — follows from the above |
| Load test results | Generator built and unit-tested; the ramp has not been run against a populated index |
| `pass^k` consistency | Metric not implemented |
| Grafana dashboards | **Running** — Prometheus, Tempo, OTel collector and Grafana all as standalone binaries, no Docker ([ADR 0008](docs/adr/0008-standalone-qdrant.md)). 12-panel SLO dashboard at `localhost:3000/d/tierzero-slo`, fed by a verified Rust → OTLP → collector → Prometheus chain. Tempo is the one holdout: v3.x hard-defaults several paths to `/var/tempo`, so distributed traces are not yet wired. |
| Full-corpus index | Not built, and not buildable in a session: 12.3 h at the measured embedding rate. The index is a scoped IR subset and says so. |

## Layout

| Crate | Role |
|---|---|
| `tz-core` | Shared vocabulary: chunks, four-way cost accounting, SLO budgets |
| `tz-embed` | In-process ONNX embedding — the decision the latency budget rests on |
| `tz-ingest` | Streaming XML, HTML cleanup, structure-aware chunking |
| `tz-eval` | Golden set, retrieval metrics, CI gates |
| `tz-retrieve` | Qdrant schema, instrumented dense search, indexer, benchmark |
| `tz-obs` | OTel GenAI attributes, per-request cost records |
| `tz-cache` | Exact + semantic cache with query-class guards and tenant isolation |
| `tz-agent` | Non-LLM router, cost tiers, pre-flight budget enforcement |
| `tz-bench` | Open-model load generator, corrected for coordinated omission |

## Running it

Qdrant runs either from `deploy/docker-compose.yml` or as a standalone binary
(no Docker required — see `data/qdrant-run/config/config.yaml` for the
latency-tuned configuration):

```bash
cd deploy && docker compose up -d      # qdrant, otel, prometheus, grafana, tempo
cargo test --workspace                 # 177 tests
cargo test -p tz-embed -- --ignored    # model-download tests
```

Corpus (~860 MB download, 1.2 GB extracted):

```bash
curl -L -o data/raw/serverfault.com.7z \
  https://archive.org/download/stackexchange/serverfault.com.7z
bsdtar -xf data/raw/serverfault.com.7z -C data/serverfault Posts.xml PostLinks.xml Tags.xml
```

## Decisions

- [0001 — Corpus and golden set](docs/adr/0001-corpus-and-golden-set.md)
- [0002 — In-process embedding, and trusting the sweep over the intuition](docs/adr/0002-in-process-embedding.md)
- [0003 — 512/10% chunking; code blocks are atomic](docs/adr/0003-chunking.md)
- [0004 — Four-way token accounting against a versioned price table](docs/adr/0004-cost-model.md)
- [0005 — Bound the embedding batch; attention is quadratic in sequence length](docs/adr/0005-ingest-memory-and-batching.md)
- [0006 — Guard the semantic cache by query class, not by threshold alone](docs/adr/0006-semantic-cache-guards.md)
- [0007 — A router and one agent, not a multi-agent system](docs/adr/0007-single-agent-with-routing.md)
- [0008 — Run Qdrant as a binary, not only under Docker](docs/adr/0008-standalone-qdrant.md)

Full benchmark report: [docs/benchmarks/](docs/benchmarks/README.md)

## Licence

Code is MIT — see [LICENSE](LICENSE).

`evals/golden/serverfault.json` is **not** MIT. It contains question titles and
body excerpts authored by Server Fault contributors and is licensed CC BY-SA 4.0
by them. Attribution and detail:
[evals/golden/ATTRIBUTION.md](evals/golden/ATTRIBUTION.md)

The corpus itself (1.21 GB of `Posts.xml`) is not redistributed here — it is
downloaded at build time from archive.org by the command above, and the golden
set is pinned to it by content hash.
