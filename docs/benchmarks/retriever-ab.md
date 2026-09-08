# Hybrid retrieval made things worse, and query shape mattered more

**Measured 2026-09-07.** 10,000 documents / 17,813 chunks, 300 held-out queries,
idle machine. Only the retriever and the query text vary; the index is identical
throughout.

| Query shape | Retriever | p95 | recall@10 | recall@50 | recall@100 | MRR |
|---|---|---|---|---|---|---|
| title | dense | 31.54 | 0.560 | 0.757 | 0.820 | 0.362 |
| title | sparse (BM25) | 22.93 | 0.430 | 0.587 | 0.663 | 0.235 |
| title | hybrid RRF | 46.98 | 0.543 | 0.730 | 0.810 | 0.332 |
| body | **dense** | 41.70 | **0.673** | **0.857** | **0.907** | **0.471** |
| body | sparse (BM25) | 32.53 | 0.540 | 0.717 | 0.793 | 0.301 |
| body | hybrid RRF | 60.67 | 0.663 | 0.830 | 0.893 | 0.428 |

## Finding 1: hybrid RRF lost to dense, on both query shapes

This contradicts the usual expectation, so the first job was to rule out a bug.
Measuring the sparse retriever **alone** settles it: BM25 works — 0.587
recall@50 on titles, 0.717 on bodies, which is respectable for pure lexical
matching. It is simply **much weaker than dense on this corpus**.

Reciprocal rank fusion combines by rank with equal weight. Fusing a strong
retriever with a weak one therefore drags the strong one down, and hybrid lands
*between* its two constituents rather than above them — 0.730 sits between
sparse's 0.587 and dense's 0.757. That is not a malfunction; it is what
equal-weight rank fusion does when the retrievers are unequal.

The "hybrid always wins" result in the literature comes from settings where BM25
and dense retrieval are **comparably strong**. That precondition does not hold
here, and it is cheap to check before adopting the technique — one sparse-only
measurement.

Hybrid also cost ~15ms of p95 for the privilege, because both prefetches must
complete before fusion.

**Remedies, none adopted, all now motivated by measurement rather than
assumption:** weight the fusion toward dense (Qdrant supports DBSF); route to
sparse only for queries containing rare tokens, making it a precision tool for
identifier lookup rather than a blanket second opinion; or improve the sparse
side with better tokenization and stemming until the retrievers are comparable
enough for equal-weight fusion to be appropriate.

## Finding 2: query shape beat every retriever choice

Going from a title-only query to title + body:

| | recall@50 | change |
|---|---|---|
| Title only, dense | 0.757 | — |
| **Title + body, dense** | **0.857** | **+0.100** |

That single change moved recall further than any retriever swap in the table,
and it **clears the 0.85 SLO floor**. A title is 5–10 words; a body carries the
error strings, commands, versions and symptoms that make a question findable.

The system-design consequence: a support assistant that only sees a ticket
*subject line* is working at 0.757 recall, and one that sees the whole ticket is
at 0.857 — for free, with no model, index or infrastructure change. Before
tuning a retriever, check what you are handing it.

It also partly vindicates and partly refutes the hypothesis behind BM25's
weakness. Lexical signal *does* live in bodies — sparse gained +0.130 recall@50
from the richer query, more than dense gained in relative terms. But dense
gained +0.100 in absolute terms from the same change, so the gap did not close
and hybrid still lost.

## Cost of the richer query

| Query shape | embed p95 | total p95 |
|---|---|---|
| title | ~14 ms | 31.54 ms |
| body | ~24 ms | 41.70 ms |

Longer queries mean more tokens to encode, and encoding is already the largest
stage in the budget. So the two SLOs pull against each other here: the query
shape that clears the recall floor pushes p95 from 31.5 ms to 41.7 ms against a
25 ms budget.

That tension is the real result. It is not resolvable by tuning either number in
isolation, and it is exactly the kind of trade a system with only one headline
metric would never have surfaced.
