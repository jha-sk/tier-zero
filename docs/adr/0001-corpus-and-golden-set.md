# ADR 0001: Stack Exchange as corpus, duplicate links as golden set

**Status:** Accepted · **Date:** 2026-09-07

## Context

The system needs a corpus that is large, genuinely unstructured, plausibly
"business", and legally usable. It also needs a retrieval eval set, and that is
the harder requirement: retrieval quality claims are only as good as the
relevance judgements behind them, and most portfolio projects generate those
judgements with an LLM — the same class of system being evaluated.

## Decision

Use the Stack Exchange data dump (CC BY-SA), Server Fault as the core site,
framed as an internal IT/DevOps support knowledge base.

Build the golden set from `PostLinks` rows with `LinkTypeId=3`. That row means a
moderator read two questions and ruled that one is answered by the other. The
duplicate question becomes the query; the canonical question becomes the
relevant document.

## Consequences

**What this buys.** 4,437 usable query/document pairs after filtering, from
5,003 raw duplicate links. Human-authored, domain-expert judgements, at a scale
no hand-labelling budget would reach, and independent of any model. Vote scores
supply graded relevance for nDCG; tags supply eval slices; edit history supplies
a natural change-data-capture signal for the incremental re-index path.

**What it costs.** Duplicate judgements are one-sided: they name *a* relevant
document, not *every* relevant document. Recall@k is therefore a lower bound —
a retriever may surface a genuinely good answer that the set scores as a miss.
This is stated wherever the numbers are reported rather than quietly ignored.

**Filters applied, and why.** Each is recorded in the build stats so the
filtering is visible:

| Filter | Dropped | Reason |
|---|---|---|
| Referenced post absent from dump | 50 | Cannot score against a document that does not exist |
| Gold document is itself a duplicate | 91 | A closed question is not a canonical retrieval target |
| Gold score < 1 | 251 | A community-downvoted document is an unfair target |
| Query title < 20 chars | 108 | Non-discriminating; measures noise, not retrieval |

The set is pinned to a corpus hash (`582503f9…`). A recall number computed
against a different corpus is a different measurement, and without the pin the
two are indistinguishable in a results table.

The train/test split is by hash of query id, not by random seed, so it is stable
across machines without carrying a seed file.

## Index composition

The 4,437 pairs point at **1,775 distinct gold documents** — popular canonical
questions are the target of many duplicates each.

Because CPU embedding makes a full-corpus index a 12-to-23-hour job (see
[ADR 0005](0005-ingest-memory-and-batching.md) and the throughput benchmark),
the index is built as an IR benchmark subset: **all 1,775 gold documents, plus
distractors up to a budget.** Gold documents are mandatory. Recall measured
against an index that does not contain the answer is not a low score, it is not
a measurement.

**Gold documents are indexed in their own first pass.** The obvious
implementation — one pass admitting gold and distractors together until a budget
fills — was written first and its two faults only appeared once it was running:

1. Documents arrive in ascending post id, so the distractor budget was consumed
   entirely by the **oldest** posts before a meaningful number of gold documents
   was reached. A random sample of 40 gold documents found **2** in the index
   after 15 minutes of embedding.
2. Because gold documents are spread across the whole file, the index would not
   have become measurable until the final moments of a 50-minute run — so a bug
   in the measurement path would surface only at the very end.

Two passes cost one extra sequential scan (~13s, against ~45 minutes of
embedding) and buy a deterministic index composition plus a measurement that is
valid as soon as the first pass completes. The second property is the one that
mattered: it turned a 38-minute feedback loop into an 8-minute one.

Distractors are taken in stream order, which is oldest-first rather than random.
That is a sampling bias and is stated as one; a random sample would be a
one-line change if the bias ever looked load-bearing.
