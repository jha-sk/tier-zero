# ADR 0006: Guard the semantic cache by query class, not by threshold alone

**Status:** Accepted · **Date:** 2026-09-07

## Context

A per-request cost SLO of $0.06 is much easier to hold if a meaningful share of
traffic never reaches a model. Caching is the obvious lever. Exact-match caching
is free and cannot be wrong. Semantic caching — serving a stored answer when a
new query is *similar enough* to an old one — has a far higher hit rate and a
failure mode that exact matching does not have.

Reported production hit rates for semantic caching sit around 20–45%, against
vendor claims near 95%. The gap is not the interesting part. The interesting
part is what the hits cost when they are wrong.

## The problem with tuning a threshold

Semantic caching rests on an equivalence that does not hold: it treats
*embedding proximity* as *answer interchangeability*. Those are different
relations, and the cases where they diverge are systematic rather than random:

| Pair | Similarity | Same answer? |
|---|---|---|
| "disk usage in Q1 **2024**" / "…Q1 **2025**" | > 0.95 | No |
| "why did latency **increase**" / "…**decrease**" | > 0.95 | No, opposite |
| "**how many** servers" / "**which** servers" | > 0.95 | No, different shape |

Embedding models place these close together because they *are* topically close.
That is the model working correctly. No threshold separates them, because the
discriminating token — a year, a negation, a question word — contributes little
to a pooled sentence embedding by design.

Lowering the threshold raises the hit rate and the error rate together. Raising
it does the reverse. There is no setting at which this class of error goes away,
which means threshold tuning is the wrong control.

## Decision

**Exclude query classes where embeddings are known to under-weight the
discriminator, before similarity is ever computed.** A query containing a
temporal reference, a negation, a comparative, or a version/size/port number is
refused semantic lookup entirely.

Three properties follow:

1. **Guarded queries still hit the exact cache.** Guards restrict *similarity*
   matching; equality is still equality. Refusing an exact repeat of
   "disk usage in Q1 2024" would throw away a free, correct hit.
2. **The threshold can then be strict.** 0.95, against the 0.80–0.85 commonly
   suggested. Once the systematically dangerous classes are removed, the
   remaining traffic can afford a high bar — and the cost of a wrong ops
   instruction is not symmetric with the cost of a miss.
3. **Refusals are counted separately from misses.** They are different events
   with different fixes: a high refusal rate means the guards are too broad, a
   high miss rate means the cache is too small. Collapsing them into one number
   makes both invisible.

Guards are conservative by construction: a false refusal costs one model call,
a false acceptance returns a confidently wrong answer to an operator.

## Tenant isolation

Every entry is namespaced by tenant, and the semantic scan iterates only the
requesting tenant's entries. A cross-tenant cache hit is a data breach, and it
is the kind that leaves **no trace in the vector store's access logs**, because
on a cache hit the store is never consulted. A test asserts that an identical
query with an identical embedding from a different tenant misses.

## Consequences

The false-hit rate becomes an SLI (≤ 1%), sitting next to the cost SLO rather
than beneath it. Every semantic hit records its similarity score and the query
it matched, because that score is the only evidence available after a wrong
answer has been served.

The honest limitation: guards are regex-based and therefore catch *lexical*
signals of the dangerous classes. A query whose time-dependence is implicit
("is the certificate still valid") carries no temporal token and will pass. The
guards reduce a systematic error class; they do not eliminate it, and the
sampled review of scored hits exists because of exactly that gap.
