# ADR 0003: 512 tokens with 10% overlap; code blocks are atomic

**Status:** Accepted · **Date:** 2026-09-07

## Context

Chunking is the decision most often made by copying a tutorial default. The
default in question — 512-character or 800-token chunks with 400 overlap — is
widely reproduced and, on the published evidence, close to the worst available
choice.

## Decision

Chunk at **512 tokens with ~10% overlap (51 tokens)**, split on document
structure, and treat fenced code blocks as **atomic units that are never split**
unless a single block exceeds the budget on its own.

Size chunks with the **embedding model's own tokenizer**, not a character
heuristic.

## Rationale

**Against 800/400.** In Chroma's chunking evaluation, the 800/400 recursive
configuration scored worst of every configuration tested on precision, on
Precision_Ω, and on IoU, and was not competitive on recall either. A 50% overlap
means the same sentence appears in several neighbouring chunks, which inflates
index size and token cost while crowding the candidate list with near-copies of
itself. Precision falls for a recall gain that does not arrive.

**Against semantic chunking.** The peer-reviewed result (Findings of NAACL 2025,
Vectara + UW-Madison) is that fixed-size chunking matched or beat semantic
chunking on realistic document sets, with the apparent gains confined to
artificially stitched multi-topic benchmarks. Semantic chunking costs an
embedding pass or an LLM call per document. We do not pay it. This is recorded
as a *prediction*, and the benchmark table keeps a row for it: if measurement on
this corpus contradicts the literature, the row reports that.

**Why chunk size is not a universal constant.** Published multi-dataset work
shows the optimum swinging by 3.7× on a single dataset purely as a function of
chunk size, in opposite directions for short-factoid versus long-narrative
corpora. 512 is a starting point to be tuned against the golden set, not a
finding.

**Why code blocks are atomic.** In an operations corpus the command *is* the
answer. A configuration snippet split across a boundary yields two fragments:
one truncated mid-solution, and one with no indication of what it belongs to.
Both are retrievable and both are wrong. Blocks that individually exceed the
budget are split at **line** boundaries and flagged `split_code`, so the claim
can be reported with its exceptions rather than merely asserted. Overlap is
never carried across a code boundary, which would duplicate part of a command
into a neighbouring chunk.

**Why the model's tokenizer.** Tokens-per-character is far higher for code than
prose. A chunk sized in characters overflows the encoder window on code-dense
text, and the failure is silent: the encoder truncates and the tail of the chunk
is never indexed.

## The invariant

**No chunk may exceed the encoder window (512 tokens).** This is stated as an
invariant rather than a target because an oversized chunk does not fail loudly:
the encoder truncates it, the tail is never indexed, and nothing reports it. The
only symptom is recall that is worse than it should be, with no way to attribute
the loss.

Getting there took five fixes, four of them found by tests and one by running
the chunker over the real corpus. Every one fails silently.

| # | Bug | Effect | Found by |
|---|---|---|---|
| 1 | Packer could only split *between* segments | A document with no code fences was one segment → a single **2,564-token** chunk, 5× budget | Unit test |
| 2 | Overlap helper cut at line boundaries only | Prose with no line breaks has one "line", so a 51-token overlap carried a full 512-token chunk; sizes crept toward double budget | Unit test |
| 3 | `split_code_by_lines` could not split a single over-long line | Base64 dumps and minified configs produced chunks up to **12,971 tokens** — 1.12% of the corpus over budget | **Full-corpus run** |
| 4 | Character budget estimated from a line's *average* token density | Mixed prose/base64 lines undershot the estimate; worst case still 2.1× budget | Unit test on mixed-density input |
| 5 | `pack_units` did not count joiner tokens | Running sum drifted one-directionally below true tokenization; long runs of small units overflowed | Invariant test |

Bug 3 is the one worth dwelling on. It was invisible to every synthetic test and
appeared only when the chunker met 325,169 real documents. Unit tests establish
that the logic is right; a full-corpus run establishes that the *assumptions*
are. Both are necessary.

The final defence is not an estimate. Chunk size is verified against the real
tokenization at flush, and anything over budget is re-split rather than emitted
— because a tokenizer merges and splits across boundaries, so accounting alone
is necessary but not sufficient.

## Measured outcome

Full corpus (325,169 documents → **638,156 chunks**):

| Metric | Value |
|---|---|
| Chunk tokens | p50 353 · p95 493 · p99 509 · **max 512** |
| Over budget | **0 (0.000%)** |
| Under 50 tokens (fragment signature) | 14,568 (2.28%) |
| **Code chunks** | 360,862 (**56.5%**) |
| Chunks per document | p50 1 · p95 5 · max 64 |
| Resident index (int8 + HNSW graph, 384 dims) | **0.33 GB** |

56.5% of chunks contain code. That single number is what makes the
atomic-code-block rule load-bearing rather than decorative: on this corpus, a
chunker that splits code carelessly damages the majority of the index.

Throughput: 182s for the full corpus (1,786 docs/s, 3,504 chunks/s) on 12 cores.
The first implementation took 937s serially; tokenization dominates and is
per-document independent, so documents are assembled serially from the stream
and chunked in parallel batches.
