# ADR 0005: Bound the embedding batch; attention is quadratic in sequence length

**Status:** Accepted · **Date:** 2026-09-07

## Context

The indexer assembles documents, chunks them, embeds the chunks, and upserts
them. The natural implementation hands each assembled batch to the encoder in
one call and lets the library decide how to schedule it.

## What happened

On the first real ingest run, resident memory reached **15.4 GB** and the
machine began thrashing. No error was raised; the process simply stopped making
progress, and from the outside it was indistinguishable from a slow job.

## Cause

Transformer attention is **O(batch × heads × seq²)**. Batch size and sequence
length are not independent knobs — they multiply, and the sequence term is
squared. At a batch of 256 with 512-token chunks, the attention tensors alone
run to gigabytes *per layer*.

This is invisible at small scale. Unit tests embed a handful of short strings
and the working set is megabytes. The failure needs both a real batch and real
chunk lengths to appear, which means it only shows up on a full ingest run.

## Decision

Batch size is an explicit field on `EmbedderConfig`, documented as a **memory**
knob before a throughput one, and the embedder sub-batches internally rather
than passing the caller's batch through:

| Path | Batch | Why |
|---|---|---|
| Query (`for_queries`) | 1 | One query at a time by definition |
| Ingest (`for_ingest`) | 32 | Bounded working set at 512-token sequences |

Length bucketing (sorting by length before batching) does double duty here: it
keeps padding waste low *and* it bounds peak memory to the longest sequence in
the **current sub-batch** rather than in the whole call.

**Result: peak RSS 15.4 GB → 1.7 GB, a 9× reduction**, with no change in output.

## Consequences

A regression test asserts the ingest batch stays within a sane range and that
the query path uses a batch of one. It cannot catch the underlying quadratic —
that needs a real run — but it prevents the value being raised casually by
someone chasing throughput without knowing what it trades against.

The general lesson, which generalises past this project: **library defaults are
tuned for throughput on the author's hardware, not for the memory envelope of
your workload.** Two of this project's three performance findings so far have
been defaults that were wrong here — ONNX Runtime's thread count (ADR 0002) and
this one — in opposite directions. Neither was discoverable by reading the code.
