# ADR 0002: Embed in-process, and trust the sweep over the intuition

**Status:** Accepted · **Date:** 2026-09-07

## Context

The retrieval SLO is a p95 of 25ms covering query embedding, ANN search, fusion
and payload fetch. Query embedding is the first thing in that budget and the
easiest to get wrong.

## Decision

Run the encoder in-process via ONNX Runtime. Use `bge-small-en-v1.5` at fp32,
384 dimensions, with ONNX Runtime's **default** thread configuration.

## Rationale

**Why not a hosted embedding API.** A TLS round trip to a hosted endpoint costs
15–40ms before any compute. Against a 25ms budget that is not a tax, it is a
wall. This single decision is what makes the SLO reachable, and it is worth
being explicit that it is a *constraint-driven* choice: we give up access to
stronger hosted models and accept an open 384-dimension encoder.

**Why the defaults, measured rather than assumed.** The first honest measurement
of the naive configuration was **11.51ms p95** — 46% of the entire budget for a
33M-parameter model on a ten-token input. Rather than accept it, we swept
encoder, thread count and sequence length
(`crates/tz-embed/examples/encoder_sweep.rs`, 300 sampled embeds per row after
50 warm-up). Three results, two of which contradicted the prior:

| Lever | Expected | Measured |
|---|---|---|
| INT8 quantization | ~3.5× faster | **2× slower** — 15.97ms vs 8.90ms p95 |
| Pinning to 1 thread | Faster; a 15-token matmul is too small to parallelise | **Wrong** — 19.60ms vs 8.70ms for the default fan-out |
| Capping max_length 512 → 64 | Meaningful win | Negligible |

*On INT8.* Dynamically-quantized ONNX inserts Quantize/Dequantize pairs around
each matmul. Where those do not fuse into a native INT8 GEMM, the conversion
cost lands on top of the arithmetic rather than replacing it. The project's own
pipeline tag originally read `bge-small-en-v1.5-int8`; the measurement retired
that assumption before it reached the index.

*On threads.* The intuition that a tiny forward pass should not be parallelised
is wrong here by more than a factor of two. Recorded because it is exactly the
kind of plausible reasoning that survives review and loses to a stopwatch.

*On max_length.* Reading the tokenizer configuration explained the null result:
padding is `BatchLongest`, not padding-to-max, so a short query was never being
padded to 512 in the first place. The lever had nothing to pull.

**Selected configuration:** `bge-small-en-v1.5`, fp32, default threads →
**8.70ms p95**, inside the ~9ms this stage was allocated.

## Consequences

`all-MiniLM-L6-v2` measured considerably faster (**3.63ms p95**, 2.4× better)
and is the fallback if the rest of the budget proves tight. It is not the
default because it is the weaker retriever on published benchmarks, and this
project does not trade recall for latency without measuring the trade. That
comparison is a golden-set experiment, not a judgement call, and it is deferred
until the retrieval path exists to run it against.

These numbers are hardware-specific (12-core x86_64). Re-run the sweep on
different hardware; do not port the constants by intuition — that is precisely
what failed above.
