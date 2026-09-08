# Cost SLO: measured without spending anything

**Measured 2026-09-07.** 300 held-out queries, real retrieval, real prompts,
priced against `config/prices.toml` version `2026-09-06.1`.

## Method, and why it needs no API calls

Cost is arithmetic on token counts against a rate card. The API does not compute
the price — the rate card does — so the only thing a live call contributes is
the token count, and prompts can be counted locally. What this measures is what
the **architecture** controls: how many tokens the retrieval and routing
decisions put in front of a model.

Two limits, stated rather than buried:

1. **Tokenizer proxy.** Counting uses the cached BERT-wordpiece tokenizer from
   the embedding model, not Anthropic's BPE tokenizer. Wordpiece splits more
   aggressively on code-dense text, so on this corpus the estimate runs *high* —
   a conservative error, which is the right direction for a budget. A
   sensitivity band is reported instead of a single point.
2. **Output length is bounded, not predicted.** Only the model decides how much
   it writes, so cost is charged against `max_tokens = 900`, the enforced
   ceiling. That is worst case, which is what a p95 budget is about.

## Result

| | Value |
|---|---|
| **Cost p95** | **$0.01493** |
| Budget | $0.06 |
| **Verdict** | **PASS, 4x headroom** |

| Percentile | Cost |
|---|---|
| p50 | $0.00927 |
| p95 | $0.01493 |
| p99 | $0.01594 |
| max | $0.01642 |

Routing: 94.0% tier 1 (Haiku 4.5), 6.0% tier 2 (Sonnet 5), 0% tier 0.

| Tier | p50 | p95 |
|---|---|---|
| tier 1 | $0.00924 | $0.00998 |
| tier 2 | $0.01554 | $0.01625 |

Input tokens per request: p50 4,707 · p95 5,484 · p99 5,782.

The verdict survives the tokenizer uncertainty:

| Token count error | p95 | |
|---|---|---|
| −20% | $0.01194 | PASS |
| −10% | $0.01344 | PASS |
| 0% | $0.01493 | PASS |
| +10% | $0.01642 | PASS |
| +20% | $0.01791 | PASS |

A result that only holds at one point estimate is not a result. This one holds
across the whole plausible band.

## The finding: the cache breakpoint can never fire

`answer.rs` places a `cache_control` breakpoint on the system prompt, and its
module documentation explains at length why stable content must precede volatile
content for prefix caching to work. That reasoning is correct and the
implementation is irrelevant, because **the system prompt is 170 tokens and the
minimum cacheable prefix is 2,048 (Haiku) or 1,024 (Sonnet)**.

The marker is **silently ignored**. No error, no warning; `cache_read_input_tokens`
would simply have stayed at zero forever, and the caching section of the cost
dashboard would have read 0% with nothing indicating why.

The general form, which outlives this project: **prompt caching is only
available to workloads with a large stable prefix.** For RAG where nearly all
the input is per-query retrieved context, there is no such prefix — the stable
part is a short system prompt, and the large part changes every request. The
common advice to "use prompt caching to cut RAG costs" does not apply to this
shape of workload at all.

Options, none adopted because cost is passing with 4x headroom and none of them
would be motivated by a real constraint:

- Pad the system prompt to the minimum with genuinely useful content — few-shot
  examples, a domain glossary, an escalation policy. Only worth it if the
  content earns its tokens on quality grounds; padding purely to unlock caching
  buys a discount on tokens that did not need to exist.
- Cache a stable *corpus* prefix, if the same documents recur across requests.
  Retrieval here is per-query, so they do not.
- Accept no caching. This is the current state, and it is the right one.

## What this changes

Cost is **not** the binding constraint on this system. It passes at a quarter of
budget while latency misses by 17-67% and recall sits at the floor. The
architecture has room to spend more per request — a stronger model, more
context, a reranking pass — if that buys quality.

That reverses the assumption the design started from. The original reasoning was
that a 6-cent budget forces routing, tight context caps and aggressive caching.
The routing and caps are built and they work; the caching cannot work at all;
and it turns out none of it was load-bearing, because the workload is far
cheaper than budgeted. **Worth knowing before optimising a number that was never
in danger.**
