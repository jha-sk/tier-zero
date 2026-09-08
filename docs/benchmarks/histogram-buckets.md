# Default histogram buckets made the SLO unmeasurable

**Found 2026-09-07, while verifying the metrics pipeline.**

The retrieval path was instrumented, metrics reached Prometheus, the dashboard
rendered, and the p95 panel read **4,750 ms** for an operation measured
in-process at **~80 ms**.

## Cause

OpenTelemetry's default explicit-bucket boundaries are:

```
[0, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]
```

Those are sized for **milliseconds**. The instrument records **seconds**, per
the OTel semantic conventions, which put every real value into the first
bucket — the range 0 to 5. `histogram_quantile` then interpolates within that
bucket and returns its upper reaches. 4,750 ms was not noise or a scrape
artefact; it was the honest midpoint of a range containing every sample.

The same flaw applied to cost. With no boundary between $0 and $5, a $0.06
budget could not be evaluated at all.

## Why it is worth writing down

Every individual piece was correct. The units followed the conventions, the
exporter worked, the collector received the data, Prometheus stored it, the
dashboard queried it. The composition was wrong, and it failed by **rendering a
plausible number** rather than an error or an empty panel.

A p95 of 4,750 ms is not obviously absurd for an LLM pipeline. On a dashboard
next to other panels it would have been read as a real result and acted on —
someone would have optimised a system that was already 60x faster than the
chart claimed.

## Fix

Explicit boundaries, dense around the budgets being measured:

```rust
// seconds
[0.001, 0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040,
 0.050, 0.075, 0.100, 0.250, 0.500, 1.0, 2.5, 5.0, 10.0]

// USD
[0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.010, 0.020, 0.030, 0.040, 0.050,
 0.060, 0.070, 0.080, 0.100, 0.250, 0.500, 1.0]
```

Both include a boundary **at** the budget (0.025 s, $0.060), because a
percentile is only as precise as the bucket it falls in and an SLO measured
through a bucket wider than the budget cannot be evaluated.

Verified from the raw bucket counts rather than from the rendered panel:

| le (s) | cumulative count |
|---|---|
| 0.040 | 0 |
| 0.050 | 2 |
| 0.075 | 54 |
| 0.100 | 93 |
| 0.250 | 121 |

The distribution is now resolved where it matters. Tests assert that both bucket
sets keep a boundary at their budget and stay dense around it, so a future edit
cannot quietly coarsen them.

## The general form

**Instrumentation can be individually correct at every step and wrong as a
composition, and it fails by producing a believable number.** Checking a metric
against a known ground truth — here, the in-process histogram the same binary
already prints — is part of building the dashboard, not an optional follow-up.
