# Load test: coordinated omission, demonstrated

**Measured 2026-09-07.** Open-model constant-arrival-rate generator against the
10,000-document index, `top_k=50` (production shape), 8s per step, idle machine.

| Rate | Corrected p50/p95/p99 | Naive p95 | Max sched delay | |
|---|---|---|---|---|
| 10/s | 20.96 / **25.49** / 27.44 ms | 24.14 ms | 1.46 ms | |
| 25/s | 20.29 / **25.93** / 28.54 ms | 25.14 ms | 1.48 ms | |
| 50/s | 16.98 / **52.96** / 93.57 ms | **21.05 ms** | 89.86 ms | generator fell behind |

Zero errors at every rate. The knee sits between 25 and 50 rps.

## The 50/s row

At 50 rps the naive measurement reports **21.05 ms p95 — the lowest figure
anywhere in the table**, while the corrected measurement reports 52.96 ms. The
naive number does not merely understate the problem; it inverts it, showing the
system at its *fastest* precisely when it has stopped keeping up.

The mechanism: a closed-loop generator sends a request, waits for the response,
then sends the next. When the server slows, the generator slows with it, so the
slow period receives *fewer* samples than a fast one. Worse, the requests that
do get sent are the ones the server was ready for. The p95 then describes a load
level the system never actually experienced.

Corrected accounting fixes this by scheduling arrivals in advance and charging
each request for the time it spent waiting to be sent — latency measured from
*intended* send time, not actual. The `p50` falling (20.29 → 16.98 ms) while
`p95` triples is the signature: a subset of requests still complete quickly
while the queue behind them grows.

## Why this is reported rather than the headline

At 50 rps the generator itself could not keep to its schedule, which is flagged
in the output rather than silently folded into the percentile. **A latency
figure from a run where the generator fell behind is not a measurement of the
system; it is a measurement of the generator.** The defensible capacity claim
from this run is therefore: **25 rps sustained at 25.93 ms p95 corrected, zero
errors**, with the knee below 50 rps.

Single machine, with Qdrant, the encoder, and the load generator all sharing 12
cores — so this is a lower bound on what separated ingest and serving would
reach.
