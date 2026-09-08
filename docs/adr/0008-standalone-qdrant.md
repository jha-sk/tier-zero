# ADR 0008: Run Qdrant as a binary, not only under Docker

**Status:** Accepted · **Date:** 2026-09-07

## Context

The stack was specified as Docker Compose: Qdrant, an OTel collector,
Prometheus, Grafana and Tempo. On the development machine the Docker socket is
owned by `root:docker` and the user is not in the `docker` group.

The standard fix — adding the user to the `docker` group — grants effective root
on the host, because the Docker daemon runs as root and group membership is
unrestricted access to it. That is a real security decision belonging to the
machine's owner, not a build step to be performed on their behalf.

## Decision

Support both. `deploy/docker-compose.yml` remains the documented path for the
full observability stack. For development and benchmarking, Qdrant runs as a
**standalone binary** with the same latency-tuned configuration in
`data/qdrant-run/config/config.yaml`.

## Consequences

The binary path turned out to be better for this project's purpose, not merely
an acceptable substitute:

- **A newer server.** The pinned image was 1.16.1; the release binary is 1.19.1,
  which includes `read_fan_out_delay_ms` (1.17) and `X-Qdrant-Route-Affinity`
  (1.19) — the two best tail-latency levers available, both absent from the
  version originally pinned.
- **One less layer under measurement.** Container networking sits between client
  and server in the compose path. On a project whose headline claim is a 25ms
  p95 over loopback gRPC, removing that layer removes a confound.
- **Faster iteration.** Startup is ~1 second.

The tuning is identical in both paths and lives in configuration, not in the
container definition, so the two cannot drift:

| Setting | Value | Reason |
|---|---|---|
| `default_segment_number` | 12 | One per core; a single query fans out across segments |
| `optimizer_cpu_budget` | 6 | Ingest must not be able to starve the query path |
| `max_optimization_threads` | 2 | Same |
| `deleted_threshold` | 0.5 | Aggressive vacuum; at 0.2 a 25%-deleted segment measured 22.7ms p95 against 4.3ms at 0.5 |
| `on_disk_payload` | true | Only the final top-k needs payload text |

The full observability stack (Grafana dashboards, Tempo traces) still requires
Docker. That work is blocked on the group decision and is recorded as such
rather than worked around.
