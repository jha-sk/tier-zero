#!/usr/bin/env bash
# Starts the observability stack from standalone binaries -- no Docker, and so
# no need to grant anyone root-equivalent access to run a dashboard.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OBS="$HERE/../../data/obs"
mkdir -p "$HERE/run" && cd "$HERE/run"

start() { echo "starting $1"; shift; nohup "$@" >> "$1.log" 2>&1 & }

nohup "$OBS/prometheus/prometheus" \
  --config.file="$HERE/prometheus.yml" \
  --storage.tsdb.path=./prom-data \
  --enable-feature=native-histograms > prometheus.log 2>&1 &

nohup "$OBS/tempo/tempo" -config.file="$HERE/tempo.yaml" > tempo.log 2>&1 &

nohup "$OBS/otelcol/otelcol-contrib" --config="$HERE/otelcol.yaml" > otelcol.log 2>&1 &

nohup "$OBS/grafana/bin/grafana" server \
  --homepath "$OBS/grafana" \
  --config /dev/null \
  cfg:default.paths.provisioning="$HERE/provisioning" \
  cfg:default.security.admin_password=admin \
  cfg:default.auth.anonymous.enabled=true \
  cfg:default.auth.anonymous.org_role=Admin \
  cfg:default.server.http_port=3000 \
  > grafana.log 2>&1 &

echo "prometheus :9090  tempo :3200  otel :4317/:4318  grafana :3000"
