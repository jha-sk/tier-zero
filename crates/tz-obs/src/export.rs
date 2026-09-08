//! OTLP metric export.
//!
//! Without this the crate would define attribute names and emit nothing, which
//! is the usual state of "observability" in a portfolio project: conventions
//! documented, no data.
//!
//! Metrics rather than spans, deliberately. The claims this project makes are
//! distributional — a p95, a cost distribution, a hit rate — and those are
//! aggregations over many requests. Traces answer "what happened to *this*
//! request", which matters for debugging but is not what an SLO is stated in.

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Histogram, Meter};
use tz_core::RouteTier;

use crate::record::RequestRecord;

/// Instruments for one process.
pub struct Metrics {
    /// Seconds per retrieval stage, labelled by stage.
    stage_seconds: Histogram<f64>,
    /// Dollars per user-visible request.
    ///
    /// A histogram, not a counter: the SLO is a p95 of cost, and total spend
    /// divided by request count cannot be turned back into a percentile.
    cost_usd: Histogram<f64>,
    /// Tokens per request, labelled by billing class so cache reads and cache
    /// writes stay distinguishable.
    tokens: Histogram<u64>,
}

/// Bucket boundaries for latency, in **seconds**.
///
/// The SDK default is `[0, 5, 10, 25, 50, 75, 100, 250, 500, ...]`, which is
/// sized for milliseconds. Recording seconds against it puts every real value
/// in the first bucket, and `histogram_quantile` then interpolates across a
/// bucket spanning 0-5 seconds: an 80ms retrieval reported a p95 of 4,750ms.
/// The number was not noise, it was the midpoint of an empty range.
///
/// These boundaries straddle the 25ms budget closely enough that a p95 near the
/// SLO is resolved rather than guessed.
const LATENCY_BOUNDS_S: &[f64] = &[
    0.001, 0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040,
    0.050, 0.075, 0.100, 0.250, 0.500, 1.0, 2.5, 5.0, 10.0,
];

/// Bucket boundaries for cost, in **USD**.
///
/// Dense around the $0.06 budget for the same reason: a percentile is only as
/// precise as the bucket it falls in, and an SLO measured through a bucket
/// three times wider than the budget cannot be evaluated.
const COST_BOUNDS_USD: &[f64] = &[
    0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.010, 0.020, 0.030, 0.040, 0.050,
    0.060, 0.070, 0.080, 0.100, 0.250, 0.500, 1.0,
];

impl Metrics {
    pub fn new(meter: &Meter) -> Self {
        Self {
            stage_seconds: meter
                .f64_histogram("tierzero.stage.duration")
                .with_description("Wall time per retrieval stage")
                .with_unit("s")
                .with_boundaries(LATENCY_BOUNDS_S.to_vec())
                .build(),
            cost_usd: meter
                .f64_histogram("tierzero.request.cost")
                .with_description("Cost of one user-visible request, whole trace")
                .with_unit("USD")
                .with_boundaries(COST_BOUNDS_USD.to_vec())
                .build(),
            tokens: meter
                .u64_histogram("gen_ai.client.token.usage")
                .with_description("Tokens per request by billing class")
                .with_unit("{token}")
                .build(),
        }
    }

    /// Emit everything for one completed request.
    ///
    /// Every metric carries tenant, tier and cache outcome, so a dashboard can
    /// slice without a join. A single unlabelled aggregate is the tell of a
    /// dashboard that cannot answer a real question.
    pub fn record(&self, r: &RequestRecord) {
        let base = [
            KeyValue::new("tenant", r.tenant_id.clone()),
            KeyValue::new("route_tier", tier_label(r.route_tier)),
            KeyValue::new("cache_outcome", r.cache_outcome.as_str()),
            KeyValue::new("degraded", r.degraded.to_string()),
        ];

        for (stage, micros) in &r.stages {
            let mut attrs = base.to_vec();
            attrs.push(KeyValue::new("stage", stage.clone()));
            self.stage_seconds.record(*micros as f64 / 1_000_000.0, &attrs);
        }

        self.cost_usd.record(r.total_usd(), &base);

        let u = r.cost.total_usage();
        for (class, n) in [
            ("input", u.input_tokens),
            ("output", u.output_tokens),
            ("cache_read", u.cache_read_input_tokens),
            ("cache_write", u.cache_creation_input_tokens),
        ] {
            if n == 0 {
                continue;
            }
            let mut attrs = base.to_vec();
            attrs.push(KeyValue::new("token_class", class));
            self.tokens.record(n, &attrs);
        }
    }
}

fn tier_label(t: RouteTier) -> &'static str {
    t.as_str()
}

/// Build an OTLP meter provider pointed at a collector.
///
/// Returns the provider so the caller owns shutdown; dropping it without
/// flushing loses whatever is still batched, which on a short benchmark run is
/// most of the data.
pub fn init_meter_provider(
    endpoint: &str,
    service_name: &'static str,
) -> anyhow::Result<opentelemetry_sdk::metrics::SdkMeterProvider> {
    use opentelemetry_otlp::WithExportConfig;

    let exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_periodic_exporter(exporter)
        .with_resource(
            opentelemetry_sdk::Resource::builder()
                .with_service_name(service_name)
                .build(),
        )
        .build();

    opentelemetry::global::set_meter_provider(provider.clone());
    Ok(provider)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::CacheOutcome;
    use tz_core::{CacheTtl, PriceTable, Usage};

    fn record() -> RequestRecord {
        let t = PriceTable::from_toml_str(
            r#"
version = "t.1"
fetched_at = "2026-09-07"
source = "test"
[models."claude-haiku-4-5"]
input_per_mtok = 1.0
output_per_mtok = 5.0
cache_write_5m_per_mtok = 1.25
cache_write_1h_per_mtok = 2.0
cache_read_per_mtok = 0.1
"#,
        )
        .unwrap();
        let mut r = RequestRecord::new("trace-1", "acme");
        r.stage("embed", 8_700);
        r.stage("search", 6_000);
        r.cache_outcome = CacheOutcome::Miss;
        r.cost.push(
            t.cost_of(
                "claude-haiku-4-5",
                &Usage {
                    input_tokens: 2_000,
                    output_tokens: 300,
                    cache_read_input_tokens: 12_000,
                    ..Default::default()
                },
                CacheTtl::FiveMinutes,
            )
            .unwrap(),
        );
        r
    }

    #[test]
    fn recording_a_request_does_not_panic_without_a_configured_provider() {
        // The no-op global provider is the default. Instrumentation must be
        // safe to call in tests and in a process with no collector reachable.
        let meter = opentelemetry::global::meter("test");
        let m = Metrics::new(&meter);
        m.record(&record());
    }

    #[test]
    fn latency_buckets_resolve_the_twenty_five_millisecond_budget() {
        // A percentile is only as precise as the bucket it lands in. With the
        // SDK defaults (sized for ms, fed seconds) an 80ms retrieval reported
        // p95 = 4,750ms -- the midpoint of an empty 0-5s bucket.
        let near_budget: Vec<f64> =
            LATENCY_BOUNDS_S.iter().copied().filter(|b| (0.015..=0.050).contains(b)).collect();
        assert!(near_budget.len() >= 4, "too coarse around 25ms: {near_budget:?}");
        assert!(LATENCY_BOUNDS_S.contains(&0.025), "no boundary at the budget itself");
    }

    #[test]
    fn cost_buckets_resolve_the_six_cent_budget() {
        assert!(COST_BOUNDS_USD.contains(&0.060), "no boundary at the cost budget");
        let near: Vec<f64> =
            COST_BOUNDS_USD.iter().copied().filter(|b| (0.030..=0.080).contains(b)).collect();
        assert!(near.len() >= 5, "too coarse around $0.06: {near:?}");
    }

    #[test]
    fn boundaries_are_sorted_and_positive() {
        for b in [LATENCY_BOUNDS_S, COST_BOUNDS_USD] {
            assert!(b.windows(2).all(|w| w[0] < w[1]), "boundaries must ascend");
            assert!(b.iter().all(|x| *x > 0.0));
        }
    }

    #[test]
    fn tier_labels_match_the_ones_used_in_span_attributes() {
        // A mismatch here would silently split every dashboard series in two.
        for t in [RouteTier::Zero, RouteTier::One, RouteTier::Two] {
            assert_eq!(tier_label(t), t.as_str());
        }
    }

    #[test]
    fn a_request_with_no_tokens_still_records_its_stages() {
        let meter = opentelemetry::global::meter("test");
        let m = Metrics::new(&meter);
        let mut r = RequestRecord::new("t", "acme");
        r.stage("embed", 9_000);
        m.record(&r); // tier-zero path: no model call, so no token classes
    }
}
