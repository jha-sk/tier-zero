//! One request's full accounting: what it cost, how long each stage took, and
//! under what conditions.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tz_core::{RequestCost, RouteTier};

/// Wall time of one named stage.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StageTiming {
    pub name: &'static str,
    pub micros: u64,
}

/// Everything worth knowing about one user-visible request.
///
/// Serialized to the trace and to structured logs. Deliberately flat and
/// explicit: a dashboard should not have to join anything to answer "what did
/// this request cost, who was it for, and which tier served it".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRecord {
    pub trace_id: String,
    pub tenant_id: String,
    pub route_tier: RouteTier,
    /// Cache outcome: exact hit, semantic hit, or miss.
    pub cache_outcome: CacheOutcome,
    /// Similarity of a semantic cache hit, retained for sampled review. A
    /// semantic cache trades a cost problem for a silent correctness problem,
    /// and the score is the only evidence available after the fact.
    pub cache_similarity: Option<f32>,
    pub cost: RequestCost,
    pub stages: Vec<(String, u64)>,
    pub retrieval_candidates: usize,
    /// True when the request was served by a degraded path. How often the
    /// fallback fires is itself an SLI: a system that is "99.9% successful"
    /// while serving 40% degraded answers is not healthy.
    pub degraded: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheOutcome {
    ExactHit,
    SemanticHit,
    Miss,
}

impl CacheOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            CacheOutcome::ExactHit => "exact_hit",
            CacheOutcome::SemanticHit => "semantic_hit",
            CacheOutcome::Miss => "miss",
        }
    }
    pub fn is_hit(&self) -> bool {
        !matches!(self, CacheOutcome::Miss)
    }
}

impl RequestRecord {
    pub fn new(trace_id: impl Into<String>, tenant_id: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            tenant_id: tenant_id.into(),
            route_tier: RouteTier::One,
            cache_outcome: CacheOutcome::Miss,
            cache_similarity: None,
            cost: RequestCost::default(),
            stages: Vec::new(),
            retrieval_candidates: 0,
            degraded: false,
            error: None,
        }
    }

    pub fn stage(&mut self, name: impl Into<String>, micros: u64) {
        self.stages.push((name.into(), micros));
    }

    pub fn total_usd(&self) -> f64 {
        self.cost.total_usd()
    }

    pub fn total_ms(&self) -> f64 {
        self.stages.iter().map(|(_, us)| *us).sum::<u64>() as f64 / 1000.0
    }

    /// Span attributes for this request.
    ///
    /// Emitted as a map rather than set ad hoc at call sites, so every request
    /// carries the same keys and a dashboard query cannot silently miss a slice.
    pub fn attributes(&self) -> BTreeMap<&'static str, String> {
        let mut m = BTreeMap::new();
        m.insert(crate::attr::TZ_TENANT_ID, self.tenant_id.clone());
        m.insert(crate::attr::TZ_ROUTE_TIER, self.route_tier.as_str().to_string());
        m.insert(crate::attr::TZ_CACHE_OUTCOME, self.cache_outcome.as_str().to_string());
        m.insert(crate::attr::TZ_COST_USD, format!("{:.6}", self.total_usd()));
        m.insert(crate::attr::TZ_DEGRADED, self.degraded.to_string());
        m.insert(crate::attr::TZ_RETRIEVAL_CANDIDATES, self.retrieval_candidates.to_string());
        if let Some(s) = self.cache_similarity {
            m.insert(crate::attr::TZ_CACHE_SIMILARITY, format!("{s:.4}"));
        }
        let u = self.cost.total_usage();
        m.insert(crate::attr::GEN_AI_USAGE_INPUT_TOKENS, u.input_tokens.to_string());
        m.insert(crate::attr::GEN_AI_USAGE_OUTPUT_TOKENS, u.output_tokens.to_string());
        m.insert(crate::attr::TZ_CACHE_READ_TOKENS, u.cache_read_input_tokens.to_string());
        m.insert(crate::attr::TZ_CACHE_WRITE_TOKENS, u.cache_creation_input_tokens.to_string());
        if let Some(c) = self.cost.calls.first() {
            m.insert(crate::attr::TZ_PRICE_TABLE_VERSION, c.price_table_version.clone());
            m.insert(crate::attr::GEN_AI_RESPONSE_MODEL, c.model.clone());
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tz_core::{CacheTtl, PriceTable, Usage};

    fn table() -> PriceTable {
        PriceTable::from_toml_str(
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
        .unwrap()
    }

    #[test]
    fn a_record_carries_every_slice_a_dashboard_needs() {
        let mut r = RequestRecord::new("trace-1", "acme");
        r.cost.push(
            table()
                .cost_of(
                    "claude-haiku-4-5",
                    &Usage { input_tokens: 1000, output_tokens: 100, ..Default::default() },
                    CacheTtl::FiveMinutes,
                )
                .unwrap(),
        );
        let a = r.attributes();
        for k in [
            crate::attr::TZ_TENANT_ID,
            crate::attr::TZ_ROUTE_TIER,
            crate::attr::TZ_COST_USD,
            crate::attr::TZ_PRICE_TABLE_VERSION,
            crate::attr::TZ_CACHE_OUTCOME,
        ] {
            assert!(a.contains_key(k), "missing {k}");
        }
        assert_eq!(a[crate::attr::TZ_PRICE_TABLE_VERSION], "t.1");
    }

    #[test]
    fn cost_is_reported_per_request_not_per_call() {
        let mut r = RequestRecord::new("t", "acme");
        let t = table();
        for _ in 0..3 {
            r.cost.push(
                t.cost_of(
                    "claude-haiku-4-5",
                    &Usage { input_tokens: 1000, ..Default::default() },
                    CacheTtl::FiveMinutes,
                )
                .unwrap(),
            );
        }
        assert_eq!(r.cost.calls.len(), 3);
        assert!((r.total_usd() - 0.003).abs() < 1e-9, "got {}", r.total_usd());
    }

    #[test]
    fn cache_similarity_is_recorded_only_for_semantic_hits() {
        let mut r = RequestRecord::new("t", "acme");
        assert!(!r.attributes().contains_key(crate::attr::TZ_CACHE_SIMILARITY));
        r.cache_outcome = CacheOutcome::SemanticHit;
        r.cache_similarity = Some(0.94);
        let a = r.attributes();
        assert_eq!(a[crate::attr::TZ_CACHE_SIMILARITY], "0.9400");
        assert_eq!(a[crate::attr::TZ_CACHE_OUTCOME], "semantic_hit");
    }

    #[test]
    fn the_degraded_flag_is_always_present_because_it_is_an_sli() {
        // How often the fallback fires matters as much as the success rate.
        let r = RequestRecord::new("t", "acme");
        assert_eq!(r.attributes()[crate::attr::TZ_DEGRADED], "false");
    }

    #[test]
    fn stage_timings_sum_to_the_reported_total() {
        let mut r = RequestRecord::new("t", "acme");
        r.stage("embed", 9_000);
        r.stage("search", 6_000);
        r.stage("generate", 800_000);
        assert!((r.total_ms() - 815.0).abs() < 1e-6);
    }

    #[test]
    fn a_cache_miss_is_not_a_hit() {
        assert!(!CacheOutcome::Miss.is_hit());
        assert!(CacheOutcome::ExactHit.is_hit());
        assert!(CacheOutcome::SemanticHit.is_hit());
    }
}
