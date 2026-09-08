//! Service level objectives, declared as data and checked against measurements.
//!
//! The budgets here are the ones the project committed to before any code was
//! written. They are deliberately expressed as a struct rather than scattered
//! through assertions so that a single report can say which held and which did
//! not, and so that a run against a changed budget is visibly a different run.

use serde::{Deserialize, Serialize};

/// The four budgets. Quality is one of them on purpose: the other three can all
/// be satisfied by returning garbage instantly and for free.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SloBudget {
    /// p95 of the retrieval span: gateway ingress to candidate list ready.
    /// Includes query embedding, ANN search, fusion, payload fetch.
    /// Excludes client network and generation.
    pub retrieval_p95_ms: f64,
    /// p95 dollars per answered request, summed across the whole trace.
    pub cost_p95_usd: f64,
    pub ttft_p95_ms: f64,
    pub total_p95_ms: f64,
    pub total_p99_ms: f64,
    /// Retrieval quality floor. Latency without its recall is meaningless.
    pub recall_at_50_min: f64,
    pub ndcg_at_10_min: f64,
    pub citation_accuracy_min: f64,
    /// A semantic cache converts a cost problem into a silent correctness
    /// problem. This is the bound on that conversion.
    pub semantic_cache_false_hit_max: f64,
}

impl Default for SloBudget {
    /// The budgets as committed. Changing these is a deliberate act.
    fn default() -> Self {
        Self {
            retrieval_p95_ms: 25.0,
            cost_p95_usd: 0.06,
            ttft_p95_ms: 1_000.0,
            total_p95_ms: 5_000.0,
            total_p99_ms: 10_000.0,
            recall_at_50_min: 0.85,
            ndcg_at_10_min: 0.0, // gated baseline-relative; see EvalGate
            citation_accuracy_min: 0.90,
            semantic_cache_false_hit_max: 0.01,
        }
    }
}

/// The conditions a latency measurement was taken under.
///
/// A latency number without these is not a claim, it is a decoration. Every
/// reported percentile carries this so the README cannot accidentally quote a
/// warm, unloaded, no-ingest number as a production figure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementContext {
    /// Offered load in requests/second. Open-model, constant arrival rate.
    pub arrival_rate_rps: f64,
    pub corpus_chunks: u64,
    pub concurrent_ingest: bool,
    pub warm: bool,
    pub transport: Transport,
    pub host: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    Grpc,
    Rest,
}

/// One SLO check: a budget, a measurement, and whether it held.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloCheck {
    pub name: String,
    pub budget: f64,
    pub measured: f64,
    pub unit: String,
    /// True when lower is better (latency, cost). False for quality floors.
    pub lower_is_better: bool,
}

impl SloCheck {
    pub fn passed(&self) -> bool {
        if self.lower_is_better {
            self.measured <= self.budget
        } else {
            self.measured >= self.budget
        }
    }

    /// How much of the budget was consumed, as a fraction.
    /// Useful for spotting a check that passes with no headroom.
    pub fn headroom(&self) -> f64 {
        if self.budget == 0.0 {
            return f64::NAN;
        }
        if self.lower_is_better {
            1.0 - (self.measured / self.budget)
        } else {
            (self.measured / self.budget) - 1.0
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloReport {
    pub context: MeasurementContext,
    pub price_table_version: String,
    pub checks: Vec<SloCheck>,
}

impl SloReport {
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed())
    }

    pub fn failures(&self) -> Vec<&SloCheck> {
        self.checks.iter().filter(|c| !c.passed()).collect()
    }

    /// Checks that passed with less than 10% of budget to spare.
    /// These are the ones that will break first under any change.
    pub fn tight(&self) -> Vec<&SloCheck> {
        self.checks
            .iter()
            .filter(|c| c.passed() && c.headroom() < 0.10)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasurementContext {
        MeasurementContext {
            arrival_rate_rps: 50.0,
            corpus_chunks: 2_000_000,
            concurrent_ingest: false,
            warm: true,
            transport: Transport::Grpc,
            host: "local-12core".into(),
        }
    }

    #[test]
    fn committed_budgets_are_the_ones_in_the_plan() {
        let b = SloBudget::default();
        assert_eq!(b.retrieval_p95_ms, 25.0);
        assert_eq!(b.cost_p95_usd, 0.06);
        assert_eq!(b.ttft_p95_ms, 1_000.0);
        assert_eq!(b.total_p95_ms, 5_000.0);
        assert_eq!(b.recall_at_50_min, 0.85);
    }

    #[test]
    fn latency_check_direction_is_lower_is_better() {
        let c = SloCheck {
            name: "retrieval_p95".into(),
            budget: 25.0,
            measured: 23.1,
            unit: "ms".into(),
            lower_is_better: true,
        };
        assert!(c.passed());
        assert!(c.headroom() > 0.0 && c.headroom() < 0.1, "8% headroom is tight");
    }

    #[test]
    fn quality_check_direction_is_higher_is_better() {
        let c = SloCheck {
            name: "recall@50".into(),
            budget: 0.85,
            measured: 0.81,
            unit: "".into(),
            lower_is_better: false,
        };
        assert!(!c.passed(), "0.81 recall must fail a 0.85 floor");
    }

    #[test]
    fn report_surfaces_tight_passes_not_just_failures() {
        let r = SloReport {
            context: ctx(),
            price_table_version: "test.1".into(),
            checks: vec![
                SloCheck {
                    name: "retrieval_p95".into(),
                    budget: 25.0,
                    measured: 24.6,
                    unit: "ms".into(),
                    lower_is_better: true,
                },
                SloCheck {
                    name: "cost_p95".into(),
                    budget: 0.06,
                    measured: 0.021,
                    unit: "usd".into(),
                    lower_is_better: true,
                },
            ],
        };
        assert!(r.passed());
        assert!(r.failures().is_empty());
        let tight = r.tight();
        assert_eq!(tight.len(), 1, "retrieval is tight, cost is not");
        assert_eq!(tight[0].name, "retrieval_p95");
    }

    #[test]
    fn a_measurement_carries_the_conditions_it_was_taken_under() {
        // Guards against the most common dishonest benchmark: quoting a warm,
        // unloaded, no-ingest number as if it were production.
        let c = ctx();
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("arrival_rate_rps"));
        assert!(json.contains("concurrent_ingest"));
        assert!(json.contains("corpus_chunks"));
    }
}
