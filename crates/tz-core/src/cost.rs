//! Per-request cost attribution.
//!
//! Two rules drive this module, both from measurement practice rather than taste:
//!
//! 1. **Token classes are not interchangeable.** Cache writes carry a premium
//!    (1.25x input at the 5-minute TTL, 2.0x at the hour) and cache reads a deep
//!    discount (0.1x). Collapsing them into one "input" number makes the cost
//!    figure wrong by 25-100% on any request that writes cache.
//! 2. **Prices change.** The price table is versioned data with a fetch date,
//!    not a constant in code, and its version is stamped onto every trace so a
//!    historical cost can be recomputed exactly as it was reported.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Token counts for a single model call, split by billing class.
///
/// Field names mirror the Anthropic `usage` object so deserialization is direct
/// and a missing field cannot silently become a zero-cost request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    /// Tokens written into the cache on this request. Billed at a premium.
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    /// Tokens served from cache. Billed at 0.1x input.
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}

impl Usage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
    }

    /// Fraction of billable input served from cache.
    ///
    /// Returns `None` when there was no input at all, which is a different
    /// statement from "0% cache hit" and must not be averaged in as zero.
    pub fn cache_hit_ratio(&self) -> Option<f64> {
        let billable_input =
            self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens;
        if billable_input == 0 {
            return None;
        }
        Some(self.cache_read_input_tokens as f64 / billable_input as f64)
    }
}

impl std::ops::Add for Usage {
    type Output = Usage;
    fn add(self, o: Usage) -> Usage {
        Usage {
            input_tokens: self.input_tokens + o.input_tokens,
            output_tokens: self.output_tokens + o.output_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens
                + o.cache_creation_input_tokens,
            cache_read_input_tokens: self.cache_read_input_tokens + o.cache_read_input_tokens,
        }
    }
}

impl std::iter::Sum for Usage {
    fn sum<I: Iterator<Item = Usage>>(iter: I) -> Usage {
        iter.fold(Usage::default(), |a, b| a + b)
    }
}

/// Which cache TTL a request used. Determines the cache-write multiplier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTtl {
    #[default]
    FiveMinutes,
    OneHour,
}

/// Rates for one model, in dollars per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelPrice {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
    pub cache_write_5m_per_mtok: f64,
    pub cache_write_1h_per_mtok: f64,
    pub cache_read_per_mtok: f64,
}

impl ModelPrice {
    /// Reads needed before caching a prefix pays for itself.
    ///
    /// Writing costs `w` and each subsequent read saves `input - read`, so the
    /// break-even read count is `(w - input) / (input - read)`. At Anthropic's
    /// published multipliers this lands near 1.4 reads for the 5-minute TTL and
    /// 2.1 for the hour: caching is worth it almost immediately, which is why
    /// a low measured cache-read count usually means a broken prefix, not a
    /// workload that legitimately cannot cache.
    pub fn cache_breakeven_reads(&self, ttl: CacheTtl) -> Option<f64> {
        let write = match ttl {
            CacheTtl::FiveMinutes => self.cache_write_5m_per_mtok,
            CacheTtl::OneHour => self.cache_write_1h_per_mtok,
        };
        let saving_per_read = self.input_per_mtok - self.cache_read_per_mtok;
        if saving_per_read <= 0.0 {
            return None;
        }
        Some((write - self.input_per_mtok) / saving_per_read)
    }
}

/// A dated, versioned set of model prices.
///
/// Loaded from `config/prices.toml`. The `version` string travels with every
/// cost figure this system emits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriceTable {
    pub version: String,
    pub fetched_at: String,
    pub source: String,
    #[serde(default = "default_currency")]
    pub currency: String,
    pub models: HashMap<String, ModelPrice>,
}

fn default_currency() -> String {
    "USD".to_string()
}

#[derive(Debug, thiserror::Error)]
pub enum PriceError {
    #[error("no price entry for model {0:?} in price table {1}")]
    UnknownModel(String, String),
    #[error("failed to parse price table: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("failed to read price table: {0}")]
    Io(#[from] std::io::Error),
}

impl PriceTable {
    pub fn from_toml_str(s: &str) -> Result<Self, PriceError> {
        Ok(toml::from_str(s)?)
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, PriceError> {
        Self::from_toml_str(&std::fs::read_to_string(path)?)
    }

    pub fn price_for(&self, model: &str) -> Result<&ModelPrice, PriceError> {
        self.models
            .get(model)
            .ok_or_else(|| PriceError::UnknownModel(model.to_string(), self.version.clone()))
    }

    /// Cost of one model call.
    ///
    /// Errors on an unknown model rather than defaulting to zero. A silently
    /// free model is how a cost SLO gets reported as passing while it is not.
    pub fn cost_of(
        &self,
        model: &str,
        usage: &Usage,
        ttl: CacheTtl,
    ) -> Result<CallCost, PriceError> {
        let p = self.price_for(model)?;
        let write_rate = match ttl {
            CacheTtl::FiveMinutes => p.cache_write_5m_per_mtok,
            CacheTtl::OneHour => p.cache_write_1h_per_mtok,
        };
        const M: f64 = 1_000_000.0;
        Ok(CallCost {
            model: model.to_string(),
            input_usd: usage.input_tokens as f64 * p.input_per_mtok / M,
            output_usd: usage.output_tokens as f64 * p.output_per_mtok / M,
            cache_write_usd: usage.cache_creation_input_tokens as f64 * write_rate / M,
            cache_read_usd: usage.cache_read_input_tokens as f64 * p.cache_read_per_mtok / M,
            usage: *usage,
            price_table_version: self.version.clone(),
        })
    }
}

/// Cost of a single model call, itemized.
///
/// Kept itemized rather than summed so a dashboard can answer "where did the
/// money go" without re-deriving it from token counts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallCost {
    pub model: String,
    pub input_usd: f64,
    pub output_usd: f64,
    pub cache_write_usd: f64,
    pub cache_read_usd: f64,
    pub usage: Usage,
    pub price_table_version: String,
}

impl CallCost {
    pub fn total_usd(&self) -> f64 {
        self.input_usd + self.output_usd + self.cache_write_usd + self.cache_read_usd
    }
}

/// Cost of one user-visible request, across every model call it made.
///
/// The SLO is defined on this, not on individual calls. An agent turn that
/// routes, generates, and samples a judge is one request and one budget.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestCost {
    pub calls: Vec<CallCost>,
}

impl RequestCost {
    pub fn push(&mut self, c: CallCost) {
        self.calls.push(c);
    }

    pub fn total_usd(&self) -> f64 {
        self.calls.iter().map(|c| c.total_usd()).sum()
    }

    pub fn total_usage(&self) -> Usage {
        self.calls.iter().map(|c| c.usage).sum()
    }

    /// Per-model breakdown, for the "cost by route tier" dashboard row.
    pub fn by_model(&self) -> HashMap<&str, f64> {
        let mut m: HashMap<&str, f64> = HashMap::new();
        for c in &self.calls {
            *m.entry(c.model.as_str()).or_default() += c.total_usd();
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> PriceTable {
        PriceTable::from_toml_str(
            r#"
version = "test.1"
fetched_at = "2026-09-06"
source = "test"

[models."claude-haiku-4-5"]
input_per_mtok = 1.0
output_per_mtok = 5.0
cache_write_5m_per_mtok = 1.25
cache_write_1h_per_mtok = 2.0
cache_read_per_mtok = 0.1

[models."claude-sonnet-5"]
input_per_mtok = 2.0
output_per_mtok = 10.0
cache_write_5m_per_mtok = 2.5
cache_write_1h_per_mtok = 4.0
cache_read_per_mtok = 0.2
"#,
        )
        .expect("test price table parses")
    }

    #[test]
    fn haiku_20k_in_1k_out_fits_the_six_cent_budget() {
        let t = table();
        let u = Usage { input_tokens: 20_000, output_tokens: 1_000, ..Default::default() };
        let c = t.cost_of("claude-haiku-4-5", &u, CacheTtl::FiveMinutes).unwrap();
        // 20k * $1/Mtok = $0.02 ; 1k * $5/Mtok = $0.005
        assert!((c.total_usd() - 0.025).abs() < 1e-9, "got {}", c.total_usd());
        assert!(c.total_usd() < 0.06);
    }

    #[test]
    fn sonnet_same_shape_consumes_most_of_the_budget() {
        let t = table();
        let u = Usage { input_tokens: 20_000, output_tokens: 1_000, ..Default::default() };
        let c = t.cost_of("claude-sonnet-5", &u, CacheTtl::FiveMinutes).unwrap();
        assert!((c.total_usd() - 0.05).abs() < 1e-9, "got {}", c.total_usd());
        assert!(c.total_usd() < 0.06, "still fits, but with only 17% headroom");
    }

    #[test]
    fn cache_read_is_an_order_of_magnitude_cheaper_than_fresh_input() {
        let t = table();
        let fresh = Usage { input_tokens: 20_000, ..Default::default() };
        let cached = Usage { cache_read_input_tokens: 20_000, ..Default::default() };
        let a = t.cost_of("claude-haiku-4-5", &fresh, CacheTtl::FiveMinutes).unwrap();
        let b = t.cost_of("claude-haiku-4-5", &cached, CacheTtl::FiveMinutes).unwrap();
        assert!((a.total_usd() / b.total_usd() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn ignoring_the_cache_write_premium_understates_cost() {
        // The failure this guards against: treating cache-creation tokens as
        // plain input. At the 5-minute TTL that understates by exactly 20%.
        let t = table();
        let u = Usage { cache_creation_input_tokens: 100_000, ..Default::default() };
        let correct = t.cost_of("claude-haiku-4-5", &u, CacheTtl::FiveMinutes).unwrap();
        let naive_as_input = 100_000.0 * 1.0 / 1e6;
        assert!(correct.total_usd() > naive_as_input);
        assert!((naive_as_input / correct.total_usd() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn one_hour_ttl_costs_more_to_write_than_five_minute() {
        let t = table();
        let u = Usage { cache_creation_input_tokens: 50_000, ..Default::default() };
        let short = t.cost_of("claude-haiku-4-5", &u, CacheTtl::FiveMinutes).unwrap();
        let long = t.cost_of("claude-haiku-4-5", &u, CacheTtl::OneHour).unwrap();
        assert!(long.total_usd() > short.total_usd());
    }

    #[test]
    fn cache_breakeven_lands_near_published_multipliers() {
        let t = table();
        let p = t.price_for("claude-haiku-4-5").unwrap();
        let five = p.cache_breakeven_reads(CacheTtl::FiveMinutes).unwrap();
        let hour = p.cache_breakeven_reads(CacheTtl::OneHour).unwrap();
        // (1.25 - 1.0) / (1.0 - 0.1) = 0.278 reads beyond the write itself
        assert!((five - 0.2777).abs() < 1e-3, "got {five}");
        assert!((hour - 1.1111).abs() < 1e-3, "got {hour}");
        assert!(hour > five);
    }

    #[test]
    fn unknown_model_errors_rather_than_costing_zero() {
        let t = table();
        let u = Usage { input_tokens: 1_000_000, ..Default::default() };
        let r = t.cost_of("claude-does-not-exist", &u, CacheTtl::FiveMinutes);
        assert!(matches!(r, Err(PriceError::UnknownModel(_, _))));
    }

    #[test]
    fn request_cost_sums_the_whole_trace_not_one_call() {
        let t = table();
        let mut rc = RequestCost::default();
        // A realistic tier-2 trace: cheap router, then generation.
        rc.push(
            t.cost_of(
                "claude-haiku-4-5",
                &Usage { input_tokens: 400, output_tokens: 20, ..Default::default() },
                CacheTtl::FiveMinutes,
            )
            .unwrap(),
        );
        rc.push(
            t.cost_of(
                "claude-sonnet-5",
                &Usage {
                    input_tokens: 2_000,
                    output_tokens: 600,
                    cache_read_input_tokens: 12_000,
                    ..Default::default()
                },
                CacheTtl::FiveMinutes,
            )
            .unwrap(),
        );
        assert_eq!(rc.calls.len(), 2);
        assert_eq!(rc.by_model().len(), 2);
        assert!(rc.total_usd() < 0.06, "trace total {} over budget", rc.total_usd());
        assert_eq!(rc.total_usage().total_tokens(), 400 + 20 + 2_000 + 600 + 12_000);
    }

    #[test]
    fn cache_hit_ratio_distinguishes_no_input_from_no_hits() {
        assert_eq!(Usage::default().cache_hit_ratio(), None);
        let u = Usage { input_tokens: 100, cache_read_input_tokens: 900, ..Default::default() };
        assert!((u.cache_hit_ratio().unwrap() - 0.9).abs() < 1e-9);
    }
}
