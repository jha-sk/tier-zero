//! Enforcing the per-request cost budget *before* the spend happens.
//!
//! A cost SLO that is only checked in a dashboard is not a budget, it is a
//! report. By the time a p95 breach shows up on a chart the money is gone. The
//! guard here projects the cost of a call from its input size and the model's
//! rate card, and refuses or downgrades the call when the projection exceeds
//! what remains of the request's budget.
//!
//! Projection is necessarily approximate on the output side, because output
//! length is not known in advance. That is handled by charging the projection
//! against `max_tokens` — the only output bound that is actually enforced —
//! rather than against a hoped-for average.

use tz_core::{CacheTtl, PriceTable, RequestCost, Usage};

#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    #[error("projected cost ${projected:.4} would exceed the remaining budget ${remaining:.4}")]
    WouldExceed { projected: f64, remaining: f64 },
    #[error("budget already exhausted: spent ${spent:.4} of ${budget:.4}")]
    Exhausted { spent: f64, budget: f64 },
    #[error(transparent)]
    Price(#[from] tz_core::cost::PriceError),
}

/// Tracks spend against one request's budget.
pub struct BudgetGuard<'a> {
    prices: &'a PriceTable,
    budget_usd: f64,
    cost: RequestCost,
}

impl<'a> BudgetGuard<'a> {
    pub fn new(prices: &'a PriceTable, budget_usd: f64) -> Self {
        Self { prices, budget_usd, cost: RequestCost::default() }
    }

    pub fn spent(&self) -> f64 {
        self.cost.total_usd()
    }

    pub fn remaining(&self) -> f64 {
        (self.budget_usd - self.spent()).max(0.0)
    }

    pub fn cost(&self) -> &RequestCost {
        &self.cost
    }

    /// Worst-case cost of a call, charged against `max_tokens` rather than an
    /// expected output length. A budget check that assumes the average output
    /// passes right up until the one request that runs long.
    pub fn project(
        &self,
        model: &str,
        input_tokens: u64,
        cached_input_tokens: u64,
        max_output_tokens: u64,
        ttl: CacheTtl,
    ) -> Result<f64, BudgetError> {
        let usage = Usage {
            input_tokens,
            output_tokens: max_output_tokens,
            cache_read_input_tokens: cached_input_tokens,
            cache_creation_input_tokens: 0,
        };
        Ok(self.prices.cost_of(model, &usage, ttl)?.total_usd())
    }

    /// Check a call against the remaining budget before making it.
    pub fn check(
        &self,
        model: &str,
        input_tokens: u64,
        cached_input_tokens: u64,
        max_output_tokens: u64,
        ttl: CacheTtl,
    ) -> Result<f64, BudgetError> {
        if self.remaining() <= 0.0 {
            return Err(BudgetError::Exhausted { spent: self.spent(), budget: self.budget_usd });
        }
        let projected =
            self.project(model, input_tokens, cached_input_tokens, max_output_tokens, ttl)?;
        if projected > self.remaining() {
            return Err(BudgetError::WouldExceed { projected, remaining: self.remaining() });
        }
        Ok(projected)
    }

    /// Record what a call actually cost, from the provider's own usage figures.
    /// Always the response's numbers, never the projection: they differ, and
    /// the projection is the pessimistic one.
    pub fn charge(
        &mut self,
        model: &str,
        usage: &Usage,
        ttl: CacheTtl,
    ) -> Result<f64, BudgetError> {
        let c = self.prices.cost_of(model, usage, ttl)?;
        let total = c.total_usd();
        self.cost.push(c);
        Ok(total)
    }

    /// Largest context, in chunks, that fits the remaining budget.
    ///
    /// This is what turns the budget from a tripwire into a control: instead of
    /// refusing a request that would overspend, trim what is sent until it fits.
    pub fn affordable_context_chunks(
        &self,
        model: &str,
        tokens_per_chunk: u64,
        overhead_tokens: u64,
        max_output_tokens: u64,
        ttl: CacheTtl,
    ) -> usize {
        let mut n = 0usize;
        loop {
            let input = overhead_tokens + (n as u64 + 1) * tokens_per_chunk;
            match self.check(model, input, 0, max_output_tokens, ttl) {
                Ok(_) => n += 1,
                Err(_) => return n,
            }
            if n > 512 {
                return n;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prices() -> PriceTable {
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
[models."claude-opus-5"]
input_per_mtok = 5.0
output_per_mtok = 25.0
cache_write_5m_per_mtok = 6.25
cache_write_1h_per_mtok = 10.0
cache_read_per_mtok = 0.5
"#,
        )
        .unwrap()
    }

    #[test]
    fn a_request_that_fits_is_allowed() {
        let p = prices();
        let g = BudgetGuard::new(&p, 0.06);
        let c = g.check("claude-haiku-4-5", 20_000, 0, 1_000, CacheTtl::FiveMinutes).unwrap();
        assert!((c - 0.025).abs() < 1e-9);
    }

    #[test]
    fn a_request_that_would_breach_the_budget_is_refused_before_it_is_sent() {
        // The point of the guard: this is caught before the money is spent,
        // not observed afterwards on a dashboard.
        let p = prices();
        let g = BudgetGuard::new(&p, 0.06);
        let e = g.check("claude-opus-5", 20_000, 0, 1_000, CacheTtl::FiveMinutes).unwrap_err();
        assert!(matches!(e, BudgetError::WouldExceed { .. }));
    }

    #[test]
    fn caching_brings_an_otherwise_unaffordable_model_into_budget() {
        let p = prices();
        let g = BudgetGuard::new(&p, 0.06);
        assert!(g.check("claude-opus-5", 20_000, 0, 1_000, CacheTtl::FiveMinutes).is_err());
        // Same context size, 18K of it served from cache.
        assert!(g.check("claude-opus-5", 2_000, 18_000, 1_000, CacheTtl::FiveMinutes).is_ok());
    }

    #[test]
    fn projection_charges_max_tokens_not_a_hoped_for_average() {
        // A guard that assumes the average output passes until the one request
        // that runs long, which is exactly the p95 the SLO is about.
        let p = prices();
        let g = BudgetGuard::new(&p, 0.06);
        let short = g.project("claude-haiku-4-5", 1_000, 0, 100, CacheTtl::FiveMinutes).unwrap();
        let long = g.project("claude-haiku-4-5", 1_000, 0, 4_000, CacheTtl::FiveMinutes).unwrap();
        assert!(long > short * 2.0);
    }

    #[test]
    fn spend_accumulates_across_a_trace_and_then_refuses() {
        let p = prices();
        let mut g = BudgetGuard::new(&p, 0.06);
        for _ in 0..2 {
            g.charge(
                "claude-haiku-4-5",
                &Usage { input_tokens: 20_000, output_tokens: 1_000, ..Default::default() },
                CacheTtl::FiveMinutes,
            )
            .unwrap();
        }
        assert!((g.spent() - 0.05).abs() < 1e-9);
        assert!((g.remaining() - 0.01).abs() < 1e-9);
        // A third identical call no longer fits.
        assert!(g.check("claude-haiku-4-5", 20_000, 0, 1_000, CacheTtl::FiveMinutes).is_err());
    }

    #[test]
    fn an_exhausted_budget_reports_exhaustion_not_a_size_problem() {
        let p = prices();
        let mut g = BudgetGuard::new(&p, 0.001);
        g.charge(
            "claude-haiku-4-5",
            &Usage { input_tokens: 20_000, output_tokens: 1_000, ..Default::default() },
            CacheTtl::FiveMinutes,
        )
        .unwrap();
        let e = g.check("claude-haiku-4-5", 10, 0, 10, CacheTtl::FiveMinutes).unwrap_err();
        assert!(matches!(e, BudgetError::Exhausted { .. }), "got {e:?}");
    }

    #[test]
    fn the_budget_trims_context_instead_of_refusing_the_request() {
        // Turning the budget from a tripwire into a control.
        let p = prices();
        let g = BudgetGuard::new(&p, 0.06);
        let cheap = g.affordable_context_chunks("claude-haiku-4-5", 400, 500, 1_000, CacheTtl::FiveMinutes);
        let dear = g.affordable_context_chunks("claude-opus-5", 400, 500, 1_000, CacheTtl::FiveMinutes);
        assert!(cheap > dear, "cheap model affords more context: {cheap} vs {dear}");
        assert!(dear > 0, "even the expensive model affords some context");
    }

    #[test]
    fn an_unknown_model_is_an_error_not_a_free_pass() {
        let p = prices();
        let g = BudgetGuard::new(&p, 0.06);
        assert!(matches!(
            g.check("claude-nonexistent", 100, 0, 100, CacheTtl::FiveMinutes),
            Err(BudgetError::Price(_))
        ));
    }
}
