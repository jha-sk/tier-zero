//! The front door: decide what a request costs before spending it.
//!
//! Routing exists because of arithmetic, not elegance. At a $0.06 p95 budget
//! and a typical 20K-context request, the strongest model costs $0.125 — twice
//! the budget — while the small model costs $0.025. Sending everything to the
//! strong model fails the SLO; sending everything to the small one fails the
//! questions that need reasoning. So the request has to be classified before it
//! is served.
//!
//! The classifier is deliberately **not** a model call. A model-based router
//! adds 30-100ms and its own token cost to every request including the ones it
//! routes to tier zero, which is precisely backwards: the cheapest path should
//! not have to pay for the privilege of being identified.

use once_cell::sync::Lazy;
use regex::Regex;
use tz_core::RouteTier;

/// Which model serves each tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierModels {
    pub standard: String,
    pub escalated: String,
}

impl Default for TierModels {
    fn default() -> Self {
        Self {
            // Fits the budget with room for a retry.
            standard: "claude-haiku-4-5".into(),
            // Fits with ~17% headroom; reserved for requests that need it.
            escalated: "claude-sonnet-5".into(),
        }
    }
}

/// A routing decision, with the reason retained for telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub tier: RouteTier,
    pub model: Option<String>,
    pub reason: &'static str,
    /// Cap on retrieved context for this tier, in chunks.
    ///
    /// This is the main cost lever after model choice. It is also a latency
    /// lever: time to first token is dominated by prefill, which is linear in
    /// input length, so a smaller context improves both numbers at once.
    pub context_cap: usize,
}

static TRIVIAL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^\s*(hi|hello|hey|thanks|thank you|ok|okay|yes|no)\b\s*[.!?]?\s*$")
        .expect("static regex")
});

static NEEDS_REASONING: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)
        \b(why|compare|comparison|versus|vs|trade-?off|design|architect|architecture)\b
        | \b(root\s+cause|diagnose|debug|troubleshoot|investigate)\b
        | \b(should\s+(i|we)|which\s+is\s+better|recommend|best\s+practice)\b
        | \b(migrate|migration|refactor|redesign)\b
        ",
    )
    .expect("static regex")
});

/// Classify a request.
///
/// Order matters: the cheapest determination is made first so that a tier-zero
/// request never pays for classification it did not need.
pub fn route(query: &str, cache_hit: bool, models: &TierModels) -> Route {
    if cache_hit {
        return Route {
            tier: RouteTier::Zero,
            model: None,
            reason: "cache_hit",
            context_cap: 0,
        };
    }
    if TRIVIAL.is_match(query) {
        return Route {
            tier: RouteTier::Zero,
            model: None,
            reason: "trivial_no_llm",
            context_cap: 0,
        };
    }
    if NEEDS_REASONING.is_match(query) || query.split_whitespace().count() > 60 {
        return Route {
            tier: RouteTier::Two,
            model: Some(models.escalated.clone()),
            reason: "reasoning_or_long",
            // A tighter cap on the expensive tier: it is the tier that can
            // actually breach the budget.
            context_cap: 8,
        };
    }
    Route {
        tier: RouteTier::One,
        model: Some(models.standard.clone()),
        reason: "standard",
        context_cap: 12,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(q: &str) -> Route {
        route(q, false, &TierModels::default())
    }

    #[test]
    fn a_cache_hit_never_reaches_a_model() {
        let x = route("anything at all", true, &TierModels::default());
        assert_eq!(x.tier, RouteTier::Zero);
        assert_eq!(x.model, None);
        assert_eq!(x.context_cap, 0);
    }

    #[test]
    fn pleasantries_do_not_cost_a_model_call() {
        for q in ["hi", "thanks", "  Hello ", "ok."] {
            assert_eq!(r(q).tier, RouteTier::Zero, "{q:?} should be tier zero");
        }
    }

    #[test]
    fn an_ordinary_how_do_i_question_goes_to_the_cheap_model() {
        let x = r("how do I restart nginx on ubuntu");
        assert_eq!(x.tier, RouteTier::One);
        assert_eq!(x.model.as_deref(), Some("claude-haiku-4-5"));
    }

    #[test]
    fn diagnostic_and_comparative_questions_escalate() {
        for q in [
            "why is our latency spiking after the upgrade",
            "should we migrate from apache to nginx",
            "root cause of the intermittent 502s",
            "compare postgres and mysql replication",
        ] {
            assert_eq!(r(q).tier, RouteTier::Two, "{q:?} should escalate");
        }
    }

    #[test]
    fn a_very_long_request_escalates_on_length_alone() {
        let long = "word ".repeat(80);
        assert_eq!(r(&long).tier, RouteTier::Two);
    }

    #[test]
    fn the_expensive_tier_gets_the_tighter_context_cap() {
        // Counter-intuitive but deliberate: tier two is the only tier that can
        // breach the budget, so it is the one that must be capped hardest.
        assert!(r("why is this slow").context_cap < r("how do I restart nginx").context_cap);
    }

    #[test]
    fn every_route_carries_a_reason_for_telemetry() {
        for q in ["hi", "how do I restart nginx", "why is this failing"] {
            assert!(!r(q).reason.is_empty());
        }
        assert_eq!(r("hi").reason, "trivial_no_llm");
        assert_eq!(route("x", true, &TierModels::default()).reason, "cache_hit");
    }

    #[test]
    fn the_default_models_are_the_ones_that_fit_the_budget() {
        // Opus is deliberately absent: at 20K context it is 2x over budget.
        let m = TierModels::default();
        assert_eq!(m.standard, "claude-haiku-4-5");
        assert_eq!(m.escalated, "claude-sonnet-5");
    }
}
