//! Guards that decide whether a query may be answered from a semantic cache.
//!
//! A semantic cache converts a cost problem into a **silent correctness
//! problem**, and that is the whole design consideration. Two queries whose
//! embeddings sit above 0.95 cosine similarity can mean opposite things:
//!
//! * *"disk usage in Q1 2024"* vs *"disk usage in Q1 2025"* — different period
//! * *"why did latency increase"* vs *"why did latency decrease"* — opposite polarity
//! * *"how many servers"* vs *"which servers"* — different intent
//!
//! Embedding models are trained to place these near each other, because they
//! *are* topically near. Relevance and equivalence are not the same relation,
//! and a semantic cache silently treats the first as the second.
//!
//! So rather than tune the threshold until the errors are rare enough to ignore,
//! queries carrying a discriminator the embedding is known to under-weight are
//! excluded from semantic lookup entirely. They can still hit the exact cache,
//! where equality is equality.

use once_cell::sync::Lazy;
use regex::Regex;

/// Why a query was refused a semantic cache lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardReason {
    /// Contains a year, date, quarter or relative time reference.
    Temporal,
    /// Contains a number, version, port or size that changes the answer.
    Numeric,
    /// Contains a negation, which embeddings represent weakly.
    Negation,
    /// Comparative or superlative framing, where the answer depends on a set
    /// the cache cannot know is unchanged.
    Comparative,
}

impl GuardReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            GuardReason::Temporal => "temporal",
            GuardReason::Numeric => "numeric",
            GuardReason::Negation => "negation",
            GuardReason::Comparative => "comparative",
        }
    }
}

static TEMPORAL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)
        \b(19|20)\d{2}\b            # a year
        | \bq[1-4]\b                # a quarter
        | \b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b
        | \b(today|yesterday|tomorrow|now|current|currently|latest|recent|recently)\b
        | \b(last|next|past|this)\s+(week|month|year|quarter|day|hour)\b
        | \b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b
        ",
    )
    .expect("static regex")
});

static NUMERIC: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)
        \bv?\d+\.\d+(\.\d+)?\b      # a version
        | \b\d+\s*(gb|mb|tb|kb|gib|mib|ms|sec|seconds|minutes|hours|%)\b
        | \bport\s*\d+\b
        | \b\d{3,}\b                # a large bare number
        ",
    )
    .expect("static regex")
});

static NEGATION: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)\b(not|no|never|without|except|exclude|excluding|cannot|can't|won't|doesn't|isn't|aren't|didn't|unable|fail|failed|failing)\b",
    )
    .expect("static regex")
});

static COMPARATIVE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)\b(more|less|fewer|greater|higher|highest|lower|lowest|better|best|worse|worst|most|least|fastest|slowest|largest|smallest|biggest|top|compare|comparison|versus|vs|difference between)\b",
    )
    .expect("static regex")
});

/// Whether `q` may be served from a semantic cache, and why not if not.
///
/// Conservative by construction: a false refusal costs one model call, a false
/// acceptance returns a confidently wrong answer.
pub fn semantic_lookup_allowed(q: &str) -> Result<(), GuardReason> {
    if TEMPORAL.is_match(q) {
        return Err(GuardReason::Temporal);
    }
    if NEGATION.is_match(q) {
        return Err(GuardReason::Negation);
    }
    if COMPARATIVE.is_match(q) {
        return Err(GuardReason::Comparative);
    }
    if NUMERIC.is_match(q) {
        return Err(GuardReason::Numeric);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(q: &str) -> Option<GuardReason> {
        semantic_lookup_allowed(q).err()
    }

    #[test]
    fn ordinary_operational_questions_are_allowed() {
        for q in [
            "how do I restart nginx",
            "what causes a bad gateway error",
            "configure apache virtual hosts",
            "explain systemd unit files",
        ] {
            assert_eq!(semantic_lookup_allowed(q), Ok(()), "{q} should be cacheable");
        }
    }

    #[test]
    fn queries_differing_only_by_year_are_refused() {
        // The canonical false hit: these two embed almost identically and have
        // different answers.
        assert_eq!(refused("disk usage report for Q1 2024"), Some(GuardReason::Temporal));
        assert_eq!(refused("disk usage report for Q1 2025"), Some(GuardReason::Temporal));
    }

    #[test]
    fn relative_time_references_are_refused() {
        for q in ["what is the current kernel version", "errors from last week", "latest release"] {
            assert!(refused(q).is_some(), "{q} should be refused");
        }
    }

    #[test]
    fn polarity_is_refused_because_embeddings_represent_it_weakly() {
        assert_eq!(refused("why is the service not starting"), Some(GuardReason::Negation));
        assert_eq!(refused("servers without monitoring"), Some(GuardReason::Negation));
    }

    #[test]
    fn comparatives_are_refused_because_the_answer_depends_on_a_changing_set() {
        assert_eq!(refused("which server has the highest load"), Some(GuardReason::Comparative));
        assert_eq!(refused("nginx versus apache performance"), Some(GuardReason::Comparative));
    }

    #[test]
    fn versions_and_sizes_are_refused() {
        assert_eq!(refused("upgrade to postgres 14.2"), Some(GuardReason::Numeric));
        assert_eq!(refused("allocate 512 GB of storage"), Some(GuardReason::Numeric));
        assert_eq!(refused("is port 8080 open"), Some(GuardReason::Numeric));
    }

    #[test]
    fn guards_are_checked_in_a_stable_order() {
        // A query can trip several guards; the reported reason must be
        // deterministic or the telemetry is not aggregatable.
        let q = "did not upgrade to version 2.1 in 2024";
        assert_eq!(refused(q), Some(GuardReason::Temporal));
        assert_eq!(refused(q), Some(GuardReason::Temporal));
    }

    #[test]
    fn a_short_bare_number_does_not_trip_the_numeric_guard() {
        // Over-refusing costs a model call on every query mentioning a small
        // number, which would gut the hit rate for no correctness gain.
        assert_eq!(semantic_lookup_allowed("how do I add 2 disks to a pool"), Ok(()));
    }
}
