//! The cache itself: exact first, semantic second, model last.
//!
//! Ordering matters. Exact matching is free and cannot be wrong, so it runs
//! first and takes as much traffic as it can. Semantic matching runs only on
//! what exact matching missed *and* the guards permit.
//!
//! Every entry is namespaced by tenant. This is not tidiness — a cross-tenant
//! cache hit is a data breach, and it is the kind that leaves no trace in the
//! vector store's own access logs because the store was never consulted.

use crate::guard::{GuardReason, semantic_lookup_allowed};
use std::collections::HashMap;
use tz_embed::cosine;

/// Exact-match key: tenant plus a normalized form of the query.
///
/// Normalization is limited to case and whitespace. It deliberately does not
/// strip punctuation or stopwords: "restart nginx" and "restart nginx?" are the
/// same request, but "can I restart nginx" is not, and aggressive normalization
/// is how an exact cache quietly becomes a fuzzy one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey(String);

impl CacheKey {
    pub fn new(tenant: &str, query: &str) -> Self {
        let norm = query.trim().to_lowercase();
        let collapsed: Vec<&str> = norm.split_whitespace().collect();
        let mut h = blake3::Hasher::new();
        h.update(tenant.as_bytes());
        h.update(b"\x00");
        h.update(collapsed.join(" ").as_bytes());
        CacheKey(h.finalize().to_hex().to_string())
    }
}

/// What a cache lookup produced.
#[derive(Debug, Clone, PartialEq)]
pub enum Lookup {
    Exact(String),
    /// A semantic hit, with the similarity that produced it. The score is
    /// retained so hits can be sampled and reviewed after the fact — the only
    /// evidence available once a wrong answer has been served.
    Semantic { answer: String, similarity: f32, matched_query: String },
    /// No hit. `refused` is set when the guards blocked a semantic lookup that
    /// might otherwise have been attempted, so the refusal rate is observable
    /// rather than hidden inside the miss rate.
    Miss { refused: Option<GuardReason> },
}

struct Entry {
    query: String,
    embedding: Vec<f32>,
    answer: String,
}

/// In-process cache. A production deployment would put this in Redis; the
/// semantics are what matter here, not the storage.
pub struct CacheStore {
    exact: HashMap<CacheKey, String>,
    /// Semantic entries, per tenant. Keyed by tenant so a scan can never cross
    /// the boundary even by accident.
    semantic: HashMap<String, Vec<Entry>>,
    threshold: f32,
    stats: CacheStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub lookups: u64,
    pub exact_hits: u64,
    pub semantic_hits: u64,
    pub misses: u64,
    pub refused: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        (self.exact_hits + self.semantic_hits) as f64 / self.lookups as f64
    }
}

impl CacheStore {
    /// `threshold` is the cosine similarity above which a semantic hit is
    /// accepted. 0.95 is deliberately stricter than the 0.80-0.85 commonly
    /// suggested: the guards already remove the query classes where similarity
    /// is most misleading, so the remaining traffic can afford a high bar, and
    /// the cost of a wrong answer here is a confidently wrong ops instruction.
    pub fn new(threshold: f32) -> Self {
        Self {
            exact: HashMap::new(),
            semantic: HashMap::new(),
            threshold,
            stats: CacheStats::default(),
        }
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }
    pub fn stats(&self) -> CacheStats {
        self.stats
    }
    pub fn len(&self) -> usize {
        self.exact.len()
    }
    pub fn is_empty(&self) -> bool {
        self.exact.is_empty()
    }

    pub fn insert(&mut self, tenant: &str, query: &str, embedding: Vec<f32>, answer: String) {
        self.exact.insert(CacheKey::new(tenant, query), answer.clone());
        self.semantic.entry(tenant.to_string()).or_default().push(Entry {
            query: query.to_string(),
            embedding,
            answer,
        });
    }

    /// Look up an answer. `embedding` may be `None` to skip semantic matching.
    pub fn get(&mut self, tenant: &str, query: &str, embedding: Option<&[f32]>) -> Lookup {
        self.stats.lookups += 1;

        if let Some(a) = self.exact.get(&CacheKey::new(tenant, query)) {
            self.stats.exact_hits += 1;
            return Lookup::Exact(a.clone());
        }

        if let Err(reason) = semantic_lookup_allowed(query) {
            self.stats.misses += 1;
            self.stats.refused += 1;
            return Lookup::Miss { refused: Some(reason) };
        }

        let Some(emb) = embedding else {
            self.stats.misses += 1;
            return Lookup::Miss { refused: None };
        };

        // Only this tenant's entries are ever considered.
        let mut best: Option<(f32, &Entry)> = None;
        for e in self.semantic.get(tenant).into_iter().flatten() {
            let s = cosine(emb, &e.embedding);
            if best.as_ref().is_none_or(|(b, _)| s > *b) {
                best = Some((s, e));
            }
        }

        match best {
            Some((s, e)) if s >= self.threshold => {
                self.stats.semantic_hits += 1;
                Lookup::Semantic {
                    answer: e.answer.clone(),
                    similarity: s,
                    matched_query: e.query.clone(),
                }
            }
            _ => {
                self.stats.misses += 1;
                Lookup::Miss { refused: None }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(v: &[f32]) -> Vec<f32> {
        v.to_vec()
    }

    #[test]
    fn an_exact_repeat_hits_without_needing_an_embedding() {
        let mut c = CacheStore::new(0.95);
        c.insert("acme", "how do I restart nginx", emb(&[1.0, 0.0]), "systemctl restart nginx".into());
        assert_eq!(
            c.get("acme", "how do I restart nginx", None),
            Lookup::Exact("systemctl restart nginx".into())
        );
        assert_eq!(c.stats().exact_hits, 1);
    }

    #[test]
    fn exact_matching_ignores_case_and_whitespace_but_not_wording() {
        let mut c = CacheStore::new(0.95);
        c.insert("acme", "restart nginx", emb(&[1.0, 0.0]), "answer".into());
        assert!(matches!(c.get("acme", "  RESTART   nginx ", None), Lookup::Exact(_)));
        // Different wording is a different request, not a normalization case.
        assert!(matches!(c.get("acme", "can I restart nginx", None), Lookup::Miss { .. }));
    }

    #[test]
    fn a_near_identical_query_hits_semantically() {
        let mut c = CacheStore::new(0.95);
        c.insert("acme", "how do I restart the web server", emb(&[1.0, 0.0]), "answer".into());
        match c.get("acme", "how do I reboot the web server", Some(&[0.99, 0.14])) {
            Lookup::Semantic { similarity, .. } => assert!(similarity >= 0.95),
            other => panic!("expected a semantic hit, got {other:?}"),
        }
    }

    #[test]
    fn a_merely_related_query_does_not_hit() {
        let mut c = CacheStore::new(0.95);
        c.insert("acme", "how do I restart the web server", emb(&[1.0, 0.0]), "answer".into());
        assert!(matches!(
            c.get("acme", "how do I configure a load balancer", Some(&[0.5, 0.87])),
            Lookup::Miss { refused: None }
        ));
    }

    #[test]
    fn a_cache_entry_never_crosses_a_tenant_boundary() {
        // The failure this prevents leaves no trace in the vector store's logs,
        // because the store is never consulted on a cache hit.
        let mut c = CacheStore::new(0.95);
        c.insert("acme", "what is our root password policy", emb(&[1.0, 0.0]), "acme secret".into());
        assert!(matches!(
            c.get("globex", "what is our root password policy", Some(&[1.0, 0.0])),
            Lookup::Miss { .. }
        ));
        assert_eq!(c.stats().exact_hits, 0);
        assert_eq!(c.stats().semantic_hits, 0);
    }

    #[test]
    fn guarded_queries_are_refused_semantic_lookup_even_at_perfect_similarity() {
        // The headline case: identical embeddings, different years, different
        // answers. Similarity alone would serve the wrong one.
        let mut c = CacheStore::new(0.95);
        c.insert("acme", "disk usage in Q1 2024", emb(&[1.0, 0.0]), "2024 answer".into());
        let r = c.get("acme", "disk usage in Q1 2025", Some(&[1.0, 0.0]));
        assert_eq!(r, Lookup::Miss { refused: Some(GuardReason::Temporal) });
        assert_eq!(c.stats().refused, 1);
    }

    #[test]
    fn a_guarded_query_can_still_hit_the_exact_cache() {
        // Guards restrict *similarity* matching. Equality is still equality,
        // and refusing an exact repeat would throw away free hits.
        let mut c = CacheStore::new(0.95);
        c.insert("acme", "disk usage in Q1 2024", emb(&[1.0, 0.0]), "2024 answer".into());
        assert_eq!(
            c.get("acme", "disk usage in Q1 2024", Some(&[1.0, 0.0])),
            Lookup::Exact("2024 answer".into())
        );
    }

    #[test]
    fn refusals_are_counted_separately_from_ordinary_misses() {
        let mut c = CacheStore::new(0.95);
        c.get("acme", "errors from last week", Some(&[1.0, 0.0]));
        c.get("acme", "how do I restart nginx", Some(&[1.0, 0.0]));
        let s = c.stats();
        assert_eq!(s.misses, 2);
        assert_eq!(s.refused, 1, "a guard refusal is not the same event as a cache miss");
    }

    #[test]
    fn hit_rate_is_zero_rather_than_nan_before_any_lookup() {
        assert_eq!(CacheStore::new(0.95).stats().hit_rate(), 0.0);
    }

    #[test]
    fn the_threshold_is_stricter_than_the_common_recommendation() {
        // 0.80-0.85 is widely suggested; the guards let us afford a higher bar,
        // and the cost of a wrong ops instruction is not symmetric with a miss.
        assert!(CacheStore::new(0.95).threshold() >= 0.90);
    }
}
