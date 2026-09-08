//! Query, retrieval results, and the route tiers that enforce the cost budget.

use crate::chunk::ChunkId;
use serde::{Deserialize, Serialize};

/// Which cost tier a request was routed to.
///
/// The tiers exist because a 6-cent p95 budget cannot survive sending every
/// request to a frontier model. Tier 0 is the one that never calls an LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteTier {
    /// Served from cache or a deterministic template. No LLM call.
    Zero,
    /// Single agent with tools on a small model, hard context cap.
    One,
    /// Escalation to a stronger model, still bounded.
    Two,
}

impl RouteTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            RouteTier::Zero => "tier_0",
            RouteTier::One => "tier_1",
            RouteTier::Two => "tier_2",
        }
    }
}

/// A user request, scoped to a tenant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    pub text: String,
    /// From a signed token. Never model-controlled.
    pub tenant_id: String,
    /// Optional tag pre-filter. Note that filters matching 5-25% of the corpus
    /// sit in Qdrant's filterable-HNSW danger zone and must be measured.
    pub tag_filter: Vec<String>,
    pub top_k: usize,
}

impl Query {
    pub fn new(text: impl Into<String>, tenant_id: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tenant_id: tenant_id.into(),
            tag_filter: Vec::new(),
            // Retrieve deep, pass few. Recall@5 climbs 0.458 -> 0.888 going
            // from 20 to 100 candidates, so a shallow candidate pool caps
            // quality no matter how good the reranker is.
            top_k: 100,
        }
    }
}

/// One retrieved candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub chunk_id: ChunkId,
    pub score: f32,
    pub doc_id: String,
    pub display_text: String,
    pub source_uri: String,
    pub section_path: Vec<String>,
    /// Which retriever(s) surfaced this, for fusion diagnostics.
    pub retrievers: Vec<Retriever>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Retriever {
    Dense,
    Sparse,
}

/// A citation attached to a generated answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    pub chunk_id: ChunkId,
    pub source_uri: String,
    /// The span of the answer this citation supports.
    pub claim: String,
}

/// Reciprocal rank fusion.
///
/// `k` damps the contribution of top ranks. Qdrant's Query API defaults to
/// k=2 rather than the k=60 common in the literature; the right value depends
/// on relevance density (low k when there is roughly one relevant document per
/// query, higher when there are many), so it stays a parameter and gets tuned
/// against the golden set rather than guessed.
pub fn rrf_fuse(lists: &[Vec<ChunkId>], k: f32) -> Vec<(ChunkId, f32)> {
    use std::collections::HashMap;
    let mut scores: HashMap<ChunkId, f32> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *scores.entry(*id).or_default() += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut v: Vec<(ChunkId, f32)> = scores.into_iter().collect();
    // Sort by score desc, then by id for a deterministic order under ties.
    // Ties are common at low k (roughly an eighth of a top-10 at k=2), so an
    // unstable order here would make eval results irreproducible.
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{ChunkId, ContentHash};

    fn id(n: u128) -> ChunkId {
        ChunkId(n)
    }

    #[test]
    fn default_top_k_retrieves_deep_enough_to_rerank() {
        let q = Query::new("nginx 502", "acme");
        assert_eq!(q.top_k, 100, "a 20-candidate pool caps recall@5 near 0.46");
    }

    #[test]
    fn rrf_rewards_documents_found_by_both_retrievers() {
        let dense = vec![id(1), id(2), id(3)];
        let sparse = vec![id(3), id(2), id(9)];
        let fused = rrf_fuse(&[dense, sparse], 2.0);
        // id(2) and id(3) appear in both lists; id(1) and id(9) in one each.
        let top_two: Vec<ChunkId> = fused.iter().take(2).map(|(i, _)| *i).collect();
        assert!(top_two.contains(&id(2)));
        assert!(top_two.contains(&id(3)));
    }

    #[test]
    fn rrf_is_deterministic_under_ties() {
        // Two symmetric lists produce tied scores; the order must still be
        // stable or eval numbers stop being reproducible.
        let a = vec![id(10), id(20)];
        let b = vec![id(20), id(10)];
        let first = rrf_fuse(&[a.clone(), b.clone()], 2.0);
        let second = rrf_fuse(&[a, b], 2.0);
        assert_eq!(first, second);
        assert!((first[0].1 - first[1].1).abs() < 1e-6, "scores are tied");
        assert!(first[0].0 < first[1].0, "tie broken by id, ascending");
    }

    #[test]
    fn rrf_k_changes_how_sharply_top_ranks_dominate() {
        let dense = vec![id(1), id(2)];
        let sparse = vec![id(2), id(1)];
        let low = rrf_fuse(&[dense.clone(), sparse.clone()], 2.0);
        let high = rrf_fuse(&[dense, sparse], 60.0);
        let spread = |v: &Vec<(ChunkId, f32)>| v[0].1 - v[v.len() - 1].1;
        assert!(spread(&low) >= spread(&high), "low k spreads scores wider");
    }

    #[test]
    fn empty_input_fuses_to_empty_rather_than_panicking() {
        assert!(rrf_fuse(&[], 2.0).is_empty());
        assert!(rrf_fuse(&[vec![]], 2.0).is_empty());
    }

    #[test]
    fn single_list_preserves_its_own_ranking() {
        let only = vec![id(5), id(6), id(7)];
        let fused: Vec<ChunkId> = rrf_fuse(&[only.clone()], 2.0).into_iter().map(|(i, _)| i).collect();
        assert_eq!(fused, only);
    }

    #[test]
    fn route_tiers_render_stable_labels_for_dashboards() {
        assert_eq!(RouteTier::Zero.as_str(), "tier_0");
        assert_eq!(RouteTier::One.as_str(), "tier_1");
        assert_eq!(RouteTier::Two.as_str(), "tier_2");
    }

    #[test]
    fn candidate_records_which_retriever_found_it() {
        let c = Candidate {
            chunk_id: ChunkId::derive("q1", 0, &ContentHash::of("x")),
            score: 0.9,
            doc_id: "q1".into(),
            display_text: "restart nginx".into(),
            source_uri: "https://serverfault.com/q/1".into(),
            section_path: vec![],
            retrievers: vec![Retriever::Dense, Retriever::Sparse],
        };
        assert_eq!(c.retrievers.len(), 2);
    }
}
