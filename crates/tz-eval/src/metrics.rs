//! Retrieval metrics.
//!
//! Implemented here rather than imported. They are short, and owning them means
//! the definitions are inspectable: every number this project publishes can be
//! traced to the exact arithmetic that produced it, which is not true of a
//! metric pulled from a framework and quoted by name.
//!
//! A note on reporting: recall and latency are always published as a pair. A
//! latency figure without its recall is not a result, because any retriever can
//! be made arbitrarily fast by returning less.

/// Did any relevant document appear in the top k?
///
/// With one relevant document per query -- which is what a duplicate judgement
/// gives -- recall@k, hit-rate@k and success@k all coincide. Naming it recall
/// keeps it comparable to the literature.
pub fn recall_at_k(retrieved: &[u64], relevant: &[u64], k: usize) -> f64 {
    if relevant.is_empty() {
        return f64::NAN;
    }
    let top: &[u64] = &retrieved[..k.min(retrieved.len())];
    let hits = relevant.iter().filter(|r| top.contains(r)).count();
    hits as f64 / relevant.len() as f64
}

/// Reciprocal of the rank of the first relevant document; 0 if absent.
///
/// Sensitive to where in the list the answer landed, which recall@k is not.
pub fn reciprocal_rank(retrieved: &[u64], relevant: &[u64]) -> f64 {
    retrieved
        .iter()
        .position(|d| relevant.contains(d))
        .map(|i| 1.0 / (i + 1) as f64)
        .unwrap_or(0.0)
}

/// Discounted cumulative gain at k, using binary gain unless graded gains are
/// supplied.
fn dcg(gains: &[f64]) -> f64 {
    gains
        .iter()
        .enumerate()
        .map(|(i, g)| g / ((i + 2) as f64).log2())
        .sum()
}

/// nDCG@k with per-document graded relevance.
///
/// `gain_of` returns the gain for a retrieved document id, 0 for irrelevant.
/// The ideal ranking is computed from the gains actually available, so a query
/// whose relevant document is missing from the corpus scores 0 rather than
/// dividing by zero.
pub fn ndcg_at_k<F: Fn(u64) -> f64>(
    retrieved: &[u64],
    ideal_gains: &[f64],
    k: usize,
    gain_of: F,
) -> f64 {
    let top: Vec<f64> =
        retrieved.iter().take(k).map(|d| gain_of(*d)).collect();
    let mut ideal: Vec<f64> = ideal_gains.to_vec();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    ideal.truncate(k);
    let idcg = dcg(&ideal);
    if idcg == 0.0 {
        return 0.0;
    }
    (dcg(&top) / idcg).clamp(0.0, 1.0)
}

/// Binary-relevance nDCG@k, the common case for duplicate judgements.
pub fn ndcg_at_k_binary(retrieved: &[u64], relevant: &[u64], k: usize) -> f64 {
    let ideal: Vec<f64> = vec![1.0; relevant.len()];
    ndcg_at_k(retrieved, &ideal, k, |d| if relevant.contains(&d) { 1.0 } else { 0.0 })
}

/// Aggregate over a set of queries.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RetrievalReport {
    pub queries: usize,
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub recall_at_50: f64,
    pub recall_at_100: f64,
    pub mrr: f64,
    pub ndcg_at_10: f64,
    /// Queries where the gold document was not retrieved at any depth. These
    /// are the interesting ones: aggregate metrics hide them, and they are
    /// where the failure-mode catalogue comes from.
    pub misses: Vec<u64>,
}

/// One query's outcome: what was retrieved, and what should have been.
pub struct QueryOutcome {
    pub query_id: u64,
    pub retrieved: Vec<u64>,
    pub relevant: Vec<u64>,
}

pub fn aggregate(outcomes: &[QueryOutcome]) -> RetrievalReport {
    if outcomes.is_empty() {
        return RetrievalReport::default();
    }
    let n = outcomes.len() as f64;
    let mut r = RetrievalReport { queries: outcomes.len(), ..Default::default() };
    for o in outcomes {
        r.recall_at_1 += recall_at_k(&o.retrieved, &o.relevant, 1);
        r.recall_at_5 += recall_at_k(&o.retrieved, &o.relevant, 5);
        r.recall_at_10 += recall_at_k(&o.retrieved, &o.relevant, 10);
        r.recall_at_50 += recall_at_k(&o.retrieved, &o.relevant, 50);
        r.recall_at_100 += recall_at_k(&o.retrieved, &o.relevant, 100);
        r.mrr += reciprocal_rank(&o.retrieved, &o.relevant);
        r.ndcg_at_10 += ndcg_at_k_binary(&o.retrieved, &o.relevant, 10);
        if reciprocal_rank(&o.retrieved, &o.relevant) == 0.0 {
            r.misses.push(o.query_id);
        }
    }
    for v in [
        &mut r.recall_at_1,
        &mut r.recall_at_5,
        &mut r.recall_at_10,
        &mut r.recall_at_50,
        &mut r.recall_at_100,
        &mut r.mrr,
        &mut r.ndcg_at_10,
    ] {
        *v /= n;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_is_one_when_the_gold_document_is_in_range() {
        assert_eq!(recall_at_k(&[9, 8, 7, 42], &[42], 5), 1.0);
    }

    #[test]
    fn recall_is_zero_when_the_gold_document_falls_outside_k() {
        // The document is retrieved, but not within the cutoff. This is the
        // case that motivates retrieving 100 and reranking rather than
        // retrieving 5.
        assert_eq!(recall_at_k(&[9, 8, 7, 42], &[42], 3), 0.0);
        assert_eq!(recall_at_k(&[9, 8, 7, 42], &[42], 4), 1.0);
    }

    #[test]
    fn recall_with_several_relevant_documents_is_a_fraction() {
        assert_eq!(recall_at_k(&[1, 2, 9], &[1, 2, 3, 4], 10), 0.5);
    }

    #[test]
    fn reciprocal_rank_rewards_ranking_the_answer_first() {
        assert_eq!(reciprocal_rank(&[42, 1, 2], &[42]), 1.0);
        assert_eq!(reciprocal_rank(&[1, 42, 2], &[42]), 0.5);
        assert!((reciprocal_rank(&[1, 2, 42], &[42]) - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn reciprocal_rank_is_zero_on_a_complete_miss() {
        assert_eq!(reciprocal_rank(&[1, 2, 3], &[42]), 0.0);
    }

    #[test]
    fn ndcg_is_one_for_a_perfect_ranking_and_less_otherwise() {
        assert!((ndcg_at_k_binary(&[42], &[42], 10) - 1.0).abs() < 1e-9);
        let worse = ndcg_at_k_binary(&[1, 2, 42], &[42], 10);
        assert!(worse < 1.0 && worse > 0.0, "got {worse}");
    }

    #[test]
    fn ndcg_ranks_earlier_hits_above_later_ones() {
        let early = ndcg_at_k_binary(&[42, 1, 2, 3], &[42], 10);
        let late = ndcg_at_k_binary(&[1, 2, 3, 42], &[42], 10);
        assert!(early > late);
    }

    #[test]
    fn ndcg_never_exceeds_one_even_with_graded_gains() {
        let g = ndcg_at_k(&[1, 2, 3], &[3.0, 2.0, 1.0], 3, |d| match d {
            1 => 3.0,
            2 => 2.0,
            3 => 1.0,
            _ => 0.0,
        });
        assert!((g - 1.0).abs() < 1e-9, "got {g}");
    }

    #[test]
    fn a_query_with_no_relevant_documents_is_nan_not_zero() {
        // Silently scoring 0 would drag an aggregate down for a query that
        // should have been excluded from the set entirely.
        assert!(recall_at_k(&[1, 2], &[], 5).is_nan());
    }

    #[test]
    fn ndcg_is_zero_rather_than_nan_when_nothing_is_retrievable() {
        assert_eq!(ndcg_at_k_binary(&[1, 2], &[], 10), 0.0);
    }

    #[test]
    fn aggregate_reports_per_cutoff_and_records_misses() {
        let outcomes = vec![
            QueryOutcome { query_id: 1, retrieved: vec![42, 1, 2], relevant: vec![42] },
            QueryOutcome { query_id: 2, retrieved: vec![1, 2, 3], relevant: vec![99] },
        ];
        let r = aggregate(&outcomes);
        assert_eq!(r.queries, 2);
        assert_eq!(r.recall_at_1, 0.5, "one of two ranked first");
        assert_eq!(r.recall_at_10, 0.5);
        assert_eq!(r.mrr, 0.5);
        assert_eq!(r.misses, vec![2], "the missed query is named, not just counted");
    }

    #[test]
    fn recall_is_monotonic_in_k() {
        let ret: Vec<u64> = (0..100).collect();
        let rel = vec![73];
        let mut prev = 0.0;
        for k in [1usize, 5, 10, 50, 100] {
            let r = recall_at_k(&ret, &rel, k);
            assert!(r >= prev, "recall@{k}={r} dropped below {prev}");
            prev = r;
        }
        assert_eq!(prev, 1.0);
    }

    #[test]
    fn empty_aggregate_does_not_divide_by_zero() {
        let r = aggregate(&[]);
        assert_eq!(r.queries, 0);
        assert!(!r.mrr.is_nan());
    }
}
