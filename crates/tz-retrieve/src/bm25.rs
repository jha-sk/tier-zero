//! BM25 sparse vectors for hybrid retrieval.
//!
//! Built because of what the `hnsw_ef` sweep showed: recall was **identical**
//! at ef 32/64/128/256, which means the approximate index was not costing
//! anything and further graph tuning could not help. The recall gap is lexical
//! -- dense embeddings miss exact identifiers, error strings, flags and command
//! names, which is most of what an operations corpus is made of. A query for
//! `CrashLoopBackOff` or `-Xmx` needs a retriever that can match the literal
//! token.
//!
//! Scoring is split the way Qdrant's sparse dot product requires: the document
//! vector carries the term-frequency component, the query vector carries the
//! IDF. Their dot product over shared terms is then exactly the BM25 score.

use std::collections::HashMap;

/// Term-frequency saturation. Higher means repeated terms keep adding signal.
pub const K1: f32 = 1.2;
/// Length normalization strength.
pub const B: f32 = 0.75;

/// Words carrying no retrieval signal. Deliberately short: an aggressive list
/// removes terms that are discriminating in a technical corpus ("no" in "no
/// route to host", "up" in "link is up").
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on",
    "with", "as", "at", "by", "be", "are", "was", "this", "that", "from",
];

/// Split text into index terms.
///
/// Keeps `.`, `-` and `_` inside tokens, because `2.4.1`, `--max-old-space-size`
/// and `wp_options` are single identifiers in this corpus and splitting them
/// destroys exactly the lexical signal sparse retrieval exists to capture.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        let keep = ch.is_ascii_alphanumeric()
            || ((ch == '.' || ch == '-' || ch == '_') && !cur.is_empty());
        if keep {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            push_token(&mut out, std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        push_token(&mut out, cur);
    }
    out
}

fn push_token(out: &mut Vec<String>, mut t: String) {
    // Trailing punctuation kept inside a token by the rule above.
    while t.ends_with('.') || t.ends_with('-') || t.ends_with('_') {
        t.pop();
    }
    if t.is_empty() || t.len() > 64 {
        return;
    }
    if STOPWORDS.contains(&t.as_str()) {
        return;
    }
    out.push(t);
}

/// Stable term id. Qdrant sparse vectors are indexed by `u32`.
///
/// Hashing rather than a stored vocabulary means query and ingest agree without
/// shipping a dictionary, at the cost of rare collisions: two unrelated terms
/// sharing an id add a little noise to one another's scores. At 32 bits over a
/// vocabulary of this size that is a negligible trade for not having to version
/// and distribute a vocabulary alongside the index.
pub fn term_id(term: &str) -> u32 {
    let h = blake3::hash(term.as_bytes());
    u32::from_le_bytes(h.as_bytes()[..4].try_into().expect("4 bytes"))
}

/// Corpus statistics needed to score. Accumulated in one pass over the chunks.
#[derive(Debug, Default, Clone)]
pub struct Bm25Stats {
    /// Documents containing each term.
    pub doc_freq: HashMap<u32, u64>,
    pub doc_count: u64,
    total_len: u64,
}

impl Bm25Stats {
    pub fn observe(&mut self, tokens: &[String]) {
        self.doc_count += 1;
        self.total_len += tokens.len() as u64;
        let mut seen = std::collections::HashSet::new();
        for t in tokens {
            let id = term_id(t);
            if seen.insert(id) {
                *self.doc_freq.entry(id).or_default() += 1;
            }
        }
    }

    pub fn avg_doc_len(&self) -> f32 {
        if self.doc_count == 0 {
            return 1.0;
        }
        (self.total_len as f32 / self.doc_count as f32).max(1.0)
    }

    /// Probabilistic IDF with the +1 smoothing that keeps it non-negative.
    ///
    /// Without the smoothing, a term appearing in more than half the corpus
    /// gets a *negative* weight and actively penalises documents containing it.
    pub fn idf(&self, term_id: u32) -> f32 {
        let df = self.doc_freq.get(&term_id).copied().unwrap_or(0) as f32;
        let n = self.doc_count as f32;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }
}

impl Bm25Stats {
    /// Load statistics persisted at index time.
    ///
    /// Query-side IDF must come from the *indexed* corpus. Recomputing it from
    /// anything else weights terms against a different document population and
    /// silently changes what the scores mean.
    pub fn load(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let doc_count = v["doc_count"].as_u64().unwrap_or(0);
        let avg = v["avg_doc_len"].as_f64().unwrap_or(1.0) as f32;
        let mut doc_freq = HashMap::new();
        if let Some(m) = v["doc_freq"].as_object() {
            for (k, val) in m {
                if let (Ok(id), Some(n)) = (k.parse::<u32>(), val.as_u64()) {
                    doc_freq.insert(id, n);
                }
            }
        }
        Ok(Self { doc_freq, doc_count, total_len: (avg as f64 * doc_count as f64) as u64 })
    }
}

/// A sparse vector as Qdrant expects it.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseVec {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVec {
    pub fn len(&self) -> usize {
        self.indices.len()
    }
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Document-side vector: the term-frequency component of BM25.
///
/// IDF is deliberately *not* applied here. It lives on the query side so that
/// corpus statistics can change without rewriting every stored vector.
pub fn document_vector(tokens: &[String], stats: &Bm25Stats) -> SparseVec {
    let mut tf: HashMap<u32, f32> = HashMap::new();
    for t in tokens {
        *tf.entry(term_id(t)).or_default() += 1.0;
    }
    let len_norm = 1.0 - B + B * (tokens.len() as f32 / stats.avg_doc_len());
    let mut indices = Vec::with_capacity(tf.len());
    let mut values = Vec::with_capacity(tf.len());
    for (id, f) in tf {
        indices.push(id);
        values.push((f * (K1 + 1.0)) / (f + K1 * len_norm));
    }
    SparseVec { indices, values }
}

/// Query-side vector: IDF weights.
///
/// The dot product of this with a document vector is the BM25 score.
pub fn query_vector(tokens: &[String], stats: &Bm25Stats) -> SparseVec {
    let mut seen: HashMap<u32, f32> = HashMap::new();
    for t in tokens {
        let id = term_id(t);
        seen.entry(id).or_insert_with(|| stats.idf(id));
    }
    let mut indices = Vec::with_capacity(seen.len());
    let mut values = Vec::with_capacity(seen.len());
    for (id, w) in seen {
        indices.push(id);
        values.push(w);
    }
    SparseVec { indices, values }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats_for(docs: &[&str]) -> Bm25Stats {
        let mut s = Bm25Stats::default();
        for d in docs {
            s.observe(&tokenize(d));
        }
        s
    }

    fn score(doc: &str, query: &str, stats: &Bm25Stats) -> f32 {
        let d = document_vector(&tokenize(doc), stats);
        let q = query_vector(&tokenize(query), stats);
        let dm: HashMap<u32, f32> = d.indices.iter().copied().zip(d.values).collect();
        q.indices.iter().zip(q.values).map(|(i, w)| dm.get(i).copied().unwrap_or(0.0) * w).sum()
    }

    #[test]
    fn identifiers_survive_tokenization() {
        // The whole reason to add sparse retrieval: dense embeddings lose these.
        let t = tokenize("upgrade to nginx-1.24.0 and set --max-old-space-size");
        assert!(t.contains(&"nginx-1.24.0".to_string()), "{t:?}");
        assert!(t.contains(&"--max-old-space-size".to_string().replace("--", "")) || t.iter().any(|x| x.contains("max-old-space-size")), "{t:?}");
    }

    #[test]
    fn error_strings_tokenize_as_single_terms() {
        let t = tokenize("pod stuck in CrashLoopBackOff state");
        assert!(t.contains(&"crashloopbackoff".to_string()), "{t:?}");
    }

    #[test]
    fn stopwords_are_removed_but_technical_words_are_kept() {
        let t = tokenize("no route to host and the link is up");
        assert!(!t.contains(&"the".to_string()));
        assert!(!t.contains(&"and".to_string()));
        // "no" and "up" carry meaning here and must survive.
        assert!(t.contains(&"no".to_string()), "{t:?}");
        assert!(t.contains(&"up".to_string()), "{t:?}");
    }

    #[test]
    fn a_rare_term_outweighs_a_common_one() {
        let docs = ["nginx error", "nginx timeout", "nginx restart", "nginx crashloopbackoff"];
        let s = stats_for(&docs);
        let common = s.idf(term_id("nginx"));
        let rare = s.idf(term_id("crashloopbackoff"));
        assert!(rare > common, "rare {rare} should outweigh common {common}");
    }

    #[test]
    fn idf_never_goes_negative_for_a_very_common_term() {
        // Without +1 smoothing a term in >50% of documents gets a negative
        // weight and penalises the documents that contain it.
        let docs = ["nginx a", "nginx b", "nginx c", "nginx d", "other e"];
        let s = stats_for(&docs);
        assert!(s.idf(term_id("nginx")) > 0.0, "got {}", s.idf(term_id("nginx")));
    }

    #[test]
    fn a_matching_document_scores_above_a_non_matching_one() {
        let docs = ["nginx 502 bad gateway upstream", "postgres vacuum autovacuum tuning"];
        let s = stats_for(&docs);
        let hit = score(docs[0], "nginx 502 upstream", &s);
        let miss = score(docs[1], "nginx 502 upstream", &s);
        assert!(hit > miss, "hit {hit} vs miss {miss}");
        assert_eq!(miss, 0.0, "no shared terms means no score");
    }

    #[test]
    fn term_frequency_saturates_rather_than_scaling_linearly() {
        // Ten occurrences must not score ten times one; that is the point of k1.
        let s = stats_for(&["nginx error", "other text"]);
        let once = score("nginx", "nginx", &s);
        let many = score("nginx nginx nginx nginx nginx nginx nginx nginx nginx nginx", "nginx", &s);
        assert!(many > once);
        assert!(many < once * 3.0, "saturation failed: {once} -> {many}");
    }

    #[test]
    fn length_normalization_penalises_padding() {
        let s = stats_for(&["nginx timeout", "a b c d e f g h i j k l m n o p"]);
        let short = score("nginx timeout", "nginx", &s);
        let padded = format!("nginx timeout {}", "filler ".repeat(40));
        let long = score(&padded, "nginx", &s);
        assert!(short > long, "short {short} should beat padded {long}");
    }

    #[test]
    fn term_ids_are_stable_across_calls() {
        // Ingest and query must agree without shipping a vocabulary file.
        assert_eq!(term_id("crashloopbackoff"), term_id("crashloopbackoff"));
        assert_ne!(term_id("nginx"), term_id("apache"));
    }

    #[test]
    fn empty_input_produces_an_empty_vector_not_a_panic() {
        let s = Bm25Stats::default();
        assert!(document_vector(&[], &s).is_empty());
        assert!(query_vector(&[], &s).is_empty());
        assert_eq!(s.avg_doc_len(), 1.0, "no division by zero on an empty corpus");
    }

    #[test]
    fn sparse_vectors_pair_indices_with_values() {
        let s = stats_for(&["nginx timeout error"]);
        let v = document_vector(&tokenize("nginx timeout error"), &s);
        assert_eq!(v.indices.len(), v.values.len());
        assert_eq!(v.len(), 3);
        assert!(v.values.iter().all(|x| *x > 0.0));
    }
}
