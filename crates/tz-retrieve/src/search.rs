//! Query execution, instrumented per stage.
//!
//! The 25ms budget is spent across four stages and the only way to defend it is
//! to measure each one separately. A single end-to-end number tells you the
//! budget was missed; per-stage histograms tell you which stage to fix.

use crate::bm25::{Bm25Stats, query_vector, tokenize};
use crate::schema::{DENSE, SPARSE};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, Filter, Fusion, PrefetchQueryBuilder, QuantizationSearchParamsBuilder,
    Query as QdrantQuery, QueryPointsBuilder, SearchParamsBuilder,
    with_payload_selector::SelectorOptions,
};
use std::sync::Arc;
use std::sync::Mutex;
use tz_obs::{CacheOutcome, Metrics, RequestRecord};
use std::time::Instant;
use tz_core::{Candidate, ChunkId, Query, Retriever as RetrieverKind};
use tz_embed::{Embedder, LatencyStats};

/// Search-time knobs. These are swept against the golden set; the defaults are
/// a starting point, not a finding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchTuning {
    /// Runtime candidate list. The main recall/latency dial, settable per query
    /// with no reindex.
    pub hnsw_ef: u64,
    /// Fetch this multiple of `top_k` from the quantized index, then rescore
    /// the survivors against full-precision vectors.
    pub oversampling: f64,
    /// Rescore against original vectors. Without it, int8 quality loss is an
    /// order of magnitude worse.
    pub rescore: bool,
    /// Skip segments still being indexed. Bounds latency during ingest at the
    /// cost of transient recall on very fresh data.
    pub indexed_only: bool,
}

impl Default for SearchTuning {
    fn default() -> Self {
        Self { hnsw_ef: 128, oversampling: 4.0, rescore: true, indexed_only: false }
    }
}

/// Per-stage latency, which is the only honest way to report the budget.
#[derive(Debug)]
pub struct SpanStats {
    pub embed: LatencyStats,
    pub search: LatencyStats,
    pub decode: LatencyStats,
    pub total: LatencyStats,
}

impl SpanStats {
    fn new() -> Self {
        Self {
            embed: LatencyStats::new("embed"),
            search: LatencyStats::new("qdrant_search"),
            decode: LatencyStats::new("decode"),
            total: LatencyStats::new("retrieval_total"),
        }
    }

    /// Table for the benchmark report.
    pub fn render(&self) -> String {
        let row = |s: &LatencyStats| {
            format!(
                "{:<16} {:>8} {:>9.2} {:>9.2} {:>9.2} {:>9.2}\n",
                s.name(),
                s.count(),
                s.p50_ms(),
                s.p95_ms(),
                s.p99_ms(),
                s.max_ms()
            )
        };
        let mut out = format!(
            "{:<16} {:>8} {:>9} {:>9} {:>9} {:>9}\n",
            "stage", "n", "p50 ms", "p95 ms", "p99 ms", "max ms"
        );
        out.push_str(&"-".repeat(64));
        out.push('\n');
        out.push_str(&row(&self.embed));
        out.push_str(&row(&self.search));
        out.push_str(&row(&self.decode));
        out.push_str(&row(&self.total));
        out
    }
}

/// Executes queries against a collection.
pub struct Retriever {
    client: Qdrant,
    embedder: Arc<Embedder>,
    collection: String,
    tuning: SearchTuning,
    spans: Mutex<SpanStats>,
    /// Corpus statistics for the sparse retriever. `None` means dense-only.
    bm25: Option<Bm25Stats>,
    /// Optional metric sink. `None` keeps the retriever usable in tests and in
    /// any process with no collector reachable, without a null-object dance at
    /// every call site.
    metrics: Option<Arc<Metrics>>,
}

impl Retriever {
    pub fn new(
        client: Qdrant,
        embedder: Arc<Embedder>,
        collection: impl Into<String>,
        tuning: SearchTuning,
    ) -> Self {
        Self {
            client,
            embedder,
            collection: collection.into(),
            tuning,
            spans: Mutex::new(SpanStats::new()),
            bm25: None,
            metrics: None,
        }
    }

    /// Attach a metric sink so per-request timings reach the dashboard.
    pub fn with_metrics(mut self, m: Arc<Metrics>) -> Self {
        self.metrics = Some(m);
        self
    }

    pub fn tuning(&self) -> SearchTuning {
        self.tuning
    }

    pub fn set_tuning(&mut self, t: SearchTuning) {
        self.tuning = t;
    }

    pub fn reset_stats(&self) {
        *self.spans.lock().expect("span lock") = SpanStats::new();
    }

    pub fn with_stats<R>(&self, f: impl FnOnce(&SpanStats) -> R) -> R {
        f(&self.spans.lock().expect("span lock"))
    }

    /// Dense retrieval.
    ///
    /// The tenant filter is applied server-side from a value the caller supplies
    /// out of band. It is never derived from query text or model output: a
    /// cross-tenant result is a data breach, not a relevance bug.
    #[tracing::instrument(skip_all, fields(tenant = %q.tenant_id, top_k = q.top_k))]
    pub async fn search_dense(&self, q: &Query) -> anyhow::Result<Vec<Candidate>> {
        let t_total = Instant::now();

        let t0 = Instant::now();
        let vector = self.embedder.embed_query(&q.text)?;
        let embed_us = t0.elapsed().as_micros() as u64;

        let mut filter = Filter::must([Condition::matches("tenant_id", q.tenant_id.clone())]);
        if !q.tag_filter.is_empty() {
            // Tag filters land in the range where filterable-HNSW edges may be
            // absent, so their latency and recall are measured separately.
            filter
                .must
                .push(Condition::matches("tags", q.tag_filter.clone()));
        }

        let mut params = SearchParamsBuilder::default()
            .hnsw_ef(self.tuning.hnsw_ef)
            .indexed_only(self.tuning.indexed_only);
        if self.tuning.rescore {
            params = params.quantization(
                QuantizationSearchParamsBuilder::default()
                    .rescore(true)
                    .oversampling(self.tuning.oversampling)
                    .build(),
            );
        }

        let req = QueryPointsBuilder::new(&self.collection)
            .query(vector)
            .using(DENSE)
            .limit(q.top_k as u64)
            .filter(filter)
            .params(params)
            .with_payload(SelectorOptions::Enable(true));

        let t1 = Instant::now();
        let resp = self.client.query(req).await?;
        let search_us = t1.elapsed().as_micros() as u64;

        let t2 = Instant::now();
        let out: Vec<Candidate> = resp
            .result
            .into_iter()
            .map(|p| {
                let get = |k: &str| -> String {
                    p.payload.get(k).and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_default()
                };
                Candidate {
                    chunk_id: ChunkId(
                        p.id
                            .as_ref()
                            .and_then(|i| i.point_id_options.as_ref())
                            .and_then(|o| match o {
                                qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u) => {
                                    u128::from_str_radix(&u.replace('-', ""), 16).ok()
                                }
                                qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => {
                                    Some(*n as u128)
                                }
                            })
                            .unwrap_or(0),
                    ),
                    score: p.score,
                    doc_id: get("doc_id"),
                    display_text: get("display_text"),
                    source_uri: get("source_uri"),
                    section_path: vec![],
                    retrievers: vec![RetrieverKind::Dense],
                }
            })
            .collect();
        let decode_us = t2.elapsed().as_micros() as u64;

        let total_us = t_total.elapsed().as_micros() as u64;
        {
            let mut s = self.spans.lock().expect("span lock");
            s.embed.record(embed_us);
            s.search.record(search_us);
            s.decode.record(decode_us);
            s.total.record(total_us);
        }

        if let Some(m) = &self.metrics {
            // Only stages actually measured here are emitted. Cost is absent
            // because no model was called: an empty cost panel is honest, a
            // synthetic one is not.
            let mut rec = RequestRecord::new(String::new(), q.tenant_id.clone());
            rec.cache_outcome = CacheOutcome::Miss;
            rec.retrieval_candidates = out.len();
            rec.stage("embed", embed_us);
            rec.stage("qdrant_search", search_us);
            rec.stage("decode", decode_us);
            rec.stage("retrieval_total", total_us);
            m.record(&rec);
        }
        Ok(out)
    }

    /// Attach BM25 statistics, enabling hybrid retrieval.
    pub fn with_bm25(mut self, stats: Bm25Stats) -> Self {
        self.bm25 = Some(stats);
        self
    }

    pub fn has_bm25(&self) -> bool {
        self.bm25.is_some()
    }

    /// Sparse-only retrieval. Exists as a diagnostic: a hybrid result that is
    /// worse than dense is uninterpretable without knowing how strong each
    /// constituent is on its own.
    pub async fn search_sparse(&self, q: &Query) -> anyhow::Result<Vec<Candidate>> {
        let Some(stats) = &self.bm25 else {
            anyhow::bail!("no bm25 statistics loaded");
        };
        let t_total = Instant::now();
        let sparse = query_vector(&tokenize(&q.text), stats);
        if sparse.is_empty() {
            return Ok(Vec::new());
        }
        let pairs: Vec<(u32, f32)> =
            sparse.indices.iter().copied().zip(sparse.values.iter().copied()).collect();
        let t1 = Instant::now();
        let resp = self
            .client
            .query(
                QueryPointsBuilder::new(&self.collection)
                    .query(QdrantQuery::new_nearest(pairs.as_slice()))
                    .using(SPARSE)
                    .filter(self.tenant_filter(q))
                    .limit(q.top_k as u64)
                    .with_payload(SelectorOptions::Enable(true)),
            )
            .await?;
        let search_us = t1.elapsed().as_micros() as u64;
        let out = Self::decode(resp, &[RetrieverKind::Sparse]);
        self.record_spans(0, search_us, 0, t_total, q, out.len());
        Ok(out)
    }

    /// Hybrid retrieval: dense and sparse candidates fused server-side.
    ///
    /// Both retrievers run as prefetches inside a single request, so this costs
    /// one round trip rather than two. Fusion is RRF, which combines by *rank*
    /// rather than by score -- necessary because a cosine similarity and a BM25
    /// score share no scale, and normalising between them is a dataset-specific
    /// fudge that has to be retuned whenever the corpus changes.
    ///
    /// Prefetch depth is larger than the final limit on purpose: fusion can
    /// only reorder what each retriever surfaced, so a shallow prefetch caps
    /// the benefit no matter how good the fusion is.
    #[tracing::instrument(skip_all, fields(tenant = %q.tenant_id, top_k = q.top_k))]
    pub async fn search_hybrid(&self, q: &Query) -> anyhow::Result<Vec<Candidate>> {
        let Some(stats) = &self.bm25 else {
            return self.search_dense(q).await;
        };
        let t_total = Instant::now();

        let t0 = Instant::now();
        let dense = self.embedder.embed_query(&q.text)?;
        let embed_us = t0.elapsed().as_micros() as u64;

        let sparse = query_vector(&tokenize(&q.text), stats);
        // No usable lexical signal (all stopwords, or nothing shared with the
        // corpus vocabulary): fall back rather than send a prefetch that can
        // only match nothing and dilute the fusion.
        if sparse.is_empty() {
            return self.search_dense(q).await;
        }

        // The client takes sparse input as (index, weight) pairs.
        let sparse_pairs: Vec<(u32, f32)> =
            sparse.indices.iter().copied().zip(sparse.values.iter().copied()).collect();

        let filter = self.tenant_filter(q);
        let prefetch_limit = (q.top_k as u64).max(50);

        let req = QueryPointsBuilder::new(&self.collection)
            .add_prefetch(
                PrefetchQueryBuilder::default()
                    .query(QdrantQuery::new_nearest(dense))
                    .using(DENSE)
                    .filter(filter.clone())
                    .limit(prefetch_limit)
                    .params(self.search_params())
                    .build(),
            )
            .add_prefetch(
                PrefetchQueryBuilder::default()
                    .query(QdrantQuery::new_nearest(sparse_pairs.as_slice()))
                    .using(SPARSE)
                    .filter(filter)
                    .limit(prefetch_limit)
                    .build(),
            )
            .query(QdrantQuery::new_fusion(Fusion::Rrf))
            .limit(q.top_k as u64)
            .with_payload(SelectorOptions::Enable(true));

        let t1 = Instant::now();
        let resp = self.client.query(req).await?;
        let search_us = t1.elapsed().as_micros() as u64;

        let t2 = Instant::now();
        let out = Self::decode(resp, &[RetrieverKind::Dense, RetrieverKind::Sparse]);
        let decode_us = t2.elapsed().as_micros() as u64;

        self.record_spans(embed_us, search_us, decode_us, t_total, q, out.len());
        Ok(out)
    }

    fn tenant_filter(&self, q: &Query) -> Filter {
        let mut filter = Filter::must([Condition::matches("tenant_id", q.tenant_id.clone())]);
        if !q.tag_filter.is_empty() {
            filter.must.push(Condition::matches("tags", q.tag_filter.clone()));
        }
        filter
    }

    fn search_params(&self) -> qdrant_client::qdrant::SearchParams {
        let mut params = SearchParamsBuilder::default()
            .hnsw_ef(self.tuning.hnsw_ef)
            .indexed_only(self.tuning.indexed_only);
        if self.tuning.rescore {
            params = params.quantization(
                QuantizationSearchParamsBuilder::default()
                    .rescore(true)
                    .oversampling(self.tuning.oversampling)
                    .build(),
            );
        }
        params.build()
    }

    fn record_spans(
        &self,
        embed_us: u64,
        search_us: u64,
        decode_us: u64,
        t_total: Instant,
        q: &Query,
        n: usize,
    ) {
        let total_us = t_total.elapsed().as_micros() as u64;
        {
            let mut s = self.spans.lock().expect("span lock");
            s.embed.record(embed_us);
            s.search.record(search_us);
            s.decode.record(decode_us);
            s.total.record(total_us);
        }
        if let Some(m) = &self.metrics {
            let mut rec = RequestRecord::new(String::new(), q.tenant_id.clone());
            rec.cache_outcome = CacheOutcome::Miss;
            rec.retrieval_candidates = n;
            rec.stage("embed", embed_us);
            rec.stage("qdrant_search", search_us);
            rec.stage("decode", decode_us);
            rec.stage("retrieval_total", total_us);
            m.record(&rec);
        }
    }

    fn decode(
        resp: qdrant_client::qdrant::QueryResponse,
        retrievers: &[RetrieverKind],
    ) -> Vec<Candidate> {
        resp.result
            .into_iter()
            .map(|p| {
                let get = |k: &str| -> String {
                    p.payload
                        .get(k)
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                        .unwrap_or_default()
                };
                Candidate {
                    chunk_id: ChunkId(
                        p.id
                            .as_ref()
                            .and_then(|i| i.point_id_options.as_ref())
                            .and_then(|o| match o {
                                qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u) => {
                                    u128::from_str_radix(&u.replace('-', ""), 16).ok()
                                }
                                qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => {
                                    Some(*n as u128)
                                }
                            })
                            .unwrap_or(0),
                    ),
                    score: p.score,
                    doc_id: get("doc_id"),
                    display_text: get("display_text"),
                    source_uri: get("source_uri"),
                    section_path: vec![],
                    retrievers: retrievers.to_vec(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tuning_rescores_and_oversamples() {
        let t = SearchTuning::default();
        assert!(t.rescore, "int8 without rescoring costs an order of magnitude more quality");
        assert!(t.oversampling >= 2.0);
    }

    #[test]
    fn span_table_reports_every_stage_separately() {
        // A single end-to-end number says the budget was missed; per-stage
        // histograms say which stage to fix.
        let s = SpanStats::new();
        let out = s.render();
        for stage in ["embed", "qdrant_search", "decode", "retrieval_total"] {
            assert!(out.contains(stage), "missing {stage} in:\n{out}");
        }
        assert!(out.contains("p95 ms"));
    }

    #[test]
    fn stats_start_empty_and_reset_cleanly() {
        let mut s = SpanStats::new();
        assert_eq!(s.embed.count(), 0);
        s.embed.record(1000);
        assert_eq!(s.embed.count(), 1);
        s = SpanStats::new();
        assert_eq!(s.embed.count(), 0);
    }
}
