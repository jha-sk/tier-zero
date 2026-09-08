//! In-process query and document embedding.
//!
//! This crate exists because of one measurement: a TLS round trip to a hosted
//! embedding API costs 15-40ms before any compute happens. Against a 25ms
//! retrieval budget that is not a tax, it is a wall. Running the encoder in the
//! same process as the retrieval gateway removes the hop entirely, and it is
//! the single decision the latency SLO rests on.
//!
//! The trade is real and worth stating: we give up the strongest hosted
//! embedding models and accept a 384-dimension open model. Whether that costs
//! retrieval quality is an empirical question answered by the golden set, not
//! an assumption.

use std::sync::Mutex;
use std::time::Instant;

pub mod stats;
pub use stats::LatencyStats;

/// Dimensionality of the embeddings this crate produces.
///
/// 384 is a deliberate choice, not a default. Storage and distance-computation
/// cost scale linearly with it, and evidence across models is that truncating
/// toward 256-512 dimensions retains the large majority of retrieval quality.
/// At 2M chunks the difference between 384 and 1536 dims is roughly 3GB of RAM.
pub const EMBEDDING_DIM: usize = 384;

/// Queries are short. Capping the sequence length keeps the encoder from
/// padding out to its full context and paying for tokens that do not exist.
pub const QUERY_MAX_TOKENS: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("embedding model failed to initialise: {0}")]
    Init(String),
    #[error("embedding failed: {0}")]
    Embed(String),
    #[error("model returned {got} dimensions, expected {expected}")]
    DimensionMismatch { got: usize, expected: usize },
    #[error("empty input: refusing to embed an empty batch")]
    EmptyInput,
}

/// Which encoder to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoder {
    /// bge-small-en-v1.5, 384 dims. The default: small enough for a warm CPU
    /// session inside the latency budget, strong enough to be worth indexing.
    BgeSmallEnV15,
    /// all-MiniLM-L6-v2, 384 dims. Kept as a comparison point for the
    /// embedder-choice row of the benchmark table.
    AllMiniLmL6V2,
    /// bge-small-en-v1.5 with INT8-quantized weights, 384 dims.
    /// Same vector space, cheaper forward pass. Whether the recall cost is
    /// worth the latency is measured, not assumed.
    BgeSmallEnV15Int8,
}

impl Encoder {
    fn to_fastembed(self) -> fastembed::EmbeddingModel {
        match self {
            Encoder::BgeSmallEnV15 => fastembed::EmbeddingModel::BGESmallENV15,
            Encoder::AllMiniLmL6V2 => fastembed::EmbeddingModel::AllMiniLML6V2,
            Encoder::BgeSmallEnV15Int8 => fastembed::EmbeddingModel::BGESmallENV15Q,
        }
    }

    /// Tag recorded in `PipelineVersion.embedder`, so a chunk always says which
    /// encoder produced its vector.
    pub fn tag(self) -> &'static str {
        match self {
            Encoder::BgeSmallEnV15 => "bge-small-en-v1.5",
            Encoder::AllMiniLmL6V2 => "all-minilm-l6-v2",
            Encoder::BgeSmallEnV15Int8 => "bge-small-en-v1.5-int8",
        }
    }

    /// bge models were trained with an instruction prefix on the *query* side
    /// only. Omitting it silently costs retrieval quality; applying it to
    /// documents too is the more common and more damaging mistake.
    pub fn query_prefix(self) -> &'static str {
        match self {
            Encoder::BgeSmallEnV15 => "Represent this sentence for searching relevant passages: ",
            Encoder::AllMiniLmL6V2 => "",
            Encoder::BgeSmallEnV15Int8 => {
                "Represent this sentence for searching relevant passages: "
            }
        }
    }
}

/// How to build the ONNX session.
///
/// These knobs exist because the defaults are wrong for this workload. ONNX
/// Runtime sizes its thread pool for throughput on large batches; a single
/// 15-token query is the opposite shape, and the cost of fanning out and
/// rejoining threads can exceed the arithmetic being parallelised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedderConfig {
    pub encoder: Encoder,
    /// ONNX Runtime intra-op threads. `None` uses the runtime default, which
    /// on a 12-core box means a 12-way fan-out for a tiny matmul.
    pub intra_threads: Option<usize>,
    /// Truncation limit. Queries are short; documents are chunk-sized.
    pub max_length: usize,
    /// Sequences per forward pass.
    ///
    /// This is a **memory** knob before it is a throughput knob. Transformer
    /// attention is O(batch x heads x seq^2), so batch size and sequence length
    /// interact quadratically: at batch 256 and 512 tokens the attention
    /// tensors alone run to gigabytes per layer. Handing the encoder an
    /// unbounded batch drove resident memory to 15.4 GB during ingest before
    /// this was capped.
    pub batch_size: usize,
}

impl EmbedderConfig {
    /// Configuration for the retrieval hot path.
    ///
    /// `intra_threads: None` (the ONNX Runtime default) is measured, not
    /// assumed. The intuition that a ten-token forward pass is too small to
    /// parallelise is wrong on this hardware by a factor of two: pinning to one
    /// thread costs 19.6ms p95 against 8.7ms for the default fan-out. See
    /// `examples/encoder_sweep.rs` and docs/adr/0002. Re-run the sweep when the
    /// hardware changes; do not port these values by intuition.
    pub fn for_queries(encoder: Encoder) -> Self {
        Self { encoder, intra_threads: None, max_length: QUERY_MAX_TOKENS, batch_size: 1 }
    }

    /// Configuration for bulk ingest, where batches are large and throughput
    /// matters more than any single call.
    pub fn for_ingest(encoder: Encoder) -> Self {
        Self { encoder, intra_threads: None, max_length: 512, batch_size: 32 }
    }
}

/// A loaded, warm encoder.
///
/// Construction is expensive (model load, ONNX session build) and happens once
/// at startup. Everything on the request path reuses it. A cold session inside
/// a request would blow the budget on its own.
pub struct Embedder {
    model: Mutex<fastembed::TextEmbedding>,
    encoder: Encoder,
    config: EmbedderConfig,
    query_stats: Mutex<LatencyStats>,
}

impl Embedder {
    /// Load an encoder and warm it.
    ///
    /// The warm-up embed is not optional. The first inference through an ONNX
    /// session pays for lazy allocation and kernel selection, and measuring it
    /// as if it were steady state is how cold-start numbers leak into a p95.
    pub fn new(encoder: Encoder) -> Result<Self, EmbedError> {
        Self::with_config(EmbedderConfig::for_queries(encoder))
    }

    /// Load an encoder with explicit session configuration.
    pub fn with_config(cfg: EmbedderConfig) -> Result<Self, EmbedError> {
        let mut opts = fastembed::InitOptions::new(cfg.encoder.to_fastembed())
            .with_show_download_progress(false)
            .with_max_length(cfg.max_length);
        if let Some(n) = cfg.intra_threads {
            opts = opts.with_intra_threads(n);
        }
        let model =
            fastembed::TextEmbedding::try_new(opts).map_err(|e| EmbedError::Init(e.to_string()))?;

        let this = Self {
            model: Mutex::new(model),
            encoder: cfg.encoder,
            config: cfg,
            query_stats: Mutex::new(LatencyStats::new("embed_query")),
        };

        // Warm the session, then discard the timing.
        let v = this.embed_query_inner("warmup")?;
        if v.len() != EMBEDDING_DIM {
            return Err(EmbedError::DimensionMismatch { got: v.len(), expected: EMBEDDING_DIM });
        }
        this.query_stats.lock().expect("warmup stats lock").reset();
        Ok(this)
    }

    pub fn encoder(&self) -> Encoder {
        self.encoder
    }

    pub fn config(&self) -> EmbedderConfig {
        self.config
    }

    fn embed_query_inner(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let prefixed = format!("{}{}", self.encoder.query_prefix(), text);
        let mut m = self.model.lock().expect("embedder lock");
        let out = m
            .embed(vec![prefixed], None)
            .map_err(|e| EmbedError::Embed(e.to_string()))?;
        out.into_iter().next().ok_or(EmbedError::EmptyInput)
    }

    /// Embed one query. This is the hot path; it runs inside the 25ms budget.
    #[tracing::instrument(skip_all, fields(chars = text.len()))]
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if text.trim().is_empty() {
            return Err(EmbedError::EmptyInput);
        }
        let t0 = Instant::now();
        let v = self.embed_query_inner(text)?;
        let micros = t0.elapsed().as_micros() as u64;
        self.query_stats.lock().expect("stats lock").record(micros);
        if v.len() != EMBEDDING_DIM {
            return Err(EmbedError::DimensionMismatch { got: v.len(), expected: EMBEDDING_DIM });
        }
        Ok(v)
    }

    /// Embed documents in bulk, for ingest.
    ///
    /// Inputs are bucketed by length before batching. Transformer cost is set by
    /// the longest sequence in a batch, so mixing a 20-token chunk with a
    /// 500-token one makes the short one cost as much as the long one. Bucketing
    /// is the difference between using the CPU and heating it.
    #[tracing::instrument(skip_all, fields(n = texts.len()))]
    pub fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Err(EmbedError::EmptyInput);
        }

        // Sort indices by length, embed in that order, then restore input order.
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| texts[i].len());
        let sorted: Vec<String> = order.iter().map(|&i| texts[i].clone()).collect();

        // Sub-batch explicitly rather than handing the encoder everything.
        // Length bucketing above means each sub-batch is internally uniform, so
        // padding waste stays low *and* peak memory is bounded by the longest
        // sequence in the current sub-batch rather than in the whole call.
        let bs = self.config.batch_size.max(1);
        let mut m = self.model.lock().expect("embedder lock");
        let mut embedded: Vec<Vec<f32>> = Vec::with_capacity(sorted.len());
        for group in sorted.chunks(bs) {
            let part = m
                .embed(group, Some(bs))
                .map_err(|e| EmbedError::Embed(e.to_string()))?;
            embedded.extend(part);
        }
        drop(m);

        if let Some(v) = embedded.first()
            && v.len() != EMBEDDING_DIM
        {
            return Err(EmbedError::DimensionMismatch { got: v.len(), expected: EMBEDDING_DIM });
        }

        let mut out: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        for (slot, vec) in order.into_iter().zip(embedded) {
            out[slot] = vec;
        }
        Ok(out)
    }

    /// Snapshot of query-embedding latency. Feeds the retrieval budget report.
    pub fn query_latency(&self) -> LatencyStats {
        self.query_stats.lock().expect("stats lock").clone()
    }
}

/// Cosine similarity. Inputs are expected to be normalized already, but this
/// does not assume it, because a silently unnormalized vector produces
/// plausible-looking wrong scores rather than an error.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let v = vec![0.1, 0.2, 0.3, 0.4];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_opposed_vectors_is_negative_one() {
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_with_a_zero_vector_is_zero_not_nan() {
        // A NaN here would propagate into ranking and sort unpredictably.
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert!(!cosine(&[0.0, 0.0], &[0.0, 0.0]).is_nan());
    }

    #[test]
    fn bge_applies_a_query_prefix_and_minilm_does_not() {
        assert!(Encoder::BgeSmallEnV15.query_prefix().starts_with("Represent this sentence"));
        assert_eq!(Encoder::AllMiniLmL6V2.query_prefix(), "");
    }

    #[test]
    fn ingest_batches_are_bounded_because_attention_is_quadratic_in_sequence() {
        // Regression: an unbounded batch drove resident memory to 15.4 GB.
        let c = EmbedderConfig::for_ingest(Encoder::BgeSmallEnV15);
        assert!(c.batch_size >= 1 && c.batch_size <= 64, "got {}", c.batch_size);
        assert_eq!(EmbedderConfig::for_queries(Encoder::BgeSmallEnV15).batch_size, 1);
    }

    #[test]
    fn encoder_tags_are_stable_for_pipeline_versioning() {
        assert_eq!(Encoder::BgeSmallEnV15.tag(), "bge-small-en-v1.5");
        assert_eq!(Encoder::AllMiniLmL6V2.tag(), "all-minilm-l6-v2");
    }
}
