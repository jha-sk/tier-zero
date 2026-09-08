//! Collection layout and index configuration.
//!
//! The ordering in [`ensure_collection`] is not incidental. Payload indexes are
//! created **before** any point is written, because Qdrant can only build
//! filter-aware HNSW edges for values it knows about at index time. Creating
//! them afterwards leaves a graph with no filter awareness and requires a full
//! rebuild to fix -- and the symptom is not an error, it is quietly degraded
//! recall on every filtered query.

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, FieldType, HnswConfigDiffBuilder,
    KeywordIndexParamsBuilder, OptimizersConfigDiffBuilder, QuantizationType,
    ScalarQuantizationBuilder, SparseVectorParamsBuilder, SparseVectorsConfigBuilder,
    VectorParamsBuilder, VectorsConfigBuilder,
};

/// Name of the dense vector field.
pub const DENSE: &str = "dense";
/// Name of the sparse (BM25-style) vector field.
pub const SPARSE: &str = "sparse";

/// Graph connectivity. 16 is Qdrant's default and a reasonable recall ceiling
/// for a corpus this size; raising it costs build time and memory for recall
/// that int8 rescoring already recovers.
pub const HNSW_M: u64 = 16;
/// Build-time candidate list. Higher is a better graph at higher build cost.
pub const HNSW_EF_CONSTRUCT: u64 = 128;

/// Configuration knobs that get swept against the golden set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexConfig {
    pub dim: u64,
    pub m: u64,
    pub ef_construct: u64,
    /// int8 scalar quantization. Measured quality cost with rescoring enabled
    /// is small; without rescoring it is not.
    pub quantize: bool,
    /// Keep original float vectors resident. Rescoring reads them, so pushing
    /// them to disk turns a RAM-only query into a disk-bound one and the tail
    /// collapses.
    pub originals_in_ram: bool,
    pub segments: u64,
    /// Points a segment must hold before an HNSW graph is built for it.
    ///
    /// Qdrant's default is 20,000, below which it brute-forces instead. That is
    /// the right default for a small collection -- exact search is both faster
    /// and perfectly accurate at that size -- but it is a trap for a benchmark:
    /// a subset index sized near the threshold silently measures exact search
    /// while claiming to measure approximate search, and the recall it reports
    /// is the recall of brute force, not of HNSW.
    pub indexing_threshold: u64,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dim: tz_embed::EMBEDDING_DIM as u64,
            m: HNSW_M,
            ef_construct: HNSW_EF_CONSTRUCT,
            quantize: true,
            originals_in_ram: true,
            segments: 12,
            // Low enough that the benchmark exercises the graph it claims to.
            indexing_threshold: 1_000,
        }
    }
}

/// Payload fields that are indexed, and why each one earns its cost.
///
/// An index is not free: it adds build time and memory, and on a keyword field
/// it changes how the HNSW graph is built. Only fields that are actually
/// filtered on appear here.
pub fn payload_indexes() -> Vec<(&'static str, FieldType)> {
    vec![
        // Hard isolation boundary. Also the highest-value index: marking it as
        // a tenant field lets Qdrant lay a tenant's vectors out contiguously.
        ("tenant_id", FieldType::Keyword),
        // Topic filtering and eval slicing.
        ("tags", FieldType::Keyword),
        // Selective re-indexing when the chunker or encoder changes.
        ("pipeline_version", FieldType::Keyword),
        // Distinguishes prose from code chunks in analysis.
        ("element_type", FieldType::Keyword),
        // Grouping chunks back to their document for deduplicated results.
        ("doc_id", FieldType::Keyword),
    ]
}

/// Create the collection and its payload indexes, in that order.
pub async fn ensure_collection(
    client: &Qdrant,
    name: &str,
    cfg: IndexConfig,
) -> anyhow::Result<bool> {
    if client.collection_exists(name).await? {
        return Ok(false);
    }

    let mut vectors = VectorsConfigBuilder::default();
    vectors.add_named_vector_params(
        DENSE,
        VectorParamsBuilder::new(cfg.dim, Distance::Cosine)
            .hnsw_config(
                HnswConfigDiffBuilder::default()
                    .m(cfg.m)
                    .ef_construct(cfg.ef_construct)
                    .build(),
            )
            .on_disk(!cfg.originals_in_ram)
            .build(),
    );

    let mut sparse = SparseVectorsConfigBuilder::default();
    sparse.add_named_vector_params(SPARSE, SparseVectorParamsBuilder::default().build());

    let mut builder = CreateCollectionBuilder::new(name)
        .vectors_config(vectors)
        .sparse_vectors_config(sparse)
        .optimizers_config(
            OptimizersConfigDiffBuilder::default()
                // One segment per core: a single query fans out across them.
                .default_segment_number(cfg.segments)
                .indexing_threshold(cfg.indexing_threshold)
                .build(),
        )
        // Payload lives off-heap; only the final top-k needs its text.
        .on_disk_payload(true);

    if cfg.quantize {
        builder = builder.quantization_config(
            ScalarQuantizationBuilder::default()
                .r#type(QuantizationType::Int8.into())
                // Clip outliers so a few extreme dimensions do not dominate
                // the quantization range.
                .quantile(0.99)
                // Quantized vectors resident: this is the fast path.
                .always_ram(true)
                .build(),
        );
    }

    client.create_collection(builder).await?;

    // Indexes BEFORE ingest. See the module comment: doing this afterwards
    // silently costs recall on every filtered query.
    for (field, ty) in payload_indexes() {
        let mut req = qdrant_client::qdrant::CreateFieldIndexCollectionBuilder::new(name, field, ty);
        if field == "tenant_id" {
            // Tells Qdrant most queries filter on this, so it can lay out
            // points by tenant and keep a tenant's vectors contiguous.
            req = req.field_index_params(
                KeywordIndexParamsBuilder::default().is_tenant(true).build(),
            );
        }
        client.create_field_index(req).await?;
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_the_embedding_dimension() {
        // A mismatch here is rejected by Qdrant at upsert time, but only after
        // a full embedding pass has already been paid for.
        assert_eq!(IndexConfig::default().dim, tz_embed::EMBEDDING_DIM as u64);
    }

    #[test]
    fn quantization_is_on_and_originals_stay_resident() {
        let c = IndexConfig::default();
        assert!(c.quantize);
        assert!(
            c.originals_in_ram,
            "rescoring reads the originals; on disk they would dominate the tail"
        );
    }

    #[test]
    fn tenant_id_is_indexed_first_of_all() {
        // Ordering is not cosmetic: it is the isolation boundary, and the
        // is_tenant hint changes physical layout.
        let idx = payload_indexes();
        assert_eq!(idx[0].0, "tenant_id");
    }

    #[test]
    fn every_field_that_is_filtered_on_is_indexed() {
        let fields: Vec<&str> = payload_indexes().iter().map(|(f, _)| *f).collect();
        for required in ["tenant_id", "tags", "pipeline_version", "doc_id"] {
            assert!(fields.contains(&required), "{required} is filtered on but not indexed");
        }
    }

    #[test]
    fn the_indexing_threshold_is_below_the_benchmark_corpus_size() {
        // Otherwise the benchmark measures brute-force search while reporting
        // it as approximate search, and the recall figure is meaningless as an
        // ANN result.
        let c = IndexConfig::default();
        assert!(c.indexing_threshold <= 5_000, "got {}", c.indexing_threshold);
    }

    #[test]
    fn segment_count_targets_single_query_latency() {
        // Segments are searched in parallel within one request, so more of them
        // lowers single-query latency at some cost to peak throughput.
        assert!(IndexConfig::default().segments >= 8);
    }
}
