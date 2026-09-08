//! Evaluation: golden sets, retrieval metrics, and CI gates.

pub mod golden;
pub mod metrics;

pub use golden::{BuildFilters, BuildStats, GoldenPair, GoldenSet};
pub use metrics::{QueryOutcome, RetrievalReport, aggregate, ndcg_at_k_binary, recall_at_k, reciprocal_rank};
