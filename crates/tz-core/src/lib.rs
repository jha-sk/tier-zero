//! Shared types for TierZero.
//!
//! This crate holds the vocabulary the rest of the workspace agrees on: what a
//! chunk is, what a request costs, and what the system promised. It depends on
//! nothing internal, so the budget definitions cannot drift per-crate.

pub mod chunk;
pub mod cost;
pub mod query;
pub mod slo;

pub use chunk::{Chunk, ChunkId, ContentHash, ElementType, PipelineVersion};
pub use cost::{CacheTtl, CallCost, ModelPrice, PriceTable, RequestCost, Usage};
pub use query::{Candidate, Citation, Query, Retriever, RouteTier, rrf_fuse};
pub use slo::{MeasurementContext, SloBudget, SloCheck, SloReport, Transport};
