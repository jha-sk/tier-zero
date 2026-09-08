//! Exact and semantic response caching.

pub mod guard;
pub mod store;

pub use guard::{GuardReason, semantic_lookup_allowed};
pub use store::{CacheKey, CacheStore, Lookup};
