//! Corpus ingestion: streaming XML, HTML cleanup, document assembly, chunking.

pub mod chunker;
pub mod html;
pub mod links;
pub mod posts;

pub use chunker::{Document, RawChunk, Segment, chunk_document, segment};
pub use html::clean_body;
pub use links::{DuplicateLink, LinkReader};
pub use posts::{Post, PostReader};
