//! Chunks, and the metadata that makes them retrievable, citable, and cheap to
//! re-index.
//!
//! Two hashes, not one. `doc_content_hash` says a document changed;
//! `content_hash` says which chunks changed. Only the second one lets a typo
//! fix on page 4 avoid re-embedding a 300-page document, and that distinction
//! decides what an update costs.

use serde::{Deserialize, Serialize};

/// What kind of content a chunk holds. Drives chunking rules (never split a
/// code block) and eval slicing (a change can be neutral overall and still
/// break every table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementType {
    Prose,
    Code,
    Table,
    ListItem,
    Heading,
    QuestionTitle,
}

/// The parser + chunker + embedder triple.
///
/// Stamped on every chunk so a pipeline change can re-index selectively rather
/// than rebuilding the corpus, and so two chunking strategies can be served
/// side by side during an A/B.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PipelineVersion {
    pub parser: String,
    pub chunker: String,
    pub embedder: String,
}

impl PipelineVersion {
    pub fn as_tag(&self) -> String {
        format!("{}/{}/{}", self.parser, self.chunker, self.embedder)
    }
}

impl std::fmt::Display for PipelineVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_tag())
    }
}

/// Stable 128-bit chunk identity, derived from content rather than assigned.
///
/// Deterministic in (doc_id, chunk_index, content_hash), which makes upserts
/// idempotent: replaying a partially-failed ingest overwrites rather than
/// duplicates. That property is what lets the ingest cursor be committed
/// *after* the writes instead of before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChunkId(pub u128);

impl ChunkId {
    pub fn derive(doc_id: &str, chunk_index: u32, content_hash: &ContentHash) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(doc_id.as_bytes());
        h.update(b"\x00");
        h.update(&chunk_index.to_le_bytes());
        h.update(b"\x00");
        h.update(content_hash.0.as_bytes());
        let bytes = h.finalize();
        let mut b16 = [0u8; 16];
        b16.copy_from_slice(&bytes.as_bytes()[..16]);
        ChunkId(u128::from_le_bytes(b16))
    }

    /// Qdrant point IDs accept a UUID string; this renders the same 128 bits.
    pub fn as_uuid_string(&self) -> String {
        let b = self.0.to_be_bytes();
        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
        )
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// Hash of normalized text. Normalization happens before hashing so that
/// whitespace churn does not present as a content change.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub String);

impl ContentHash {
    pub fn of(text: &str) -> Self {
        ContentHash(blake3::hash(normalize_for_hash(text).as_bytes()).to_hex().to_string())
    }
}

/// Collapse runs of whitespace and trim. Deliberately conservative: it must not
/// change meaning, only representation.
pub fn normalize_for_hash(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_ws = false;
    for c in s.trim().chars() {
        if c.is_whitespace() {
            if !last_ws {
                out.push(' ');
            }
            last_ws = true;
        } else {
            out.push(c);
            last_ws = false;
        }
    }
    out
}

/// A retrievable unit, with everything needed to rank it, filter it, cite it,
/// and decide whether it needs re-embedding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub id: ChunkId,
    pub doc_id: String,
    pub chunk_index: u32,
    pub chunk_count: u32,
    pub content_hash: ContentHash,
    pub doc_content_hash: ContentHash,

    /// Text as embedded, including any prepended context.
    pub text: String,
    /// Text as displayed to a user. Differs from `text` when cheap
    /// contextualization prepends a title, which should be indexed but not
    /// shown back as if the source said it.
    pub display_text: String,

    pub element_type: ElementType,
    /// Heading path, e.g. ["Server Fault", "nginx 502 after upgrade"].
    /// Prepending this to a chunk is a near-zero-cost approximation of
    /// LLM-generated contextual retrieval.
    pub section_path: Vec<String>,
    pub source_uri: String,
    pub tags: Vec<String>,

    /// Hard isolation boundary. Comes from a signed token in the request path,
    /// never from model output.
    pub tenant_id: String,
    /// When the source content was authored, which is not when it was ingested.
    /// Recency ranking must use this one.
    pub effective_date: Option<String>,

    pub pipeline_version: PipelineVersion,
    pub token_count: u32,
}

impl Chunk {
    /// True when this chunk's text differs from `other`, ignoring metadata.
    /// The re-embed decision.
    pub fn content_differs(&self, other: &Chunk) -> bool {
        self.content_hash != other.content_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pv() -> PipelineVersion {
        PipelineVersion {
            parser: "se-xml@1".into(),
            chunker: "struct-512-10@1".into(),
            embedder: "bge-small-en-v1.5-int8@1".into(),
        }
    }

    fn chunk(doc: &str, idx: u32, text: &str) -> Chunk {
        let ch = ContentHash::of(text);
        Chunk {
            id: ChunkId::derive(doc, idx, &ch),
            doc_id: doc.into(),
            chunk_index: idx,
            chunk_count: 1,
            content_hash: ch,
            doc_content_hash: ContentHash::of(text),
            text: text.into(),
            display_text: text.into(),
            element_type: ElementType::Prose,
            section_path: vec!["Server Fault".into()],
            source_uri: format!("https://serverfault.com/q/{doc}"),
            tags: vec!["nginx".into()],
            tenant_id: "default".into(),
            effective_date: None,
            pipeline_version: pv(),
            token_count: 10,
        }
    }

    #[test]
    fn chunk_id_is_deterministic_so_reingest_upserts_rather_than_duplicates() {
        let a = chunk("q123", 0, "hello world");
        let b = chunk("q123", 0, "hello world");
        assert_eq!(a.id, b.id);
    }

    #[test]
    fn chunk_id_varies_with_position_and_content_and_document() {
        let base = chunk("q123", 0, "hello world");
        assert_ne!(base.id, chunk("q123", 1, "hello world").id, "index matters");
        assert_ne!(base.id, chunk("q123", 0, "goodbye world").id, "content matters");
        assert_ne!(base.id, chunk("q999", 0, "hello world").id, "document matters");
    }

    #[test]
    fn whitespace_churn_is_not_a_content_change() {
        // A reformatted source must not trigger re-embedding of the corpus.
        assert_eq!(ContentHash::of("hello   world"), ContentHash::of("hello world"));
        assert_eq!(ContentHash::of("  hello world\n\n"), ContentHash::of("hello world"));
        assert_eq!(ContentHash::of("hello\n\tworld"), ContentHash::of("hello world"));
    }

    #[test]
    fn real_edits_are_content_changes() {
        assert_ne!(ContentHash::of("hello world"), ContentHash::of("hello  w0rld"));
        let a = chunk("q1", 0, "restart nginx");
        let b = chunk("q1", 0, "reload nginx");
        assert!(a.content_differs(&b));
    }

    #[test]
    fn uuid_rendering_is_stable_and_well_formed() {
        let id = ChunkId::derive("q123", 0, &ContentHash::of("x"));
        let u = id.as_uuid_string();
        assert_eq!(u.len(), 36);
        assert_eq!(u.matches('-').count(), 4);
        assert_eq!(u, id.as_uuid_string());
        assert!(u.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
    }

    #[test]
    fn pipeline_version_tags_readably_for_selective_backfill() {
        assert_eq!(pv().as_tag(), "se-xml@1/struct-512-10@1/bge-small-en-v1.5-int8@1");
    }

    #[test]
    fn indexed_text_may_differ_from_displayed_text() {
        // Cheap contextualization prepends the question title for retrieval,
        // but citing it back verbatim would misattribute it to the source.
        let mut c = chunk("q1", 0, "Restart the service.");
        c.text = "nginx 502 after upgrade\n\nRestart the service.".into();
        assert_ne!(c.text, c.display_text);
        assert!(c.text.ends_with(&c.display_text));
    }
}
