//! Reader for `PostLinks.xml`, the source of the golden retrieval set.
//!
//! `LinkTypeId=3` marks a question as a duplicate of another. That is a human
//! judgement, made by a moderator who read both, that these two questions have
//! the same answer. It is exactly a query/relevant-document pair, and getting
//! several thousand of them for free is the reason this corpus was chosen: the
//! eval set is not synthesised by the same class of model being evaluated.

use quick_xml::Reader;
use quick_xml::events::Event;
use std::io::BufRead;

pub const LINK_TYPE_LINKED: u8 = 1;
pub const LINK_TYPE_DUPLICATE: u8 = 3;

/// A moderator-asserted duplicate relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DuplicateLink {
    /// The question that was closed as a duplicate. Used as the *query*.
    pub post_id: u64,
    /// The canonical question it duplicates. Used as the *relevant document*.
    pub related_post_id: u64,
}

pub struct LinkReader<R: BufRead> {
    reader: Reader<R>,
    buf: Vec<u8>,
}

impl<R: BufRead> LinkReader<R> {
    pub fn new(inner: R) -> Self {
        Self { reader: Reader::from_reader(inner), buf: Vec::with_capacity(16 * 1024) }
    }

    /// Collect duplicate links only.
    ///
    /// Self-links are dropped: a question cannot be its own gold answer, and
    /// leaving one in would score as a free hit for any retriever.
    pub fn duplicates(mut self) -> Result<Vec<DuplicateLink>, quick_xml::Error> {
        let mut out = Vec::new();
        loop {
            self.buf.clear();
            match self.reader.read_event_into(&mut self.buf)? {
                Event::Eof => break,
                Event::Empty(e) | Event::Start(e) if e.name().as_ref() == b"row" => {
                    let mut post_id = None;
                    let mut related = None;
                    let mut link_type = 0u8;
                    for attr in e.attributes().flatten() {
                        let Ok(v) = attr.unescape_value() else { continue };
                        match attr.key.as_ref() {
                            b"PostId" => post_id = v.parse().ok(),
                            b"RelatedPostId" => related = v.parse().ok(),
                            b"LinkTypeId" => link_type = v.parse().unwrap_or(0),
                            _ => {}
                        }
                    }
                    if link_type == LINK_TYPE_DUPLICATE
                        && let (Some(p), Some(r)) = (post_id, related)
                        && p != r
                    {
                        out.push(DuplicateLink { post_id: p, related_post_id: r });
                    }
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SAMPLE: &str = r#"<postlinks>
  <row Id="1" PostId="187302" RelatedPostId="15147" LinkTypeId="1" />
  <row Id="2" PostId="487891" RelatedPostId="77541" LinkTypeId="3" />
  <row Id="3" PostId="500" RelatedPostId="500" LinkTypeId="3" />
  <row Id="4" PostId="657952" RelatedPostId="528254" LinkTypeId="3" />
</postlinks>"#;

    fn dups(s: &str) -> Vec<DuplicateLink> {
        LinkReader::new(Cursor::new(s.as_bytes())).duplicates().expect("parses")
    }

    #[test]
    fn only_duplicate_links_are_collected() {
        let d = dups(SAMPLE);
        assert_eq!(d.len(), 2, "LinkTypeId=1 is 'related', not a relevance judgement");
        assert_eq!(d[0], DuplicateLink { post_id: 487891, related_post_id: 77541 });
    }

    #[test]
    fn self_links_are_rejected_as_free_hits() {
        let d = dups(SAMPLE);
        assert!(d.iter().all(|l| l.post_id != l.related_post_id));
    }

    #[test]
    fn empty_input_yields_no_pairs() {
        assert!(dups("<postlinks></postlinks>").is_empty());
    }
}
