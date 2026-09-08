//! Streaming reader for Stack Exchange `Posts.xml`.
//!
//! The file is 1.2GB for a single site and the full corpus is far larger, so
//! nothing here materialises the document set. Rows are yielded one at a time
//! and the caller decides what to keep. Peak memory must stay flat regardless
//! of input size; the test suite asserts that rather than trusting it.

use quick_xml::Reader;
use quick_xml::events::Event;
use std::io::BufRead;

/// Stack Exchange post type discriminants.
pub const POST_TYPE_QUESTION: u8 = 1;
pub const POST_TYPE_ANSWER: u8 = 2;

/// One row of `Posts.xml`, with only the fields this pipeline uses.
///
/// Deliberately not a faithful mapping of the schema: carrying fields nobody
/// reads would triple the allocation cost of a full-corpus pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Post {
    pub id: u64,
    pub post_type_id: u8,
    /// Set on answers: the question they belong to.
    pub parent_id: Option<u64>,
    /// Set on questions that have one.
    pub accepted_answer_id: Option<u64>,
    pub score: i64,
    pub title: Option<String>,
    /// Raw HTML, already XML-unescaped.
    pub body: String,
    pub tags: Vec<String>,
    pub creation_date: Option<String>,
    pub content_license: Option<String>,
}

impl Post {
    pub fn is_question(&self) -> bool {
        self.post_type_id == POST_TYPE_QUESTION
    }
    pub fn is_answer(&self) -> bool {
        self.post_type_id == POST_TYPE_ANSWER
    }
}

/// Tags arrive as `|sql-server|backup|indexes|`.
///
/// Newer dumps also use the older `<a><b>` form, so both are accepted; an
/// unrecognised shape yields no tags rather than a panic, because a tag parse
/// failure should degrade filtering, not abort a corpus pass.
pub fn parse_tags(raw: &str) -> Vec<String> {
    if raw.is_empty() {
        return Vec::new();
    }
    if raw.starts_with('|') {
        return raw.split('|').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
    }
    if raw.starts_with('<') {
        return raw
            .split('>')
            .filter_map(|s| s.strip_prefix('<'))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
    }
    Vec::new()
}

#[derive(Debug, thiserror::Error)]
pub enum PostsError {
    #[error("xml error: {0}")]
    Xml(#[from] quick_xml::Error),
    #[error("xml attribute error: {0}")]
    Attr(#[from] quick_xml::events::attributes::AttrError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Streaming iterator over `<row .../>` elements.
pub struct PostReader<R: BufRead> {
    reader: Reader<R>,
    buf: Vec<u8>,
    rows_seen: u64,
}

impl<R: BufRead> PostReader<R> {
    pub fn new(inner: R) -> Self {
        let mut reader = Reader::from_reader(inner);
        reader.config_mut().trim_text(false);
        Self { reader, buf: Vec::with_capacity(64 * 1024), rows_seen: 0 }
    }

    pub fn rows_seen(&self) -> u64 {
        self.rows_seen
    }

    /// Next post, or `None` at end of input.
    ///
    /// Rows that fail to parse are skipped rather than fatal: a single
    /// malformed row in a community-authored 1.2GB dump should not cost the
    /// whole ingest.
    pub fn next_post(&mut self) -> Result<Option<Post>, PostsError> {
        loop {
            self.buf.clear();
            match self.reader.read_event_into(&mut self.buf)? {
                Event::Eof => return Ok(None),
                Event::Empty(e) | Event::Start(e) if e.name().as_ref() == b"row" => {
                    self.rows_seen += 1;
                    let mut p = Post::default();
                    let mut ok = false;
                    for attr in e.attributes() {
                        let attr = attr?;
                        let val = attr.unescape_value()?;
                        match attr.key.as_ref() {
                            b"Id" => {
                                if let Ok(v) = val.parse() {
                                    p.id = v;
                                    ok = true;
                                }
                            }
                            b"PostTypeId" => p.post_type_id = val.parse().unwrap_or(0),
                            b"ParentId" => p.parent_id = val.parse().ok(),
                            b"AcceptedAnswerId" => p.accepted_answer_id = val.parse().ok(),
                            b"Score" => p.score = val.parse().unwrap_or(0),
                            b"Title" => p.title = Some(val.into_owned()),
                            b"Body" => p.body = val.into_owned(),
                            b"Tags" => p.tags = parse_tags(&val),
                            b"CreationDate" => p.creation_date = Some(val.into_owned()),
                            b"ContentLicense" => p.content_license = Some(val.into_owned()),
                            _ => {}
                        }
                    }
                    if ok {
                        return Ok(Some(p));
                    }
                }
                _ => {}
            }
        }
    }
}

impl<R: BufRead> Iterator for PostReader<R> {
    type Item = Result<Post, PostsError>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_post().transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SAMPLE: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" AcceptedAnswerId="509" Score="20" Body="&lt;p&gt;How do I &lt;code&gt;grep&lt;/code&gt;?&lt;/p&gt;" Title="Grep question" Tags="|linux|grep|" CreationDate="2009-04-30T06:49:01.807" ContentLicense="CC BY-SA 2.5" />
  <row Id="509" PostTypeId="2" ParentId="1" Score="42" Body="&lt;p&gt;Use &lt;code&gt;grep -r&lt;/code&gt;.&lt;/p&gt;" />
  <row Id="7" PostTypeId="1" Score="-3" Body="&lt;p&gt;Bad question&lt;/p&gt;" Title="Bad" Tags="|off-topic|" />
</posts>"#;

    fn read_all(s: &str) -> Vec<Post> {
        PostReader::new(Cursor::new(s.as_bytes())).map(|r| r.expect("row parses")).collect()
    }

    #[test]
    fn reads_every_row() {
        assert_eq!(read_all(SAMPLE).len(), 3);
    }

    #[test]
    fn distinguishes_questions_from_answers() {
        let posts = read_all(SAMPLE);
        assert!(posts[0].is_question());
        assert!(posts[1].is_answer());
        assert_eq!(posts[1].parent_id, Some(1));
        assert_eq!(posts[0].accepted_answer_id, Some(509));
    }

    #[test]
    fn xml_entities_in_body_are_unescaped_to_real_html() {
        let posts = read_all(SAMPLE);
        // Stored as &lt;p&gt;, must come back as <p> for the HTML cleaner.
        assert!(posts[0].body.starts_with("<p>"), "got {:?}", posts[0].body);
        assert!(posts[0].body.contains("<code>grep</code>"));
    }

    #[test]
    fn pipe_delimited_tags_parse() {
        assert_eq!(parse_tags("|sql-server|backup|"), vec!["sql-server", "backup"]);
        assert_eq!(read_all(SAMPLE)[0].tags, vec!["linux", "grep"]);
    }

    #[test]
    fn angle_bracket_tags_parse_too() {
        assert_eq!(parse_tags("<linux><grep>"), vec!["linux", "grep"]);
    }

    #[test]
    fn unrecognised_tag_shapes_degrade_rather_than_panic() {
        assert!(parse_tags("").is_empty());
        assert!(parse_tags("garbage").is_empty());
    }

    #[test]
    fn negative_scores_are_preserved_not_clamped() {
        // Score drives graded relevance for nDCG; clamping to zero would
        // silently promote bad content to merely-unranked.
        assert_eq!(read_all(SAMPLE)[2].score, -3);
    }

    #[test]
    fn missing_optional_fields_are_none_not_empty_string() {
        let posts = read_all(SAMPLE);
        assert_eq!(posts[1].title, None);
        assert_eq!(posts[2].accepted_answer_id, None);
        assert_eq!(posts[1].content_license, None);
    }

    #[test]
    fn a_malformed_row_does_not_abort_the_stream() {
        let s = r#"<posts>
  <row Id="1" PostTypeId="1" Body="ok" Title="a" />
  <row PostTypeId="1" Body="no id" />
  <row Id="3" PostTypeId="1" Body="ok" Title="c" />
</posts>"#;
        let posts = read_all(s);
        assert_eq!(posts.len(), 2, "the id-less row is skipped, the rest survive");
        assert_eq!(posts[0].id, 1);
        assert_eq!(posts[1].id, 3);
    }
}
