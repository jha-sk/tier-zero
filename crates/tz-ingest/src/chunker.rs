//! Structure-aware chunking.
//!
//! Two decisions here are load-bearing and both are contrary to common practice.
//!
//! **512 tokens with ~10% overlap, not 800/400.** The 800/400 default that ships
//! in most tutorials measures worst on nearly every published metric: the large
//! overlap inflates index size and token cost while actively hurting precision,
//! because the same sentence appears in several neighbouring chunks and crowds
//! the candidate list with near-copies of itself.
//!
//! **Code blocks are atomic.** In an operations corpus the command *is* the
//! answer. A configuration snippet split across a chunk boundary produces two
//! fragments, neither of which answers anything, and the second of which has no
//! indication of what it belongs to. Blocks are packed whole; one that exceeds
//! the budget on its own is split at line boundaries and flagged, never mid-line.

use crate::html::CODE_FENCE;
use tz_core::{ChunkId, ContentHash, ElementType, PipelineVersion};

/// Target chunk size in tokens.
pub const TARGET_TOKENS: usize = 512;
/// Overlap between adjacent prose chunks, in tokens (~10%).
pub const OVERLAP_TOKENS: usize = 51;

/// A span of the cleaned document, tagged by whether it may be split.
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    Prose(String),
    /// Fenced code, including the fences. Atomic where possible.
    Code(String),
}

impl Segment {
    pub fn text(&self) -> &str {
        match self {
            Segment::Prose(s) | Segment::Code(s) => s,
        }
    }
    pub fn is_code(&self) -> bool {
        matches!(self, Segment::Code(_))
    }
}

/// Split cleaned text into alternating prose and code segments.
pub fn segment(text: &str) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_code = false;

    for line in text.lines() {
        if line.trim() == CODE_FENCE {
            // A fence closes the current segment and flips mode.
            if in_code {
                cur.push_str(line);
                cur.push('\n');
                if !cur.trim().is_empty() {
                    out.push(Segment::Code(std::mem::take(&mut cur)));
                }
                in_code = false;
            } else {
                if !cur.trim().is_empty() {
                    out.push(Segment::Prose(std::mem::take(&mut cur)));
                }
                cur.clear();
                cur.push_str(line);
                cur.push('\n');
                in_code = true;
            }
            continue;
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        // An unterminated fence is common in user-authored HTML. Treat what we
        // have as code rather than silently reclassifying it as prose.
        out.push(if in_code { Segment::Code(cur) } else { Segment::Prose(cur) });
    }
    out
}

/// Counts tokens. Abstracted so chunk sizing can use the *embedding model's own*
/// tokenizer rather than a character heuristic: a chunk sized in characters
/// routinely overflows the encoder's window on code-dense text, where tokens per
/// character is far higher than in prose.
pub trait TokenCounter: Send + Sync {
    fn count(&self, text: &str) -> usize;
}

/// Wraps a HuggingFace tokenizer.
pub struct HfCounter(pub tokenizers::Tokenizer);

impl HfCounter {
    pub fn from_file(p: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let t = tokenizers::Tokenizer::from_file(p.as_ref())
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        Ok(Self(t))
    }

    /// Find the tokenizer fastembed cached, so chunk sizing and embedding agree.
    /// Disagreement between them is silent: chunks simply get truncated at
    /// encode time and the tail of each one is never indexed.
    pub fn find_cached(root: impl AsRef<std::path::Path>) -> Option<std::path::PathBuf> {
        fn walk(dir: &std::path::Path, depth: usize, out: &mut Option<std::path::PathBuf>) {
            if depth > 6 || out.is_some() {
                return;
            }
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, depth + 1, out);
                } else if p.file_name().is_some_and(|n| n == "tokenizer.json")
                    && p.to_string_lossy().contains("bge-small")
                {
                    *out = Some(p);
                    return;
                }
            }
        }
        let mut found = None;
        walk(root.as_ref(), 0, &mut found);
        found
    }
}

impl TokenCounter for HfCounter {
    fn count(&self, text: &str) -> usize {
        self.0.encode(text, false).map(|e| e.len()).unwrap_or_else(|_| text.len() / 4)
    }
}

/// Fallback counter for tests and for environments without the model cache.
/// Deliberately conservative: it overestimates rather than under, because an
/// underestimate silently truncates at encode time.
pub struct ApproxCounter;
impl TokenCounter for ApproxCounter {
    fn count(&self, text: &str) -> usize {
        text.len().div_ceil(3)
    }
}

/// A chunk before it has been embedded or given metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct RawChunk {
    pub text: String,
    pub element_type: ElementType,
    pub token_count: usize,
    /// True when an oversized code block had to be split. Tracked so the
    /// "code blocks are never split" claim can be reported with its exceptions
    /// rather than asserted.
    pub split_code: bool,
}

/// Break any segment that exceeds the budget into sub-segments that fit.
///
/// Without this pass the packer can only split *between* segments, so a
/// document with no code fences is one prose segment and comes out as a single
/// oversized chunk. That failure is silent -- the encoder simply truncates and
/// the tail of the document is never indexed.
///
/// Prose is cut at the strongest boundary that works: paragraph, then sentence,
/// then word. Code is handed to the line-boundary splitter.
fn explode(
    segs: &[Segment],
    counter: &dyn TokenCounter,
    target: usize,
) -> Vec<(Segment, bool)> {
    let mut out = Vec::new();
    for seg in segs {
        let n = counter.count(seg.text());
        if n <= target {
            out.push((seg.clone(), false));
            continue;
        }
        match seg {
            Segment::Code(c) => {
                // Flagged so the "code blocks are never split" claim can be
                // reported with its exceptions rather than merely asserted.
                for part in split_code_by_lines(c, counter, target) {
                    out.push((Segment::Code(part), true));
                }
            }
            Segment::Prose(t) => {
                for part in split_prose(t, counter, target) {
                    out.push((Segment::Prose(part), false));
                }
            }
        }
    }
    out
}

/// Greedily pack `units` into pieces under `target`, recursing into a weaker
/// boundary when a single unit is still too large.
fn pack_units(
    units: Vec<String>,
    counter: &dyn TokenCounter,
    target: usize,
    joiner: &str,
    next: Option<&dyn Fn(&str, &dyn TokenCounter, usize) -> Vec<String>>,
) -> Vec<String> {
    // The joiner costs tokens too. Omitting it makes the running sum drift
    // below the true tokenization, one-directionally, so long runs of small
    // units silently overflow the budget.
    let joiner_n = if joiner.is_empty() { 0 } else { counter.count(joiner) };

    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_n = 0usize;
    for u in units {
        let n = counter.count(&u);
        if n > target {
            if !cur.trim().is_empty() {
                out.push(std::mem::take(&mut cur));
                cur_n = 0;
            }
            match next {
                Some(f) => out.extend(f(&u, counter, target)),
                // No weaker boundary left: hard-split rather than emit
                // something the encoder will truncate.
                None => out.extend(hard_split_line(&u, counter, target)),
            }
            continue;
        }
        let cost = if cur.is_empty() { n } else { n + joiner_n };
        if cur_n + cost > target && !cur.trim().is_empty() {
            out.push(std::mem::take(&mut cur));
            cur_n = 0;
        }
        if !cur.is_empty() {
            cur.push_str(joiner);
            cur_n += joiner_n;
        }
        cur.push_str(&u);
        cur_n += n;
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out.retain(|s| !s.trim().is_empty());

    // Verify against the real tokenization rather than trusting the running
    // sum. Tokenizers merge and split across boundaries, so an accounting-only
    // fix is necessary but not sufficient.
    if out.iter().any(|p| counter.count(p) > target) {
        out = out
            .into_iter()
            .flat_map(|p| {
                if counter.count(&p) <= target {
                    vec![p]
                } else {
                    match next {
                        Some(f) => f(&p, counter, target),
                        None => hard_split_line(&p, counter, target),
                    }
                }
            })
            .collect();
    }
    out
}

fn split_by_words(t: &str, counter: &dyn TokenCounter, target: usize) -> Vec<String> {
    // A single "word" can exceed the budget on its own -- a URL, a stack frame,
    // a base64 blob. Expand those first so the packer's terminal branch never
    // has to emit something oversized.
    let words: Vec<String> = t
        .split_whitespace()
        .flat_map(|w| hard_split_line(w, counter, target))
        .collect();
    pack_units(words, counter, target, " ", None)
}

fn split_by_sentences(t: &str, counter: &dyn TokenCounter, target: usize) -> Vec<String> {
    // Cheap sentence split. Deliberately not linguistic: an over-eager split
    // costs a little context, a missed one costs a truncated chunk.
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in t.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') && cur.len() > 1 {
            sentences.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        sentences.push(cur);
    }
    pack_units(sentences, counter, target, "", Some(&split_by_words))
}

/// Split prose at paragraph, then sentence, then word boundaries.
pub fn split_prose(t: &str, counter: &dyn TokenCounter, target: usize) -> Vec<String> {
    let paras: Vec<String> = t.split("\n\n").map(|s| s.to_string()).collect();
    pack_units(paras, counter, target, "\n\n", Some(&split_by_sentences))
}

/// Pack segments into chunks under a token budget.
pub fn chunk_segments(
    segs: &[Segment],
    counter: &dyn TokenCounter,
    target: usize,
    overlap: usize,
) -> Vec<RawChunk> {
    // Segments are sized to leave room for the overlap that will be prepended
    // to the chunk that follows. Sizing them to the full target instead yields
    // chunks of target+overlap, which the encoder then truncates -- costing
    // exactly the overlap the carry was meant to add, and silently.
    let seg_target = target.saturating_sub(overlap).max(target / 2);
    let exploded = explode(segs, counter, seg_target);
    let mut out: Vec<RawChunk> = Vec::new();
    let mut cur = String::new();
    let mut cur_tokens = 0usize;
    let mut cur_has_code = false;
    let mut cur_split_code = false;

    // The running sum used while packing is an approximation: it adds each
    // segment's own token count but not the tokens introduced by the joiners
    // between them, and a tokenizer may merge or split across a boundary. The
    // drift is small but one-directional, so the emitted count is recomputed
    // from the actual text here and the budget is enforced against *that*.
    // Reporting a running sum would mean the histogram says 512 while the
    // encoder sees more and truncates.
    let flush = |cur: &mut String,
                 tokens: &mut usize,
                 has_code: &mut bool,
                 split_code: &mut bool,
                 out: &mut Vec<RawChunk>| {
        let text = cur.trim().to_string();
        cur.clear();
        *tokens = 0;
        let was_code = std::mem::replace(has_code, false);
        let was_split = std::mem::replace(split_code, false);
        if text.is_empty() {
            return;
        }
        let element_type = if was_code { ElementType::Code } else { ElementType::Prose };
        let true_count = counter.count(&text);
        if true_count <= target {
            out.push(RawChunk { text, element_type, token_count: true_count, split_code: was_split });
            return;
        }
        // Over budget after joining. Split it for real rather than emitting
        // something the encoder will silently truncate.
        let pieces = if was_code {
            split_code_by_lines(&text, counter, target)
        } else {
            split_prose(&text, counter, target)
        };
        for piece in pieces {
            let t = piece.trim();
            if t.is_empty() {
                continue;
            }
            out.push(RawChunk {
                text: t.to_string(),
                element_type,
                token_count: counter.count(t),
                split_code: was_split,
            });
        }
    };

    for (seg, was_split) in &exploded {
        let text = seg.text();
        let n = counter.count(text);

        // A split code fragment is emitted on its own so that its provenance
        // stays legible and it is never blended with unrelated prose.
        if *was_split {
            flush(&mut cur, &mut cur_tokens, &mut cur_has_code, &mut cur_split_code, &mut out);
            out.push(RawChunk {
                text: text.trim().to_string(),
                element_type: ElementType::Code,
                token_count: n,
                split_code: true,
            });
            continue;
        }

        if cur_tokens + n > target && !cur.trim().is_empty() {
            let carry = if seg.is_code() || cur_has_code {
                // Never carry overlap across a code boundary: it would
                // duplicate part of a block into a neighbouring chunk.
                String::new()
            } else {
                tail_tokens(&cur, counter, overlap)
            };
            flush(&mut cur, &mut cur_tokens, &mut cur_has_code, &mut cur_split_code, &mut out);
            if !carry.is_empty() {
                cur.push_str(&carry);
                cur.push('\n');
                cur_tokens = counter.count(&cur);
            }
        }

        cur.push_str(text);
        if !cur.ends_with('\n') {
            cur.push('\n');
        }
        cur_tokens += n;
        cur_has_code |= seg.is_code();
    }
    flush(&mut cur, &mut cur_tokens, &mut cur_has_code, &mut cur_split_code, &mut out);
    out
}

/// Trailing ~`n` tokens of `s`, cut at a line boundary where one exists and at
/// a word boundary otherwise.
///
/// The word fallback is not a nicety. A chunk with no line breaks -- a single
/// long paragraph, which is most prose -- has exactly one "line", and a
/// line-only implementation returns the whole thing. The overlap then carries a
/// full chunk forward instead of a tenth of one, and chunk sizes creep past the
/// budget with nothing reporting it.
fn tail_tokens(s: &str, counter: &dyn TokenCounter, n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    let take_tail = |units: Vec<&str>, joiner: &str| -> Option<String> {
        let mut acc: Vec<&str> = Vec::new();
        let mut total = 0usize;
        for u in units.iter().rev() {
            let c = counter.count(u);
            if total + c > n {
                break;
            }
            total += c;
            acc.push(u);
        }
        if acc.is_empty() {
            return None;
        }
        acc.reverse();
        Some(acc.join(joiner))
    };

    take_tail(s.lines().collect(), "\n")
        .or_else(|| take_tail(s.split_whitespace().collect(), " "))
        .unwrap_or_default()
}

/// Hard-split a single line that exceeds the budget on its own.
///
/// Last resort, and it does cut mid-token. Reached only by content that has no
/// internal boundary to respect -- a base64 certificate dump, a minified
/// config, a single enormous log line. Such a line is not answerable prose, and
/// the alternative is worse: an oversized chunk is silently truncated by the
/// encoder, so its tail is never indexed and nothing reports it.
fn hard_split_line(line: &str, counter: &dyn TokenCounter, target: usize) -> Vec<String> {
    if counter.count(line) <= target {
        return vec![line.to_string()];
    }
    // A character budget is only an estimate, because token density is not
    // uniform: a line mixing prose with a base64 blob has an average density
    // that fits neither half. So split, then *verify*, and re-split whatever is
    // still over with a tighter budget. Trusting the first estimate left 1% of
    // the corpus over budget with a worst case of 2.1x.
    fn chop(s: &str, budget_chars: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        for ch in s.chars() {
            cur.push(ch);
            if cur.chars().count() >= budget_chars {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    let estimate_budget = |s: &str, tighten: f64| -> usize {
        let tokens = counter.count(s).max(1);
        let cpt = (s.chars().count() as f64 / tokens as f64).max(1.0);
        (((target as f64) * cpt * tighten) as usize).max(32)
    };

    let mut parts = chop(line, estimate_budget(line, 0.9));
    // Bounded refinement. Each round tightens the budget against the *local*
    // density of the offending piece, so pathological content converges in a
    // few passes rather than looping.
    for _ in 0..6 {
        if parts.iter().all(|p| counter.count(p) <= target) {
            break;
        }
        parts = parts
            .into_iter()
            .flat_map(|p| {
                if counter.count(&p) <= target {
                    vec![p]
                } else {
                    chop(&p, estimate_budget(&p, 0.6))
                }
            })
            .collect();
    }
    parts
}

/// Split an oversized code block at line boundaries, keeping fences on each part.
fn split_code_by_lines(code: &str, counter: &dyn TokenCounter, target: usize) -> Vec<String> {
    let budget_for_lines = target.saturating_sub(8).max(16);
    let raw: Vec<&str> = code.lines().filter(|l| l.trim() != CODE_FENCE).collect();
    // Expand any line that is itself over budget before packing.
    let expanded: Vec<String> = raw
        .into_iter()
        .flat_map(|l| hard_split_line(l, counter, budget_for_lines))
        .collect();
    let body: Vec<&str> = expanded.iter().map(|s| s.as_str()).collect();
    let mut parts = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    let mut tokens = 0usize;
    // Budget for the fences we re-add around each part.
    let budget = target.saturating_sub(8).max(16);
    for line in body {
        let c = counter.count(line);
        if tokens + c > budget && !cur.is_empty() {
            parts.push(format!("{CODE_FENCE}\n{}\n{CODE_FENCE}", cur.join("\n")));
            cur.clear();
            tokens = 0;
        }
        cur.push(line);
        tokens += c;
    }
    if !cur.is_empty() {
        parts.push(format!("{CODE_FENCE}\n{}\n{CODE_FENCE}", cur.join("\n")));
    }
    parts
}

/// A question with its answers, assembled into one retrievable document.
///
/// An answer indexed alone is close to useless: "Restart the service and it
/// will pick up the new certificate" does not say which service or what
/// problem. Binding answers to their question at ingest is cheaper and more
/// reliable than trying to recover the context at query time.
#[derive(Debug, Clone)]
pub struct Document {
    pub doc_id: String,
    pub title: String,
    pub tags: Vec<String>,
    pub body: String,
    /// Answer bodies, accepted first then by descending score.
    pub answers: Vec<String>,
    pub score: i64,
    pub effective_date: Option<String>,
    pub source_uri: String,
}

impl Document {
    /// Full text as chunked: title, question, then answers in rank order.
    pub fn full_text(&self) -> String {
        let mut s = String::with_capacity(self.body.len() + 256);
        s.push_str(&self.title);
        s.push_str("\n\n");
        s.push_str(&self.body);
        for (i, a) in self.answers.iter().enumerate() {
            s.push_str(&format!("\n\nAnswer {}:\n", i + 1));
            s.push_str(a);
        }
        s
    }

    /// Context prefix prepended to every chunk's *indexed* text.
    ///
    /// This is the cheap approximation of LLM-generated contextual retrieval.
    /// It costs one string concatenation per chunk; the LLM version costs about
    /// a dollar per million corpus tokens, roughly a hundred times the entire
    /// embedding bill. Whether the expensive version buys anything over this is
    /// an experiment the golden set can settle.
    pub fn context_prefix(&self) -> String {
        if self.tags.is_empty() {
            format!("{}\n", self.title)
        } else {
            format!("{} [{}]\n", self.title, self.tags.join(", "))
        }
    }
}

/// Chunk a document into indexable units.
pub fn chunk_document(
    doc: &Document,
    counter: &dyn TokenCounter,
    pipeline: &PipelineVersion,
    tenant_id: &str,
    contextualize: bool,
) -> Vec<tz_core::Chunk> {
    let full = doc.full_text();
    let doc_hash = ContentHash::of(&full);
    let segs = segment(&full);
    let raw = chunk_segments(&segs, counter, TARGET_TOKENS, OVERLAP_TOKENS);
    let prefix = doc.context_prefix();
    let total = raw.len() as u32;

    raw.into_iter()
        .enumerate()
        .map(|(i, rc)| {
            // Indexed text carries the context prefix; displayed text does not,
            // so a citation shows what the source actually said.
            let indexed =
                if contextualize { format!("{prefix}{}", rc.text) } else { rc.text.clone() };
            let ch = ContentHash::of(&rc.text);
            tz_core::Chunk {
                id: ChunkId::derive(&doc.doc_id, i as u32, &ch),
                doc_id: doc.doc_id.clone(),
                chunk_index: i as u32,
                chunk_count: total,
                content_hash: ch,
                doc_content_hash: doc_hash.clone(),
                text: indexed,
                display_text: rc.text,
                element_type: rc.element_type,
                section_path: vec![doc.title.clone()],
                source_uri: doc.source_uri.clone(),
                tags: doc.tags.clone(),
                tenant_id: tenant_id.to_string(),
                effective_date: doc.effective_date.clone(),
                pipeline_version: pipeline.clone(),
                token_count: rc.token_count as u32,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c() -> ApproxCounter {
        ApproxCounter
    }

    #[test]
    fn segmentation_separates_prose_from_code() {
        let t = "intro text\n```\nnginx -t\n```\nafter text";
        let s = segment(t);
        assert_eq!(s.len(), 3);
        assert!(!s[0].is_code());
        assert!(s[1].is_code());
        assert!(s[1].text().contains("nginx -t"));
        assert!(!s[2].is_code());
    }

    #[test]
    fn an_unterminated_fence_is_still_treated_as_code() {
        // Common in user-authored HTML; misclassifying it as prose would let
        // the splitter cut through a command.
        let s = segment("text\n```\nrm -rf /var/log/*");
        assert!(s.last().unwrap().is_code());
    }

    #[test]
    fn a_code_block_within_budget_is_never_split() {
        // The headline claim of this module.
        let text = "prose\n```\nline one\nline two\nline three\n```\nmore prose";
        let chunks = chunk_segments(&segment(text), &c(), 512, 51);
        let code_chunks: Vec<_> = chunks.iter().filter(|k| k.text.contains("line one")).collect();
        assert_eq!(code_chunks.len(), 1, "the block appears in exactly one chunk");
        let k = code_chunks[0];
        assert!(k.text.contains("line two") && k.text.contains("line three"));
        assert!(!k.split_code);
    }

    #[test]
    fn an_oversized_code_block_splits_at_line_boundaries_and_is_flagged() {
        let body: String = (0..400).map(|i| format!("configuration_line_number_{i}\n")).collect();
        let text = format!("```\n{body}```");
        let chunks = chunk_segments(&segment(&text), &c(), 512, 51);
        assert!(chunks.len() > 1, "must have split");
        assert!(chunks.iter().all(|k| k.split_code), "splits are flagged as such");
        // No line is cut in half.
        for k in &chunks {
            for line in k.text.lines().filter(|l| l.starts_with("configuration_line")) {
                assert!(
                    line.starts_with("configuration_line_number_"),
                    "line was cut mid-way: {line:?}"
                );
            }
        }
    }

    #[test]
    fn chunks_stay_within_the_token_budget() {
        let prose: String = (0..300).map(|i| format!("sentence number {i} here. ")).collect();
        let chunks = chunk_segments(&segment(&prose), &c(), 512, 51);
        assert!(!chunks.is_empty());
        for k in &chunks {
            assert!(k.token_count <= 512, "chunk of {} tokens", k.token_count);
        }
    }

    #[test]
    fn consecutive_prose_chunks_overlap_by_roughly_the_requested_amount() {
        let prose: String = (0..300).map(|i| format!("line {i}\n")).collect();
        let overlap = 30;
        let chunks = chunk_segments(&segment(&prose), &c(), 200, overlap);
        assert!(chunks.len() > 2);

        // Chunk n+1 must begin with lines that end chunk n, and the shared
        // region must be near the requested size -- not zero (no context
        // carried) and not a large fraction of the chunk (which is the bug
        // that inflated chunk sizes).
        let a: Vec<&str> = chunks[0].text.lines().collect();
        let b: Vec<&str> = chunks[1].text.lines().collect();
        let shared = b
            .iter()
            .take_while(|line| a.contains(line))
            .count();
        assert!(shared > 0, "no overlap carried between prose chunks");
        let shared_text = b[..shared].join("\n");
        let shared_tokens = c().count(&shared_text);
        assert!(
            shared_tokens <= overlap + 8,
            "overlap of {shared_tokens} tokens exceeds the {overlap} requested"
        );
    }

    #[test]
    fn overlap_is_not_carried_across_a_code_boundary() {
        // Duplicating part of a command into a neighbouring chunk would make
        // the same snippet retrievable twice, in two incompatible forms.
        let text = format!(
            "{}\n```\nnginx -t\nsystemctl reload nginx\n```\n{}",
            "prose line\n".repeat(80),
            "more prose\n".repeat(80)
        );
        let chunks = chunk_segments(&segment(&text), &c(), 200, 30);
        let code_idx = chunks.iter().position(|k| k.text.contains("nginx -t")).unwrap();
        let code_chunk = &chunks[code_idx];
        // The command appears in exactly one chunk.
        assert_eq!(
            chunks.iter().filter(|k| k.text.contains("systemctl reload nginx")).count(),
            1
        );
        assert!(code_chunk.text.contains("nginx -t"));
    }

    #[test]
    fn document_assembly_binds_answers_to_their_question() {
        let d = Document {
            doc_id: "q1".into(),
            title: "nginx 502 after upgrade".into(),
            tags: vec!["nginx".into()],
            body: "It broke.".into(),
            answers: vec!["Restart it.".into(), "Check upstream.".into()],
            score: 5,
            effective_date: None,
            source_uri: "https://serverfault.com/q/1".into(),
        };
        let t = d.full_text();
        assert!(t.starts_with("nginx 502 after upgrade"));
        assert!(t.contains("Answer 1:"));
        assert!(t.contains("Restart it."));
        assert!(t.contains("Check upstream."));
    }

    #[test]
    fn the_context_prefix_carries_title_and_tags() {
        let d = Document {
            doc_id: "q1".into(),
            title: "nginx 502".into(),
            tags: vec!["nginx".into(), "proxy".into()],
            body: String::new(),
            answers: vec![],
            score: 0,
            effective_date: None,
            source_uri: String::new(),
        };
        assert_eq!(d.context_prefix(), "nginx 502 [nginx, proxy]\n");
    }

    #[test]
    fn contextualization_changes_indexed_text_but_not_displayed_text() {
        let d = Document {
            doc_id: "q1".into(),
            title: "nginx 502".into(),
            tags: vec!["nginx".into()],
            body: "Restart the service.".into(),
            answers: vec![],
            score: 0,
            effective_date: None,
            source_uri: "u".into(),
        };
        let pv = PipelineVersion {
            parser: "p".into(),
            chunker: "c".into(),
            embedder: "e".into(),
        };
        let with = chunk_document(&d, &c(), &pv, "t", true);
        let without = chunk_document(&d, &c(), &pv, "t", false);
        assert!(with[0].text.starts_with("nginx 502 [nginx]"));
        assert!(!with[0].display_text.starts_with("nginx 502 [nginx]"));
        assert_eq!(with[0].display_text, without[0].display_text);
        // Same underlying content, so the same identity: contextualization is
        // an indexing choice, not a different chunk.
        assert_eq!(with[0].id, without[0].id);
    }

    #[test]
    fn chunk_indices_and_counts_are_consistent() {
        let body: String = (0..500).map(|i| format!("word{i} ")).collect();
        let d = Document {
            doc_id: "q1".into(),
            title: "t".into(),
            tags: vec![],
            body,
            answers: vec![],
            score: 0,
            effective_date: None,
            source_uri: "u".into(),
        };
        let pv = PipelineVersion { parser: "p".into(), chunker: "c".into(), embedder: "e".into() };
        let chunks = chunk_document(&d, &c(), &pv, "t", true);
        assert!(chunks.len() > 1);
        for (i, k) in chunks.iter().enumerate() {
            assert_eq!(k.chunk_index, i as u32);
            assert_eq!(k.chunk_count, chunks.len() as u32);
            assert_eq!(k.doc_content_hash, chunks[0].doc_content_hash);
        }
    }

    #[test]
    fn overlap_never_carries_more_than_it_was_asked_for() {
        // Regression: with no line breaks the line-based tail returned the
        // whole string, so a 51-token overlap silently carried 512 tokens and
        // chunks crept to nearly twice the budget.
        let one_long_line = "word ".repeat(2000);
        let carry = tail_tokens(&one_long_line, &c(), 51);
        let n = c().count(&carry);
        assert!(n <= 51, "carried {n} tokens for a 51-token overlap");
        assert!(!carry.is_empty(), "should still carry something");
    }

    #[test]
    fn a_document_with_no_line_breaks_still_chunks_within_budget() {
        let text = (0..400).map(|i| format!("sentence {i} about nginx. ")).collect::<String>();
        let chunks = chunk_segments(&segment(&text), &c(), 512, 51);
        assert!(chunks.len() > 1);
        for k in &chunks {
            assert!(k.token_count <= 512, "chunk of {} tokens", k.token_count);
        }
    }

    /// The hard invariant: nothing the chunker emits may exceed the encoder
    /// window, because the encoder does not error on an oversized input -- it
    /// truncates, and the tail is never indexed.
    #[test]
    fn no_chunk_ever_exceeds_the_target_budget() {
        let cases: Vec<String> = vec![
            (0..800).map(|i| format!("sentence {i} about nginx timeouts. ")).collect(),
            format!("prose\n```\n{}\n```\nafter", "config_line value\n".repeat(900)),
            format!("```\n{}\n```", "Q".repeat(80_000)),
            (0..300).map(|i| format!("para {i}\n\nbody text here\n\n")).collect(),
            format!("{} {}", "word ".repeat(5000), "Z".repeat(30_000)),
        ];
        for (i, text) in cases.iter().enumerate() {
            for k in chunk_segments(&segment(text), &c(), 512, 51) {
                assert!(
                    k.token_count <= 512,
                    "case {i}: emitted {} tokens, over the 512 encoder window",
                    k.token_count
                );
            }
        }
    }

    #[test]
    fn a_single_unsplittable_line_is_still_brought_under_budget() {
        // Regression from the full-corpus run: 1.12% of chunks exceeded the
        // budget, up to 12,971 tokens, all from code blocks containing one
        // enormous line (base64 dumps, minified configs). The encoder would
        // truncate them silently.
        let blob = "A".repeat(60_000);
        let text = format!("```\n{blob}\n```");
        let chunks = chunk_segments(&segment(&text), &c(), 512, 51);
        assert!(chunks.len() > 1, "must split");
        for k in &chunks {
            assert!(k.token_count <= 512, "chunk of {} tokens still over budget", k.token_count);
        }
    }

    #[test]
    fn an_enormous_single_word_in_prose_is_also_split() {
        let url = format!("see https://example.com/{} for details", "x".repeat(40_000));
        let chunks = chunk_segments(&segment(&url), &c(), 512, 51);
        for k in &chunks {
            assert!(k.token_count <= 512, "chunk of {} tokens", k.token_count);
        }
    }

    #[test]
    fn hard_split_converges_on_mixed_density_content() {
        // The case a single density estimate gets wrong: prose and base64 in
        // one line have an average characters-per-token that fits neither.
        let mixed = format!(
            "{} {} {}",
            "the quick brown fox jumps over the lazy dog ".repeat(200),
            "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5".repeat(400),
            "and then some more ordinary prose follows here ".repeat(200)
        );
        for part in hard_split_line(&mixed, &c(), 512) {
            assert!(
                c().count(&part) <= 512,
                "part of {} tokens survived refinement",
                c().count(&part)
            );
        }
    }

    #[test]
    fn no_content_is_lost_when_a_long_line_is_hard_split() {
        let line = "B".repeat(5_000);
        let parts = hard_split_line(&line, &c(), 512);
        assert!(parts.len() > 1);
        assert_eq!(parts.concat(), line, "hard split must be lossless");
    }

    #[test]
    fn empty_document_produces_no_chunks_rather_than_one_empty_one() {
        assert!(chunk_segments(&segment(""), &c(), 512, 51).is_empty());
        assert!(chunk_segments(&segment("   \n\n  "), &c(), 512, 51).is_empty());
    }
}
