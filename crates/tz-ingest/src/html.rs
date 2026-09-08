//! HTML to text, preserving the structure that retrieval depends on.
//!
//! Stack Exchange bodies are user-authored HTML. Two things must survive the
//! conversion or the corpus becomes unanswerable:
//!
//! * **Code blocks.** Half the value of an ops corpus is in the commands.
//!   Flattening `<pre><code>` into prose destroys both the content and the
//!   chunker's ability to avoid splitting it.
//! * **Tables.** A table rendered as a run of bare cell values is worse than
//!   useless, because it looks like text and answers questions wrongly.

use once_cell::sync::Lazy;
use regex::Regex;

/// Marker wrapping fenced code, used downstream by the chunker to refuse to
/// split inside a block.
pub const CODE_FENCE: &str = "```";

static WS_RUNS: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ \t]{2,}").expect("static regex"));
static BLANK_RUNS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").expect("static regex"));

/// Convert one post body to plain text with fenced code blocks.
///
/// Written as a small hand-rolled walker rather than a generic html2text pass
/// because the block/inline distinction and the code-fence rule are the whole
/// point, and generic converters get both wrong.
pub fn clean_body(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut chars = html.char_indices().peekable();
    // Depth of <pre> nesting: inside a pre, whitespace is significant.
    let mut in_pre = false;
    let mut in_code_inline = false;
    let mut skip_until_close: Option<&'static str> = None;

    while let Some((i, c)) = chars.next() {
        if c != '<' {
            if skip_until_close.is_some() {
                continue;
            }
            out.push(c);
            continue;
        }

        // Read the tag.
        let rest = &html[i..];
        // A '<' only opens a tag when what follows actually looks like one.
        // Ops corpora are full of prose like `if (a < b)`, and treating that
        // as a tag opener silently swallows everything up to the next '>' --
        // which will be a real closing tag several words later.
        if !looks_like_tag(rest) {
            if skip_until_close.is_none() {
                out.push('<');
            }
            continue;
        }
        let Some(end) = rest.find('>') else {
            // Unterminated tag at end of input: treat as a literal.
            if skip_until_close.is_none() {
                out.push('<');
            }
            continue;
        };
        let tag_src = &rest[1..end];
        // Advance the iterator past the tag.
        for _ in 0..end {
            chars.next();
        }

        let closing = tag_src.starts_with('/');
        let name_raw = tag_src.trim_start_matches('/');
        let name: String = name_raw
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();

        if let Some(want) = skip_until_close {
            if closing && name == want {
                skip_until_close = None;
            }
            continue;
        }

        match name.as_str() {
            // Dropped wholesale: their content is not prose.
            "script" | "style" => skip_until_close = Some(Box::leak(name.into_boxed_str())),
            "pre" => {
                if closing {
                    in_pre = false;
                    if !out.ends_with('\n') {
                        out.push('\n');
                    }
                    out.push_str(CODE_FENCE);
                    out.push('\n');
                } else {
                    in_pre = true;
                    if !out.is_empty() && !out.ends_with('\n') {
                        out.push('\n');
                    }
                    out.push_str(CODE_FENCE);
                    out.push('\n');
                }
            }
            // Inline <code> outside a <pre> gets backticks; inside one the
            // fence already covers it and extra ticks would corrupt the block.
            "code" => {
                if !in_pre {
                    in_code_inline = !closing;
                    out.push('`');
                }
            }
            "br" => out.push('\n'),
            "p" | "div" | "ul" | "ol" | "blockquote" | "h1" | "h2" | "h3" | "h4" | "h5"
            | "h6" => {
                if !out.ends_with('\n') {
                    out.push('\n');
                }
                if closing {
                    out.push('\n');
                }
            }
            "li" => {
                if !closing {
                    if !out.ends_with('\n') {
                        out.push('\n');
                    }
                    out.push_str("- ");
                }
            }
            // Table structure is rendered as markdown-ish pipes so that a row
            // stays recognisable as a row after chunking.
            "tr" => {
                if closing {
                    out.push_str(" |\n");
                } else if !out.ends_with('\n') {
                    out.push('\n');
                }
            }
            "td" | "th" => {
                if closing {
                    out.push(' ');
                } else {
                    out.push_str("| ");
                }
            }
            "table"
                if !out.ends_with('\n') => {
                    out.push('\n');
                }
            _ => {}
        }
        let _ = in_code_inline;
    }

    normalize(&out)
}

/// Whether `s`, which starts at a '<', begins a plausible HTML tag.
///
/// Accepts `<div`, `</div`, `<!-- ... `, `<?xml`. Rejects `< b`, `<3`, `< `.
fn looks_like_tag(s: &str) -> bool {
    let mut it = s.chars();
    if it.next() != Some('<') {
        return false;
    }
    match it.next() {
        Some('/') => it.next().is_some_and(|c| c.is_ascii_alphabetic()),
        Some('!') | Some('?') => true,
        Some(c) => c.is_ascii_alphabetic(),
        None => false,
    }
}

/// Collapse whitespace without touching the inside of code fences.
fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_fence = false;
    for line in s.lines() {
        if line.trim_start().starts_with(CODE_FENCE) {
            in_fence = !in_fence;
            out.push_str(CODE_FENCE);
            out.push('\n');
            continue;
        }
        if in_fence {
            // Indentation is meaningful in a shell snippet or a config file.
            out.push_str(line.trim_end());
        } else {
            let collapsed = WS_RUNS.replace_all(line.trim(), " ");
            out.push_str(&collapsed);
        }
        out.push('\n');
    }
    BLANK_RUNS.replace_all(out.trim(), "\n\n").to_string()
}

/// True when the text contains at least one fenced code block.
pub fn has_code(s: &str) -> bool {
    s.contains(CODE_FENCE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paragraphs_become_blank_separated_text() {
        let t = clean_body("<p>First para.</p><p>Second para.</p>");
        assert_eq!(t, "First para.\n\nSecond para.");
    }

    #[test]
    fn inline_code_keeps_backticks() {
        let t = clean_body("<p>Run <code>grep -r</code> now.</p>");
        assert_eq!(t, "Run `grep -r` now.");
    }

    #[test]
    fn pre_blocks_become_fenced_and_keep_indentation() {
        let html = "<p>Config:</p><pre><code>server {\n    listen 80;\n}\n</code></pre>";
        let t = clean_body(html);
        assert!(t.contains("```"), "got: {t}");
        assert!(t.contains("    listen 80;"), "indentation must survive: {t}");
        assert!(has_code(&t));
    }

    #[test]
    fn code_inside_pre_is_not_double_wrapped_in_backticks() {
        // A stray backtick inside a fenced block corrupts the fence.
        let t = clean_body("<pre><code>echo hi</code></pre>");
        assert!(!t.contains('`') || t.matches("```").count() == 2, "got: {t}");
        assert!(!t.contains("`echo"), "no inline ticks inside a fence: {t}");
    }

    #[test]
    fn script_and_style_content_is_dropped_entirely() {
        let t = clean_body("<p>Hi</p><script>alert('x')</script><style>p{color:red}</style>");
        assert_eq!(t, "Hi");
        assert!(!t.contains("alert"));
        assert!(!t.contains("color"));
    }

    #[test]
    fn list_items_become_dashes() {
        let t = clean_body("<ul><li>one</li><li>two</li></ul>");
        assert!(t.contains("- one"), "got: {t}");
        assert!(t.contains("- two"), "got: {t}");
    }

    #[test]
    fn table_cells_keep_row_structure() {
        // A table flattened to bare values answers numeric questions wrongly.
        let t = clean_body("<table><tr><th>Key</th><th>Val</th></tr><tr><td>a</td><td>1</td></tr></table>");
        assert!(t.contains("| Key | Val |"), "got: {t}");
        assert!(t.contains("| a | 1 |"), "got: {t}");
    }

    #[test]
    fn links_keep_their_anchor_text() {
        let t = clean_body(r#"<p>See <a href="http://example.com">the docs</a>.</p>"#);
        assert_eq!(t, "See the docs.");
    }

    #[test]
    fn br_becomes_a_newline() {
        assert_eq!(clean_body("a<br/>b"), "a\nb");
    }

    #[test]
    fn unclosed_angle_bracket_is_treated_as_literal_not_dropped() {
        // Real corpora contain `if (a < b)` written outside a code block.
        let t = clean_body("<p>check a < b here</p>");
        assert!(t.contains('<'), "got: {t}");
    }

    #[test]
    fn whitespace_runs_collapse_outside_fences_only() {
        let t = clean_body("<p>a     b</p><pre><code>x     y</code></pre>");
        assert!(t.contains("a b"), "prose collapses: {t}");
        assert!(t.contains("x     y"), "code does not: {t}");
    }

    #[test]
    fn a_bare_less_than_does_not_swallow_the_rest_of_the_sentence() {
        // The original bug: `find('>')` matched the '>' of the *later* </p>,
        // consuming "< b here</p" as if it were one tag.
        let t = clean_body("<p>check a < b here</p>");
        assert!(t.contains("check a"), "got: {t}");
        assert!(t.contains("b here"), "text after the '<' was dropped: {t}");
    }

    #[test]
    fn comparison_operators_in_prose_survive() {
        let t = clean_body("<p>when x < 5 and y > 3 restart</p>");
        assert!(t.contains("restart"), "got: {t}");
        assert!(t.contains('<'), "got: {t}");
    }

    #[test]
    fn tag_detection_accepts_real_tags_and_rejects_prose() {
        assert!(looks_like_tag("<div>"));
        assert!(looks_like_tag("</div>"));
        assert!(looks_like_tag("<!-- c -->"));
        assert!(!looks_like_tag("< b)"));
        assert!(!looks_like_tag("<3"));
        assert!(!looks_like_tag("< "));
    }

    #[test]
    fn empty_and_plain_inputs_are_handled() {
        assert_eq!(clean_body(""), "");
        assert_eq!(clean_body("just text"), "just text");
    }
}
