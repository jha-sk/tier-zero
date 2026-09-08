//! Streams a Stack Exchange `Posts.xml` and reports corpus statistics.
//!
//! Its real job is to prove the ingest path is streaming: peak RSS is printed
//! alongside the throughput, and it must stay flat as input size grows. A
//! pipeline that quietly buffers a corpus works fine on a sample and dies on
//! the real thing.

use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use tz_ingest::html::{clean_body, has_code};
use tz_ingest::posts::PostReader;

/// Peak resident set size, from the kernel rather than from an allocator hook.
fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "data/serverfault/Posts.xml".to_string()
    });
    let limit: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let bytes = std::fs::metadata(&path)?.len();
    println!("scanning {path} ({:.2} GB)", bytes as f64 / 1e9);

    let f = File::open(&path)?;
    // A generous buffer: this is a sequential scan, and syscall overhead per
    // read is pure waste here.
    let reader = PostReader::new(BufReader::with_capacity(1 << 20, f));

    let t0 = Instant::now();
    let (mut questions, mut answers, mut other) = (0u64, 0u64, 0u64);
    let (mut with_code, mut with_accepted, mut total_chars) = (0u64, 0u64, 0u64);
    let mut tag_hist: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut sample: Option<String> = None;

    for (i, post) in reader.enumerate() {
        if i >= limit {
            break;
        }
        let post = post?;
        match post.post_type_id {
            1 => {
                questions += 1;
                if post.accepted_answer_id.is_some() {
                    with_accepted += 1;
                }
                for t in &post.tags {
                    *tag_hist.entry(t.clone()).or_default() += 1;
                }
            }
            2 => answers += 1,
            _ => other += 1,
        }

        let text = clean_body(&post.body);
        total_chars += text.len() as u64;
        if has_code(&text) {
            with_code += 1;
        }
        if sample.is_none() && post.is_question() && has_code(&text) && text.len() > 400 {
            sample = Some(format!("--- sample q{} ---\n{}", post.id, &text[..400.min(text.len())]));
        }
    }

    let secs = t0.elapsed().as_secs_f64();
    let rows = questions + answers + other;
    println!();
    println!("rows            {rows}");
    println!("  questions     {questions}");
    println!("  answers       {answers}");
    println!("  other         {other}");
    println!("with accepted   {with_accepted}  ({:.1}% of questions)",
        100.0 * with_accepted as f64 / questions.max(1) as f64);
    println!("containing code {with_code}  ({:.1}% of rows)",
        100.0 * with_code as f64 / rows.max(1) as f64);
    println!("cleaned text    {:.2} GB", total_chars as f64 / 1e9);
    println!("distinct tags   {}", tag_hist.len());
    println!();
    println!("elapsed         {secs:.1}s");
    println!("throughput      {:.0} rows/s   {:.1} MB/s",
        rows as f64 / secs, bytes as f64 / 1e6 / secs);
    println!("peak RSS        {:.0} MB   <- must stay flat as input grows", peak_rss_mb());

    let mut top: Vec<_> = tag_hist.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    println!();
    println!("top tags: {}",
        top.iter().take(12).map(|(t, c)| format!("{t}({c})")).collect::<Vec<_>>().join(" "));

    if let Some(s) = sample {
        println!("\n{s}");
    }
    Ok(())
}
