//! Builds the golden retrieval set from a Stack Exchange dump.
//!
//! Two streaming passes, so neither the corpus nor the post map is ever fully
//! in memory: pass one reads the duplicate links to learn which post ids
//! matter, pass two streams the corpus and keeps only those.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use tz_eval::golden::{BuildFilters, GoldenSet, build};
use tz_ingest::{LinkReader, PostReader, html::clean_body};

/// Hash of the corpus file, pinned into the golden set.
///
/// Without this, a recall number from last week and one from today are not
/// comparable and nothing in the artefact says so.
fn corpus_hash(path: &str) -> anyhow::Result<String> {
    let mut f = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut h = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().to_hex().to_string())
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "data/serverfault".into());
    let out = std::env::args().nth(2).unwrap_or_else(|| "evals/golden/serverfault.json".into());
    let posts_path = format!("{dir}/Posts.xml");
    let links_path = format!("{dir}/PostLinks.xml");

    eprintln!("pass 1: reading duplicate links from {links_path}");
    let links = LinkReader::new(BufReader::new(File::open(&links_path)?)).duplicates()?;
    eprintln!("  {} duplicate links", links.len());

    let mut needed: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for l in &links {
        needed.insert(l.post_id);
        needed.insert(l.related_post_id);
    }
    eprintln!("  {} distinct posts referenced", needed.len());

    eprintln!("pass 2: streaming {posts_path} for those posts");
    let mut posts: HashMap<u64, (String, String, Vec<String>, i64)> = HashMap::new();
    let reader = PostReader::new(BufReader::with_capacity(1 << 20, File::open(&posts_path)?));
    for p in reader {
        let p = p?;
        // Only questions can be duplicate targets; answers are not retrievable
        // documents in their own right in this design.
        if !p.is_question() || !needed.contains(&p.id) {
            continue;
        }
        posts.insert(
            p.id,
            (p.title.unwrap_or_default(), clean_body(&p.body), p.tags, p.score),
        );
    }
    eprintln!("  resolved {} of {} referenced posts", posts.len(), needed.len());

    eprintln!("hashing corpus for provenance");
    let hash = corpus_hash(&posts_path)?;

    let (pairs, stats) = build(&links, &posts, BuildFilters::default());

    eprintln!("\nbuild stats");
    eprintln!("  duplicate links        {}", stats.duplicate_links);
    eprintln!("  both posts resolved    {}", stats.resolved_both_posts);
    eprintln!("  dropped: missing post  {}", stats.dropped_missing_post);
    eprintln!("  dropped: chained dup   {}", stats.dropped_chained_duplicate);
    eprintln!("  dropped: no title      {}", stats.dropped_no_title);
    eprintln!("  dropped: low score     {}", stats.dropped_low_score);
    eprintln!("  dropped: short query   {}", stats.dropped_short_query);
    eprintln!("  KEPT                   {}", stats.kept);

    let gs = GoldenSet {
        version: "serverfault-dup-v1".into(),
        built_at: "2026-09-07".into(),
        source: "Stack Exchange dump 2024-04-07, PostLinks LinkTypeId=3 (CC BY-SA)".into(),
        corpus_hash: hash,
        pairs,
    };

    if let Some(parent) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(parent)?;
    }
    gs.save(&out)?;

    let (train, test) = gs.split(0.2);
    eprintln!("\nwrote {out}");
    eprintln!("  pairs        {}", gs.len());
    eprintln!("  train/test   {} / {}", train.len(), test.len());
    eprintln!("  corpus hash  {}", &gs.corpus_hash[..16]);

    let mut tags: Vec<_> = gs.tag_distribution().into_iter().collect();
    tags.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!(
        "  top tags     {}",
        tags.iter().take(10).map(|(t, c)| format!("{t}({c})")).collect::<Vec<_>>().join(" ")
    );

    eprintln!("\nfirst 3 pairs:");
    for p in gs.pairs.iter().take(3) {
        eprintln!("  q{} {:?}", p.query_id, p.query_title);
        eprintln!("     -> gold {} {:?} score={}", p.gold_doc_id, p.gold_title, p.gold_score);
    }
    Ok(())
}
