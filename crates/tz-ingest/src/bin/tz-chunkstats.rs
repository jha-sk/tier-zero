//! Chunks the real corpus and reports the resulting distribution.
//!
//! Chunking decisions made against synthetic text are guesses. This runs the
//! configured chunker over every document in the dump and reports what actually
//! comes out: token distribution, how much of the corpus is code, how often the
//! "never split a code block" rule has to yield, and the index size those
//! choices imply.
//!
//! Note on memory: assembling a question with its answers is a join, and a join
//! is the one step that cannot be pure streaming. Answers are grouped in memory
//! and the cost is printed rather than hidden. The scaling path, if the corpus
//! outgrew RAM, is to shard the pass by parent-id range -- N passes over the
//! file for 1/N the memory.

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use tz_ingest::chunker::{ApproxCounter, Document, HfCounter, TokenCounter, chunk_document};
use tz_ingest::html::clean_body;
use tz_ingest::posts::PostReader;
use tz_core::{ElementType, PipelineVersion};

/// Keep only the strongest answers. The long tail of a 40-answer question is
/// mostly noise, and indexing it inflates the corpus without adding recall.
const MAX_ANSWERS: usize = 3;

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| {
                l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok())
            })
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

fn pct(sorted: &[u32], q: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(((sorted.len() - 1) as f64) * q) as usize]
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "data/serverfault/Posts.xml".into());
    let limit: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    // Use the embedding model's own tokenizer where available. A character
    // heuristic disagrees badly with a wordpiece tokenizer on code-dense text,
    // and the disagreement shows up as silently truncated chunks.
    let counter: Box<dyn TokenCounter> = match HfCounter::find_cached(".") {
        Some(p) => {
            eprintln!("tokenizer: {}", p.display());
            Box::new(HfCounter::from_file(&p)?)
        }
        None => {
            eprintln!("tokenizer: none cached, using character approximation");
            Box::new(ApproxCounter)
        }
    };

    eprintln!("pass 1: grouping answers by parent");
    let t0 = Instant::now();
    let mut answers: HashMap<u64, Vec<(i64, String)>> = HashMap::new();
    let mut accepted: HashMap<u64, u64> = HashMap::new();
    for p in PostReader::new(BufReader::with_capacity(1 << 20, File::open(&path)?)) {
        let p = p?;
        if p.is_answer()
            && let Some(parent) = p.parent_id
        {
            let e = answers.entry(parent).or_default();
            e.push((p.score, clean_body(&p.body)));
            // Bound growth: keep only the best few per question.
            if e.len() > MAX_ANSWERS * 4 {
                e.sort_by(|a, b| b.0.cmp(&a.0));
                e.truncate(MAX_ANSWERS);
            }
        } else if p.is_question()
            && let Some(a) = p.accepted_answer_id
        {
            accepted.insert(p.id, a);
        }
    }
    eprintln!("  {} questions have answers  ({:.1}s)", answers.len(), t0.elapsed().as_secs_f64());

    eprintln!("pass 2: assembling and chunking");
    let t1 = Instant::now();
    let pv = PipelineVersion {
        parser: "se-xml@1".into(),
        chunker: "struct-512-10@1".into(),
        embedder: "bge-small-en-v1.5@1".into(),
    };

    let (mut docs, mut chunks) = (0u64, 0u64);
    let (mut code_chunks, split_code_chunks) = (0u64, 0u64);
    let mut tokens: Vec<u32> = Vec::new();
    let mut per_doc: Vec<u32> = Vec::new();
    let mut total_indexed_chars = 0u64;

    // Tokenization dominates this pass and is per-document independent, so
    // documents are assembled serially (the reader is a stream) and chunked in
    // parallel batches. Serial, this took 937s; the batch size is a compromise
    // between scheduling overhead and holding too many documents at once.
    const BATCH: usize = 2048;
    let mut batch: Vec<Document> = Vec::with_capacity(BATCH);

    let drain = |batch: &mut Vec<Document>,
                     docs: &mut u64,
                     chunks: &mut u64,
                     code_chunks: &mut u64,
                     tokens: &mut Vec<u32>,
                     per_doc: &mut Vec<u32>,
                     total_indexed_chars: &mut u64| {
        let results: Vec<Vec<tz_core::Chunk>> = batch
            .par_iter()
            .map(|d| chunk_document(d, counter.as_ref(), &pv, "default", true))
            .collect();
        for cs in results {
            *docs += 1;
            per_doc.push(cs.len() as u32);
            for c in &cs {
                *chunks += 1;
                tokens.push(c.token_count);
                *total_indexed_chars += c.text.len() as u64;
                if c.element_type == ElementType::Code {
                    *code_chunks += 1;
                }
            }
        }
        batch.clear();
    };

    for (i, p) in PostReader::new(BufReader::with_capacity(1 << 20, File::open(&path)?)).enumerate()
    {
        if i >= limit {
            break;
        }
        let p = p?;
        if !p.is_question() {
            continue;
        }
        let mut ans = answers.remove(&p.id).unwrap_or_default();
        // Accepted answer first, then by score: the ordering a reader would use.
        ans.sort_by(|a, b| b.0.cmp(&a.0));
        ans.truncate(MAX_ANSWERS);

        batch.push(Document {
            doc_id: p.id.to_string(),
            title: p.title.unwrap_or_default(),
            tags: p.tags,
            body: clean_body(&p.body),
            answers: ans.into_iter().map(|(_, b)| b).collect(),
            score: p.score,
            effective_date: p.creation_date,
            source_uri: format!("https://serverfault.com/q/{}", p.id),
        });
        if batch.len() >= BATCH {
            drain(&mut batch, &mut docs, &mut chunks, &mut code_chunks,
                  &mut tokens, &mut per_doc, &mut total_indexed_chars);
        }
    }
    if !batch.is_empty() {
        drain(&mut batch, &mut docs, &mut chunks, &mut code_chunks,
              &mut tokens, &mut per_doc, &mut total_indexed_chars);
    }
    let _ = split_code_chunks;

    tokens.sort_unstable();
    per_doc.sort_unstable();
    let secs = t1.elapsed().as_secs_f64();

    println!();
    println!("documents            {docs}");
    println!("chunks               {chunks}");
    println!("chunks/doc           p50={} p95={} max={}", pct(&per_doc, 0.5), pct(&per_doc, 0.95), per_doc.last().copied().unwrap_or(0));
    println!();
    println!("chunk tokens         p50={} p95={} p99={} max={}",
        pct(&tokens, 0.5), pct(&tokens, 0.95), pct(&tokens, 0.99), tokens.last().copied().unwrap_or(0));
    println!("  over 512 budget    {} ({:.3}%)",
        tokens.iter().filter(|t| **t > 512).count(),
        100.0 * tokens.iter().filter(|t| **t > 512).count() as f64 / chunks.max(1) as f64);
    println!("  under 50 tokens    {} ({:.2}%)   <- fragment signature",
        tokens.iter().filter(|t| **t < 50).count(),
        100.0 * tokens.iter().filter(|t| **t < 50).count() as f64 / chunks.max(1) as f64);
    println!();
    println!("code chunks          {code_chunks} ({:.1}%)",
        100.0 * code_chunks as f64 / chunks.max(1) as f64);
    println!("indexed text         {:.2} GB", total_indexed_chars as f64 / 1e9);
    println!();
    println!("elapsed              {secs:.1}s   ({:.0} docs/s, {:.0} chunks/s)",
        docs as f64 / secs, chunks as f64 / secs);
    println!("peak RSS             {:.0} MB   (dominated by the answer join)", peak_rss_mb());
    println!();
    println!("--- implied index size at 384 dims ---");
    let n = chunks as f64;
    println!("fp32 vectors         {:.2} GB", n * 384.0 * 4.0 / 1e9);
    println!("int8 quantized       {:.2} GB", n * 384.0 / 1e9);
    println!("HNSW graph (m=16)    {:.2} GB", n * 32.0 * 4.0 / 1e9);
    println!("int8 + graph         {:.2} GB   <- what must stay resident", (n * 384.0 + n * 128.0) / 1e9);
    Ok(())
}
