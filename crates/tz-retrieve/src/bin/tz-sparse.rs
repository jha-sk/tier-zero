//! Attaches BM25 sparse vectors to an already-built index.
//!
//! No re-embedding. Chunk ids are deterministic in
//! `(doc_id, chunk_index, content_hash)`, so re-running the same chunking over
//! the same document selection reproduces exactly the ids already in Qdrant,
//! and the sparse vector can be written onto the existing points. Rebuilding
//! from scratch would cost another 38 minutes of CPU embedding for vectors that
//! have not changed.
//!
//! The document selection below must match `tz-index` exactly -- gold first,
//! then distractors in stream order to the same budget -- or the ids will not
//! line up and the updates will silently address nothing.

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{PointVectors, UpdatePointVectorsBuilder, Vector};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use tz_core::PipelineVersion;
use tz_ingest::chunker::{Document, HfCounter, TokenCounter, chunk_document};
use tz_ingest::html::clean_body;
use tz_ingest::posts::PostReader;
use tz_retrieve::bm25::{Bm25Stats, document_vector, tokenize};
use tz_retrieve::schema::SPARSE;

const MAX_ANSWERS: usize = 3;
const UPDATE_BATCH: usize = 256;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "data/serverfault".into());
    let collection = std::env::args().nth(2).unwrap_or_else(|| "tz_10k".into());
    let target_docs: usize =
        std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let golden_path = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "evals/golden/serverfault.json".into());
    let path = format!("{dir}/Posts.xml");

    let client = Qdrant::from_url("http://127.0.0.1:6334").build()?;
    let before = client.collection_info(&collection).await?.result.and_then(|r| r.points_count);
    eprintln!("collection {collection}: {before:?} points");

    let gold: std::collections::HashSet<u64> =
        tz_eval::golden::GoldenSet::load(&golden_path)?
            .pairs
            .iter()
            .map(|p| p.gold_doc_id)
            .collect();

    let counter: Box<dyn TokenCounter> = match HfCounter::find_cached(".") {
        Some(p) => Box::new(HfCounter::from_file(&p)?),
        None => anyhow::bail!("no cached tokenizer"),
    };
    let pv = PipelineVersion {
        parser: "se-xml@1".into(),
        chunker: "struct-512-10@1".into(),
        embedder: "bge-small-en-v1.5@1".into(),
    };

    eprintln!("pass 1: grouping answers");
    let mut answers: HashMap<u64, Vec<(i64, String)>> = HashMap::new();
    for p in PostReader::new(BufReader::with_capacity(1 << 20, File::open(&path)?)) {
        let p = p?;
        if p.is_answer()
            && let Some(parent) = p.parent_id
        {
            let e = answers.entry(parent).or_default();
            e.push((p.score, clean_body(&p.body)));
            if e.len() > MAX_ANSWERS * 4 {
                e.sort_by(|a, b| b.0.cmp(&a.0));
                e.truncate(MAX_ANSWERS);
            }
        }
    }

    eprintln!("pass 2: reproducing the indexed chunk set");
    let t0 = Instant::now();
    let mut docs_seen = 0usize;
    let mut all_chunks: Vec<tz_core::Chunk> = Vec::new();

    for phase in ["gold", "distractors"] {
        let want_gold = phase == "gold";
        let mut batch: Vec<Document> = Vec::with_capacity(512);
        let mut reader = PostReader::new(BufReader::with_capacity(1 << 20, File::open(&path)?));
        let mut done = false;
        while !done {
            batch.clear();
            while batch.len() < 512 {
                match reader.next() {
                    None => {
                        done = true;
                        break;
                    }
                    Some(p) => {
                        let p = p?;
                        if !p.is_question() {
                            continue;
                        }
                        let is_gold = gold.contains(&p.id);
                        if is_gold != want_gold {
                            continue;
                        }
                        if !want_gold && (docs_seen + batch.len()) >= target_docs {
                            done = true;
                            break;
                        }
                        let mut ans = answers.get(&p.id).cloned().unwrap_or_default();
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
                    }
                }
            }
            if batch.is_empty() {
                continue;
            }
            docs_seen += batch.len();
            let cs: Vec<tz_core::Chunk> = batch
                .par_iter()
                .flat_map(|d| chunk_document(d, counter.as_ref(), &pv, "default", true))
                .collect();
            all_chunks.extend(cs);
        }
    }
    eprintln!(
        "  {docs_seen} docs -> {} chunks ({:.1}s)",
        all_chunks.len(),
        t0.elapsed().as_secs_f64()
    );

    // Corpus statistics over exactly the indexed set. Computing IDF over a
    // different document set than the one indexed would weight terms wrongly.
    eprintln!("computing BM25 statistics");
    let tokenized: Vec<Vec<String>> =
        all_chunks.par_iter().map(|c| tokenize(&c.display_text)).collect();
    let mut stats = Bm25Stats::default();
    for t in &tokenized {
        stats.observe(t);
    }
    eprintln!(
        "  {} docs, {} distinct terms, avg len {:.1}",
        stats.doc_count,
        stats.doc_freq.len(),
        stats.avg_doc_len()
    );

    // Persist so the query side scores against the same statistics.
    let stats_path = "data/bm25-stats.json";
    let serial: HashMap<String, u64> =
        stats.doc_freq.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    std::fs::write(
        stats_path,
        serde_json::to_string(&serde_json::json!({
            "doc_count": stats.doc_count,
            "avg_doc_len": stats.avg_doc_len(),
            "doc_freq": serial,
        }))?,
    )?;
    eprintln!("  wrote {stats_path}");

    eprintln!("attaching sparse vectors to existing points");
    let t1 = Instant::now();
    let vectors: Vec<(String, tz_retrieve::bm25::SparseVec)> = all_chunks
        .par_iter()
        .zip(tokenized.par_iter())
        .map(|(c, toks)| (c.id.as_uuid_string(), document_vector(toks, &stats)))
        .collect();

    let mut updated = 0u64;
    let mut empty = 0u64;
    for group in vectors.chunks(UPDATE_BATCH) {
        let pts: Vec<PointVectors> = group
            .iter()
            .filter(|(_, v)| {
                if v.is_empty() {
                    return false;
                }
                true
            })
            .map(|(id, v)| PointVectors {
                id: Some(id.clone().into()),
                vectors: Some(
                    HashMap::from([(
                        SPARSE.to_string(),
                        Vector::new_sparse(v.indices.clone(), v.values.clone()),
                    )])
                    .into(),
                ),
            })
            .collect();
        empty += (group.len() - pts.len()) as u64;
        if pts.is_empty() {
            continue;
        }
        updated += pts.len() as u64;
        client
            .update_vectors(UpdatePointVectorsBuilder::new(&collection, pts).wait(false))
            .await?;
    }

    eprintln!(
        "  updated {updated} points ({empty} skipped as empty) in {:.1}s",
        t1.elapsed().as_secs_f64()
    );
    let after = client.collection_info(&collection).await?.result.and_then(|r| r.points_count);
    eprintln!("collection now {after:?} points (unchanged: vectors added, not points)");
    Ok(())
}
