//! Builds the vector index from a Stack Exchange dump.
//!
//! Pipeline: stream XML -> assemble question+answers -> chunk -> embed -> upsert.
//!
//! **Corpus scoping.** CPU embedding runs at ~14.5 chunks/s at this corpus's
//! median chunk length, which puts a full 638k-chunk index at 12.3 hours (see
//! `docs/benchmarks/ingest-throughput.txt`). Rather than quote a number from a
//! corpus that was never built, the index is scoped the way IR benchmark
//! subsets are: **every gold document, plus a controlled number of
//! distractors**. Gold documents are mandatory -- recall against an index that
//! does not contain the answer measures nothing -- and the distractor count is
//! the dial that makes retrieval harder.
//!
//! **Gold documents are indexed first, in their own pass.** The obvious
//! implementation -- one pass admitting gold plus distractors until a budget
//! fills -- has two faults that only appear once it is running. Documents
//! arrive in ascending post id, so the distractor budget is consumed entirely
//! by the *oldest* posts before a meaningful number of gold documents is
//! reached; and because gold documents are spread across the whole file, the
//! index does not become measurable until the very last moment of a run that
//! takes tens of minutes. Two passes cost one extra sequential scan of the
//! source (~13s, versus ~45min of embedding) and make the index both
//! deterministic in composition and measurable as soon as pass A completes.

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder, Value};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use std::time::Instant;
use tz_core::PipelineVersion;
use tz_embed::{Embedder, EmbedderConfig, Encoder};
use tz_ingest::chunker::{Document, HfCounter, TokenCounter, chunk_document};
use tz_ingest::html::clean_body;
use tz_ingest::posts::PostReader;
use tz_retrieve::schema::{DENSE, IndexConfig, ensure_collection};

const MAX_ANSWERS: usize = 3;
/// Documents assembled before a chunk+embed round.
const DOC_BATCH: usize = 512;
/// Points per upsert call.
const UPSERT_BATCH: usize = 256;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Which {
    Bge,
    MiniLm,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "data/serverfault".into());
    let collection = std::env::args().nth(2).unwrap_or_else(|| "tierzero".into());
    // Total documents to index, gold documents included.
    let target_docs: usize =
        std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let which = match std::env::args().nth(4).as_deref() {
        Some("minilm") => Which::MiniLm,
        _ => Which::Bge,
    };
    let golden_path = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "evals/golden/serverfault.json".into());
    let path = format!("{dir}/Posts.xml");

    // Gold documents are mandatory members of the index.
    let gold: std::collections::HashSet<u64> = match tz_eval::golden::GoldenSet::load(&golden_path)
    {
        Ok(gs) => gs.pairs.iter().map(|p| p.gold_doc_id).collect(),
        Err(e) => {
            eprintln!("warning: no golden set ({e}); indexing distractors only");
            Default::default()
        }
    };
    eprintln!("gold documents required in index: {}", gold.len());

    let client = Qdrant::from_url("http://127.0.0.1:6334").build()?;
    let cfg = IndexConfig::default();
    let created = ensure_collection(&client, &collection, cfg).await?;
    eprintln!(
        "collection {collection}: {}  (dim={} m={} ef_construct={} quantized={})",
        if created { "created" } else { "already exists" },
        cfg.dim,
        cfg.m,
        cfg.ef_construct,
        cfg.quantize
    );

    let counter: Box<dyn TokenCounter> = match HfCounter::find_cached(".") {
        Some(p) => Box::new(HfCounter::from_file(&p)?),
        None => anyhow::bail!("no cached tokenizer; run the embed tests first to populate it"),
    };
    let encoder = match which {
        Which::Bge => Encoder::BgeSmallEnV15,
        Which::MiniLm => Encoder::AllMiniLmL6V2,
    };
    // Batch 16 measured marginally best at this corpus's chunk length; batch
    // size is within noise here, sequence length is what costs.
    let mut ecfg = EmbedderConfig::for_ingest(encoder);
    ecfg.batch_size = 16;
    let embedder = Arc::new(Embedder::with_config(ecfg)?);
    let pv = PipelineVersion {
        parser: "se-xml@1".into(),
        chunker: "struct-512-10@1".into(),
        embedder: format!("{}@1", encoder.tag()),
    };
    eprintln!("encoder: {}  target docs: {target_docs}", encoder.tag());

    eprintln!("pass 1: grouping answers by parent");
    let t0 = Instant::now();
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
    eprintln!("  {} questions with answers ({:.1}s)", answers.len(), t0.elapsed().as_secs_f64());

    eprintln!("pass 2a: gold documents; pass 2b: distractors");
    let t1 = Instant::now();
    let (mut docs, mut chunks, mut points) = (0u64, 0u64, 0u64);
    let mut gold_indexed = 0u64;
    let (mut t_chunk, mut t_embed, mut t_upsert) = (0f64, 0f64, 0f64);
    let mut batch: Vec<Document> = Vec::with_capacity(DOC_BATCH);

    // Phase A admits only gold documents; phase B fills the remaining budget
    // with distractors. Distractors are taken in stream order (ascending post
    // id, i.e. oldest first), which is a stated sampling bias rather than a
    // random sample.
    for phase in ["gold", "distractors"] {
        let want_gold = phase == "gold";
        eprintln!("  phase: {phase}");
        let mut reader =
            PostReader::new(BufReader::with_capacity(1 << 20, File::open(&path)?)).enumerate();
        let mut done = false;

        while !done {
        batch.clear();
        while batch.len() < DOC_BATCH {
            match reader.next() {
                None => {
                    done = true;
                    break;
                }
                Some((_i, p)) => {
                    let p = p?;
                    if !p.is_question() {
                        continue;
                    }
                    let is_gold = gold.contains(&p.id);
                    if is_gold != want_gold {
                        continue;
                    }
                    if !want_gold && (docs as usize + batch.len()) >= target_docs {
                        done = true;
                        break;
                    }
                    if is_gold {
                        gold_indexed += 1;
                    }
                    let mut ans = answers.remove(&p.id).unwrap_or_default();
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

        let tc = Instant::now();
        let batch_chunks: Vec<tz_core::Chunk> = batch
            .par_iter()
            .flat_map(|d| chunk_document(d, counter.as_ref(), &pv, "default", true))
            .collect();
        t_chunk += tc.elapsed().as_secs_f64();
        docs += batch.len() as u64;
        chunks += batch_chunks.len() as u64;
        if batch_chunks.is_empty() {
            continue;
        }

        let te = Instant::now();
        let texts: Vec<String> = batch_chunks.iter().map(|c| c.text.clone()).collect();
        let vectors = embedder.embed_documents(&texts)?;
        t_embed += te.elapsed().as_secs_f64();

        let tu = Instant::now();
        for (cs, vs) in batch_chunks.chunks(UPSERT_BATCH).zip(vectors.chunks(UPSERT_BATCH)) {
            let pts: Vec<PointStruct> = cs
                .iter()
                .zip(vs)
                .map(|(c, v)| {
                    let payload: HashMap<String, Value> = HashMap::from([
                        ("doc_id".to_string(), Value::from(c.doc_id.clone())),
                        ("display_text".to_string(), Value::from(c.display_text.clone())),
                        ("source_uri".to_string(), Value::from(c.source_uri.clone())),
                        ("tenant_id".to_string(), Value::from(c.tenant_id.clone())),
                        (
                            "element_type".to_string(),
                            Value::from(format!("{:?}", c.element_type)),
                        ),
                        (
                            "pipeline_version".to_string(),
                            Value::from(c.pipeline_version.as_tag()),
                        ),
                        (
                            "tags".to_string(),
                            Value::from(
                                c.tags.iter().cloned().map(Value::from).collect::<Vec<Value>>(),
                            ),
                        ),
                    ]);
                    PointStruct::new(
                        c.id.as_uuid_string(),
                        HashMap::from([(DENSE.to_string(), v.clone())]),
                        payload,
                    )
                })
                .collect();
            points += pts.len() as u64;
            // wait=false: do not block the pipeline on the optimizer.
            client
                .upsert_points(UpsertPointsBuilder::new(&collection, pts).wait(false))
                .await?;
        }
        t_upsert += tu.elapsed().as_secs_f64();

        let el = t1.elapsed().as_secs_f64();
        eprintln!(
            "  {docs} docs ({gold_indexed} gold)  {chunks} chunks  {:.1} chunks/s  [chunk {:.0}s embed {:.0}s upsert {:.0}s]",
            chunks as f64 / el,
            t_chunk,
            t_embed,
            t_upsert
        );
        }
    }

    let el = t1.elapsed().as_secs_f64();
    let info = client.collection_info(&collection).await?;
    println!();
    println!("documents        {docs}");
    println!("  gold           {gold_indexed} of {} required", gold.len());
    println!("chunks           {chunks}");
    println!("points upserted  {points}");
    println!("indexed (server) {:?}", info.result.as_ref().and_then(|r| r.points_count));
    println!();
    println!("elapsed          {el:.1}s   ({:.0} chunks/s)", chunks as f64 / el);
    println!("  chunking       {t_chunk:.1}s");
    println!("  embedding      {t_embed:.1}s   <- dominant stage");
    println!("  upsert         {t_upsert:.1}s");
    println!("peak RSS         {:.0} MB", peak_rss_mb());
    Ok(())
}
