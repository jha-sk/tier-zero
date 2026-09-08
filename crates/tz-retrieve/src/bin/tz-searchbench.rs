//! Measures the retrieval SLO against the golden set.
//!
//! Reports latency **and** recall together, always. A latency number without
//! its recall is not a result: any retriever can be made arbitrarily fast by
//! returning less, and any recall can be bought with an unbounded ef. The pair
//! is the claim.
//!
//! Every run prints the conditions it was taken under -- corpus size, arrival
//! pattern, warm/cold, transport -- because a warm, unloaded, single-threaded
//! number quoted as a production figure is the most common dishonest benchmark
//! in this space.

use qdrant_client::Qdrant;
use std::sync::Arc;
use std::time::Instant;
use tz_core::Query;
use tz_embed::{Embedder, EmbedderConfig, Encoder};
use tz_eval::golden::GoldenSet;
use tz_eval::metrics::{QueryOutcome, aggregate};
use tz_retrieve::search::{Retriever, SearchTuning};

/// Map chunks back to distinct source documents, preserving rank order.
///
/// Recall is measured at **document** level, because that is what the golden
/// set names and because five chunks of the same document filling the top five
/// is one result, not five.
///
/// This creates a trap worth naming: retrieving 100 *chunks* does not yield 100
/// *documents*. At this corpus's measured 5 chunks per document at p95, a
/// top-100 chunk fetch collapses to far fewer documents, so a metric labelled
/// "recall@100" computed that way is silently measuring a much shallower cutoff
/// than it claims. The fetch is therefore over-provisioned (see
/// `CHUNK_FETCH_MULTIPLE`) and the document list truncated to the intended k.
fn doc_ids_in_order(cands: &[tz_core::Candidate]) -> Vec<u64> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for c in cands {
        if let Ok(id) = c.doc_id.parse::<u64>()
            && seen.insert(id)
        {
            out.push(id);
        }
    }
    out
}

/// Chunks fetched per document slot wanted. Sized from the measured chunks-per
/// document distribution (p50 1, p95 5) so that a top-100 *document* cutoff is
/// actually reachable.
const CHUNK_FETCH_MULTIPLE: usize = 5;
/// Deepest document cutoff reported.
const MAX_DOC_K: usize = 100;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let collection = std::env::args().nth(1).unwrap_or_else(|| "tierzero".into());
    let golden_path =
        std::env::args().nth(2).unwrap_or_else(|| "evals/golden/serverfault.json".into());
    let n_queries: usize =
        std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(500);

    let client = Qdrant::from_url("http://127.0.0.1:6334").build()?;
    let info = client.collection_info(&collection).await?;
    let points = info.result.as_ref().and_then(|r| r.points_count).unwrap_or(0);
    let indexed = info.result.as_ref().and_then(|r| r.indexed_vectors_count).unwrap_or(0);

    let gs = GoldenSet::load(&golden_path)?;
    // Measure on the held-out split so tuning cannot leak into the number.
    let (_train, test) = gs.split(0.2);
    let queries: Vec<_> = test.into_iter().take(n_queries).collect();

    let embedder = Arc::new(Embedder::with_config(EmbedderConfig::for_queries(
        Encoder::BgeSmallEnV15,
    ))?);

    // Export to the collector when one is reachable. A benchmark that produces
    // numbers but leaves the dashboard empty has only proved half the pipeline.
    let otlp = std::env::var("TZ_OTLP_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:4317".into());
    let meter_provider = tz_obs::init_meter_provider(&otlp, "tz-searchbench").ok();
    let metrics = meter_provider
        .as_ref()
        .map(|_| Arc::new(tz_obs::Metrics::new(&opentelemetry::global::meter("tz-searchbench"))));
    if metrics.is_some() {
        println!("exporting metrics to {otlp}");
    }

    println!("collection       {collection}");
    println!("points           {points}  (indexed {indexed})");
    println!("golden set       {} pairs, corpus {}", gs.len(), &gs.corpus_hash[..12]);
    println!("measuring on     {} held-out queries", queries.len());
    println!("transport        gRPC, loopback, co-located");
    println!();

    // Dense vs hybrid on identical queries against an identical index. The
    // only variable is the retriever, which is what makes the delta a result
    // rather than an anecdote.
    let bm25 = tz_retrieve::bm25::Bm25Stats::load("data/bm25-stats.json").ok();
    if let Some(b) = &bm25 {
        println!("BM25 statistics loaded: {} docs, {} terms\n", b.doc_count, b.doc_freq.len());

        println!(
            "{:<10} {:>9} {:>9} {:>10} {:>10} {:>10} {:>9}",
            "retriever", "p50 ms", "p95 ms", "recall@10", "recall@50", "recall@100", "mrr"
        );
        println!("{}", "-".repeat(72));

        // Two query shapes. Titles are what a user types into a search box;
        // bodies carry the error strings, commands and identifiers that lexical
        // matching is supposed to be good at. If BM25's weakness is a property
        // of the query shape rather than of the retriever, it shows up here.
        for (shape, use_body) in [("title", false), ("body", true)] {
        println!("  -- query shape: {shape} --");
        for mode in ["dense", "sparse", "hybrid"] {
            let mut r = Retriever::new(
                Qdrant::from_url("http://127.0.0.1:6334").build()?,
                embedder.clone(),
                &collection,
                SearchTuning::default(),
            );
            if mode != "dense" {
                r = r.with_bm25(b.clone());
            }
            let text_of = |g: &tz_eval::golden::GoldenPair| -> String {
                if use_body && !g.query_body.trim().is_empty() {
                    // Title plus the opening of the body: what a support ticket
                    // actually looks like.
                    format!("{} {}", g.query_title, g.query_body)
                } else {
                    g.query_title.clone()
                }
            };
            for q in queries.iter().take(20) {
                let mut qq = Query::new(text_of(q), "default");
                qq.top_k = MAX_DOC_K * CHUNK_FETCH_MULTIPLE;
                let _ = match mode {
                    "hybrid" => r.search_hybrid(&qq).await,
                    "sparse" => r.search_sparse(&qq).await,
                    _ => r.search_dense(&qq).await,
                };
            }
            r.reset_stats();

            let mut outcomes = Vec::with_capacity(queries.len());
            for gp in &queries {
                let mut q = Query::new(text_of(gp), "default");
                q.top_k = MAX_DOC_K * CHUNK_FETCH_MULTIPLE;
                let cands = match mode {
                    "hybrid" => r.search_hybrid(&q).await?,
                    "sparse" => r.search_sparse(&q).await?,
                    _ => r.search_dense(&q).await?,
                };
                let mut docs = doc_ids_in_order(&cands);
                docs.truncate(MAX_DOC_K);
                outcomes.push(QueryOutcome {
                    query_id: gp.query_id,
                    retrieved: docs,
                    relevant: vec![gp.gold_doc_id],
                });
            }
            let rep = aggregate(&outcomes);
            let (p50, p95) = r.with_stats(|s| (s.total.p50_ms(), s.total.p95_ms()));
            println!(
                "{mode:<10} {p50:>9.2} {p95:>9.2} {:>10.3} {:>10.3} {:>10.3} {:>9.3}",
                rep.recall_at_10, rep.recall_at_50, rep.recall_at_100, rep.mrr
            );
        }
        }
        println!();
    }

    // Sweep the main recall/latency dial. It is settable per query with no
    // reindex, which is what makes it the right knob to expose.
    let efs = [32u64, 64, 128, 256];
    println!(
        "{:>7} {:>9} {:>9} {:>9} {:>10} {:>10} {:>10} {:>9}",
        "hnsw_ef", "p50 ms", "p95 ms", "p99 ms", "recall@10", "recall@50", "recall@100", "mrr"
    );
    println!("{}", "-".repeat(84));

    for ef in efs {
        let mut r = Retriever::new(
            Qdrant::from_url("http://127.0.0.1:6334").build()?,
            embedder.clone(),
            &collection,
            SearchTuning { hnsw_ef: ef, ..Default::default() },
        );
        if let Some(m) = &metrics {
            r = r.with_metrics(m.clone());
        }

        // Warm the connection and page cache; a cold first query is reported
        // separately, never folded into a steady-state percentile.
        for q in queries.iter().take(20) {
            let mut qq = Query::new(q.query_title.clone(), "default");
            qq.top_k = MAX_DOC_K * CHUNK_FETCH_MULTIPLE;
            let _ = r.search_dense(&qq).await;
        }
        r.reset_stats();

        let mut outcomes = Vec::with_capacity(queries.len());
        for gp in &queries {
            let mut q = Query::new(gp.query_title.clone(), "default");
            q.top_k = MAX_DOC_K * CHUNK_FETCH_MULTIPLE;
            let cands = r.search_dense(&q).await?;
            let mut docs = doc_ids_in_order(&cands);
            docs.truncate(MAX_DOC_K);
            outcomes.push(QueryOutcome {
                query_id: gp.query_id,
                retrieved: docs,
                relevant: vec![gp.gold_doc_id],
            });
        }
        let rep = aggregate(&outcomes);
        let (p50, p95, p99) = r.with_stats(|s| (s.total.p50_ms(), s.total.p95_ms(), s.total.p99_ms()));
        println!(
            "{ef:>7} {p50:>9.2} {p95:>9.2} {p99:>9.2} {:>10.3} {:>10.3} {:>10.3} {:>9.3}",
            rep.recall_at_10, rep.recall_at_50, rep.recall_at_100, rep.mrr
        );
    }

    // Per-stage breakdown at the selected setting: this is what says *which*
    // stage to fix when the budget is missed.
    println!();
    let mut r = Retriever::new(
        Qdrant::from_url("http://127.0.0.1:6334").build()?,
        embedder.clone(),
        &collection,
        SearchTuning::default(),
    );
    if let Some(m) = &metrics {
        r = r.with_metrics(m.clone());
    }
    let mut warm_q = Query::new(queries[0].query_title.clone(), "default");
    warm_q.top_k = MAX_DOC_K * CHUNK_FETCH_MULTIPLE;
    let cold = Instant::now();
    let _ = r.search_dense(&warm_q).await?;
    let cold_ms = cold.elapsed().as_secs_f64() * 1000.0;
    for q in queries.iter().take(20) {
        let mut qq = Query::new(q.query_title.clone(), "default");
        qq.top_k = MAX_DOC_K * CHUNK_FETCH_MULTIPLE;
        let _ = r.search_dense(&qq).await;
    }
    r.reset_stats();

    let mut outcomes = Vec::with_capacity(queries.len());
    let mut distinct_docs: Vec<usize> = Vec::new();
    for gp in &queries {
        let mut q = Query::new(gp.query_title.clone(), "default");
        q.top_k = MAX_DOC_K * CHUNK_FETCH_MULTIPLE;
        let cands = r.search_dense(&q).await?;
        let mut docs = doc_ids_in_order(&cands);
        distinct_docs.push(docs.len());
        docs.truncate(MAX_DOC_K);
        outcomes.push(QueryOutcome {
            query_id: gp.query_id,
            retrieved: docs,
            relevant: vec![gp.gold_doc_id],
        });
    }
    distinct_docs.sort_unstable();
    let rep = aggregate(&outcomes);

    println!("per-stage at hnsw_ef={} (the 25ms budget)", SearchTuning::default().hnsw_ef);
    print!("{}", r.with_stats(|s| s.render()));
    println!();
    println!("cold first query   {cold_ms:.2} ms   (reported separately, never in the p95)");
    println!();
    println!("recall@1           {:.3}", rep.recall_at_1);
    println!("recall@5           {:.3}", rep.recall_at_5);
    println!("recall@10          {:.3}", rep.recall_at_10);
    println!("recall@50          {:.3}   <- SLO floor is 0.85", rep.recall_at_50);
    println!("recall@100         {:.3}", rep.recall_at_100);
    println!("nDCG@10            {:.3}", rep.ndcg_at_10);
    println!("MRR                {:.3}", rep.mrr);
    println!("complete misses    {} of {}", rep.misses.len(), rep.queries);
    println!();
    println!(
        "distinct docs per query from {} chunks: p50={} min={}",
        MAX_DOC_K * CHUNK_FETCH_MULTIPLE,
        distinct_docs[distinct_docs.len() / 2],
        distinct_docs.first().copied().unwrap_or(0)
    );
    println!(
        "  (must be >= {MAX_DOC_K} for recall@{MAX_DOC_K} to mean what it says)"
    );

    let p95 = r.with_stats(|s| s.total.p95_ms());
    let budget = tz_core::SloBudget::default();
    println!();
    println!(
        "SLO retrieval p95  {:.2} ms vs {:.0} ms budget   {}",
        p95,
        budget.retrieval_p95_ms,
        if p95 <= budget.retrieval_p95_ms { "PASS" } else { "FAIL" }
    );
    println!(
        "SLO recall@50      {:.3} vs {:.2} floor          {}",
        rep.recall_at_50,
        budget.recall_at_50_min,
        if rep.recall_at_50 >= budget.recall_at_50_min { "PASS" } else { "FAIL" }
    );
    println!();
    println!("note: duplicate judgements name *a* relevant document, not every");
    println!("      one, so these recall figures are a lower bound.");

    // Flush before exit. Dropping the provider without this loses whatever is
    // still batched, which on a short run is most of the data.
    if let Some(mp) = meter_provider {
        let _ = mp.force_flush();
        let _ = mp.shutdown();
    }
    Ok(())
}
