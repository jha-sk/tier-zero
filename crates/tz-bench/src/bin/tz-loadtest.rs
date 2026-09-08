//! Ramps offered load against the retrieval path and finds the knee.
//!
//! Reports latency corrected for coordinated omission. The uncorrected figure
//! is printed beside it, because the gap between the two *is* the queueing the
//! system imposed, and a run where they diverge is a run where the naive number
//! is describing a load level that never happened.
//!
//! The knee is where p95 starts climbing steeply, not where errors begin. By
//! the time requests fail the system has been unusable for a while.

use std::sync::Arc;
use std::time::Duration;
use qdrant_client::Qdrant;
use tz_bench::{LoadTest, Schedule};
use tz_core::Query;
use tz_embed::{Embedder, EmbedderConfig, Encoder};
use tz_eval::golden::GoldenSet;
use tz_retrieve::search::{Retriever, SearchTuning};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let collection = std::env::args().nth(1).unwrap_or_else(|| "tz_10k".into());
    let golden = std::env::args().nth(2).unwrap_or_else(|| "evals/golden/serverfault.json".into());
    let seconds: u64 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(10);

    let gs = GoldenSet::load(&golden)?;
    let queries: Vec<String> =
        gs.pairs.iter().take(2000).map(|p| p.query_title.clone()).collect();
    anyhow::ensure!(!queries.is_empty(), "no queries");

    let client = Qdrant::from_url("http://127.0.0.1:6334").build()?;
    let points = client
        .collection_info(&collection)
        .await?
        .result
        .and_then(|r| r.points_count)
        .unwrap_or(0);

    let embedder =
        Arc::new(Embedder::with_config(EmbedderConfig::for_queries(Encoder::BgeSmallEnV15))?);
    let retriever = Arc::new(Retriever::new(
        Qdrant::from_url("http://127.0.0.1:6334").build()?,
        embedder,
        &collection,
        SearchTuning::default(),
    ));

    println!("collection {collection}  points {points}");
    println!("open-model load, constant arrival rate, {seconds}s per step");
    println!("latency is corrected for coordinated omission\n");

    // Ramp until the generator can no longer keep to its own schedule.
    for rate in [10.0f64, 25.0, 50.0, 100.0, 200.0, 400.0] {
        let sched = Schedule { rate_rps: rate, duration: Duration::from_secs(seconds), warmup: 10 };
        let lt = LoadTest::new(sched);
        let r = retriever.clone();
        let qs = queries.clone();
        let report = lt
            .run(move |i| {
                let r = r.clone();
                let q = qs[i % qs.len()].clone();
                async move {
                    let mut query = Query::new(q, "default");
                    query.top_k = 50;
                    r.search_dense(&query).await.map(|_| ())
                }
            })
            .await;
        println!("{}", report.render());
        if report.fell_behind() {
            println!("\nknee reached: the generator could not sustain {rate:.0}/s.");
            println!("Numbers above this rate would describe a load the system never served.");
            break;
        }
    }
    Ok(())
}
