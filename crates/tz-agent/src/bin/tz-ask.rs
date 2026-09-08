//! Ask the knowledge base a question. The end-to-end path in one command.
//!
//!   tz-ask "why is nginx returning 502 after an upgrade"
//!
//! Prints the answer as it streams, then the citations and the receipt: which
//! tier served it, what it cost, time to first token, total time. The receipt
//! is not decoration -- a system with cost and latency SLOs should show its
//! working on every request, not only in a benchmark.

use std::sync::Arc;
use qdrant_client::Qdrant;
use tz_agent::answer::AnswerPipeline;
use tz_agent::anthropic::Client;
use tz_core::{PriceTable, Query, SloBudget};
use tz_embed::{Embedder, EmbedderConfig, Encoder};
use tz_retrieve::search::{Retriever, SearchTuning};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: tz-ask <question>");
        eprintln!("  env: TZ_COLLECTION (default tz_10k), ANTHROPIC_API_KEY");
        std::process::exit(2);
    }
    let question = args.join(" ");
    let collection = std::env::var("TZ_COLLECTION").unwrap_or_else(|_| "tz_10k".into());
    let budget = SloBudget::default();

    // --- retrieval -------------------------------------------------------
    let embedder =
        Arc::new(Embedder::with_config(EmbedderConfig::for_queries(Encoder::BgeSmallEnV15))?);
    let mut retriever = Retriever::new(
        Qdrant::from_url("http://127.0.0.1:6334").build()?,
        embedder,
        &collection,
        SearchTuning::default(),
    );
    // Hybrid is available when statistics exist, but measured *worse* than
    // dense on this corpus (docs/benchmarks/retriever-ab.md), so dense is the
    // default and hybrid is opt-in.
    if std::env::var("TZ_HYBRID").is_ok()
        && let Ok(stats) = tz_retrieve::bm25::Bm25Stats::load("data/bm25-stats.json")
    {
        retriever = retriever.with_bm25(stats);
        eprintln!("(hybrid retrieval enabled)");
    }

    let mut q = Query::new(&question, "default");
    // Retrieve deep, pass few: recall climbs steeply with depth, and the
    // pipeline deduplicates to one chunk per document before the model sees it.
    q.top_k = 60;
    let t_retrieve = std::time::Instant::now();
    let candidates = if retriever.has_bm25() {
        retriever.search_hybrid(&q).await?
    } else {
        retriever.search_dense(&q).await?
    };
    let retrieve_ms = t_retrieve.elapsed().as_secs_f64() * 1000.0;

    if candidates.is_empty() {
        println!("No relevant entries found in the knowledge base.");
        return Ok(());
    }

    // --- generation ------------------------------------------------------
    let prices = PriceTable::load("config/prices.toml")?;
    let client = match Client::from_env() {
        Ok(c) => c,
        Err(e) => {
            // The degradation ladder's top rung, exercised for real: no
            // credentials is exactly the condition it exists for.
            eprintln!("\n{e}\n-> falling back to retrieval-only\n");
            let a = AnswerPipeline::degraded_answer(&candidates);
            println!("{}", a.text);
            println!("---");
            println!("tier         degraded (retrieval only)");
            println!("retrieval    {retrieve_ms:.1} ms");
            println!("cost         $0.000000");
            return Ok(());
        }
    };

    let pipeline = AnswerPipeline::new(client, prices, budget.cost_p95_usd);
    let route = pipeline.route_for(&question, false);
    eprintln!(
        "routed to {} ({}), context cap {} chunks\n",
        route.tier.as_str(),
        route.reason,
        route.context_cap
    );

    let mut first = true;
    let answer = pipeline
        .answer(&question, &candidates, |tok| {
            if first {
                first = false;
            }
            print!("{tok}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        })
        .await?;

    // --- receipt ---------------------------------------------------------
    println!("\n");
    if answer.citations.is_empty() {
        println!("(no citations -- treat this answer with suspicion)");
    } else {
        println!("sources");
        for c in &answer.citations {
            println!("  [{}] {}", c.index, c.source_uri);
        }
    }
    println!();
    let pass = |ok: bool| if ok { "OK" } else { "OVER" };
    println!("tier         {} -> {}", answer.tier.as_str(), answer.model.as_deref().unwrap_or("-"));
    println!(
        "context      {} chunks{}",
        answer.context_chunks,
        if answer.context_trimmed { " (trimmed to fit budget)" } else { "" }
    );
    println!("retrieval    {retrieve_ms:.1} ms");
    println!(
        "TTFT         {:.0} ms   [budget {:.0} ms  {}]",
        answer.ttft_ms,
        budget.ttft_p95_ms,
        pass(answer.ttft_ms <= budget.ttft_p95_ms)
    );
    println!(
        "total        {:.0} ms   [budget {:.0} ms  {}]",
        answer.total_ms,
        budget.total_p95_ms,
        pass(answer.total_ms <= budget.total_p95_ms)
    );
    println!(
        "cost         ${:.6}   [budget ${:.2}  {}]",
        answer.cost_usd,
        budget.cost_p95_usd,
        pass(answer.cost_usd <= budget.cost_p95_usd)
    );
    println!(
        "tokens       in {} / out {} / cache read {} / cache write {}",
        answer.usage.input_tokens,
        answer.usage.output_tokens,
        answer.usage.cache_read_input_tokens,
        answer.usage.cache_creation_input_tokens
    );
    if let Some(r) = answer.usage.cache_hit_ratio() {
        println!("cache        {:.0}% of input served from cache", r * 100.0);
    }
    Ok(())
}
