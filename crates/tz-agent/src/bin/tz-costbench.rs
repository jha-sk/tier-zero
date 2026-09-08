//! Measures the per-request cost distribution without calling the API.
//!
//! Cost is arithmetic on token counts against a rate card. The API does not
//! compute the price -- `config/prices.toml` does -- so the only thing a live
//! call would contribute is the token count, and prompts can be counted
//! locally. What this measures is what the *architecture* controls: how many
//! tokens the retrieval and routing decisions put in front of a model.
//!
//! Two honest limits, both stated in the output rather than buried:
//!
//! 1. **Tokenizer proxy.** Counting uses the cached BERT-wordpiece tokenizer
//!    from the embedding model, not Anthropic's BPE tokenizer. On English prose
//!    the two are close; on code-dense text wordpiece splits more aggressively,
//!    so this estimate runs *high* on this corpus. A conservative error is the
//!    right direction for a budget.
//! 2. **Output length is bounded, not predicted.** Only the model decides how
//!    much it writes, so cost is charged against `max_tokens` -- the enforced
//!    ceiling. That is worst case, which is what a p95 budget is about.

use std::sync::Arc;
use qdrant_client::Qdrant;
use tz_agent::answer::{SYSTEM, dedupe_by_document, render_sources};
use tz_agent::router::{TierModels, route};
use tz_core::{CacheTtl, PriceTable, Query, RouteTier, SloBudget, Usage};
use tz_embed::{Embedder, EmbedderConfig, Encoder};
use tz_eval::golden::GoldenSet;
use tz_ingest::chunker::{HfCounter, TokenCounter};
use tz_retrieve::search::{Retriever, SearchTuning};

/// Enforced output ceiling. Cost is charged against this, not an average.
const MAX_OUTPUT: u64 = 900;

/// Anthropic's minimum cacheable prefix, in tokens, by model tier.
/// A `cache_control` marker on a shorter prefix is **silently ignored** -- no
/// error, no warning, and `cache_read_input_tokens` simply stays at zero.
fn min_cacheable_prefix(model: &str) -> u64 {
    if model.contains("haiku") { 2048 } else { 1024 }
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[(((sorted.len() - 1) as f64) * q) as usize]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let collection = std::env::var("TZ_COLLECTION").unwrap_or_else(|_| "tz_10k".into());
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(300);

    let prices = PriceTable::load("config/prices.toml")?;
    let budget = SloBudget::default();
    let models = TierModels::default();
    let counter = HfCounter::from_file(
        HfCounter::find_cached(".").ok_or_else(|| anyhow::anyhow!("no cached tokenizer"))?,
    )?;

    let gs = GoldenSet::load("evals/golden/serverfault.json")?;
    let (_train, test) = gs.split(0.2);
    let queries: Vec<_> = test.into_iter().take(n).collect();

    let embedder =
        Arc::new(Embedder::with_config(EmbedderConfig::for_queries(Encoder::BgeSmallEnV15))?);
    let retriever = Retriever::new(
        Qdrant::from_url("http://127.0.0.1:6334").build()?,
        embedder,
        &collection,
        SearchTuning::default(),
    );

    println!("TierZero cost measurement");
    println!("price table {} (fetched {})", prices.version, prices.fetched_at);
    println!("{} held-out queries, real retrieval, real prompts\n", queries.len());

    let system_tokens = counter.count(SYSTEM) as u64;

    let mut costs: Vec<f64> = Vec::new();
    let mut costs_cached: Vec<f64> = Vec::new();
    let mut input_tokens: Vec<f64> = Vec::new();
    let mut tiers: std::collections::HashMap<&'static str, usize> = Default::default();
    let mut per_tier_cost: std::collections::HashMap<&'static str, Vec<f64>> = Default::default();
    let mut trimmed = 0usize;

    for gp in &queries {
        let question = &gp.query_title;
        let r = route(question, false, &models);
        *tiers.entry(r.tier.as_str()).or_default() += 1;

        let Some(model) = r.model.clone() else {
            // Tier zero never calls a model: a genuine zero, not a rounding.
            costs.push(0.0);
            costs_cached.push(0.0);
            input_tokens.push(0.0);
            per_tier_cost.entry(r.tier.as_str()).or_default().push(0.0);
            continue;
        };

        let mut q = Query::new(question.clone(), "default");
        q.top_k = 60;
        let cands = retriever.search_dense(&q).await?;
        let used = dedupe_by_document(&cands, r.context_cap);
        if used.len() < r.context_cap.min(cands.len()) {
            trimmed += 1;
        }

        // The prompt exactly as `answer.rs` builds it.
        let context = render_sources(&used);
        let tokens_in =
            system_tokens + counter.count(&context) as u64 + counter.count(question) as u64;
        input_tokens.push(tokens_in as f64);

        // Cold: nothing cached, full input billed at the input rate.
        let cold = prices.cost_of(
            &model,
            &Usage { input_tokens: tokens_in, output_tokens: MAX_OUTPUT, ..Default::default() },
            CacheTtl::FiveMinutes,
        )?;

        // Warm: the system prefix served from cache -- *if* it is long enough
        // to be cacheable at all.
        let warm = if system_tokens >= min_cacheable_prefix(&model) {
            prices.cost_of(
                &model,
                &Usage {
                    input_tokens: tokens_in - system_tokens,
                    cache_read_input_tokens: system_tokens,
                    output_tokens: MAX_OUTPUT,
                    ..Default::default()
                },
                CacheTtl::FiveMinutes,
            )?
        } else {
            cold.clone()
        };

        costs.push(cold.total_usd());
        costs_cached.push(warm.total_usd());
        per_tier_cost.entry(r.tier.as_str()).or_default().push(cold.total_usd());
    }

    costs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    costs_cached.sort_by(|a, b| a.partial_cmp(b).unwrap());
    input_tokens.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("routing");
    for t in [RouteTier::Zero, RouteTier::One, RouteTier::Two] {
        let c = tiers.get(t.as_str()).copied().unwrap_or(0);
        println!(
            "  {:<8} {:>4} requests ({:>4.1}%)",
            t.as_str(),
            c,
            100.0 * c as f64 / queries.len() as f64
        );
    }
    if trimmed > 0 {
        println!("  context trimmed on {trimmed} requests");
    }

    println!("\ninput tokens per request");
    println!(
        "  p50 {:.0}   p95 {:.0}   p99 {:.0}   max {:.0}",
        pct(&input_tokens, 0.50),
        pct(&input_tokens, 0.95),
        pct(&input_tokens, 0.99),
        input_tokens.last().copied().unwrap_or(0.0)
    );
    println!("  system prompt {system_tokens} tokens (fixed)");

    println!("\ncost per request (output charged at max_tokens={MAX_OUTPUT}, worst case)");
    println!(
        "  p50 ${:.5}   p95 ${:.5}   p99 ${:.5}   max ${:.5}",
        pct(&costs, 0.50),
        pct(&costs, 0.95),
        pct(&costs, 0.99),
        costs.last().copied().unwrap_or(0.0)
    );

    println!("\ncost by tier");
    for t in [RouteTier::One, RouteTier::Two] {
        if let Some(v) = per_tier_cost.get_mut(t.as_str()) {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !v.is_empty() {
                println!(
                    "  {:<8} p50 ${:.5}   p95 ${:.5}",
                    t.as_str(),
                    pct(v, 0.50),
                    pct(v, 0.95)
                );
            }
        }
    }

    let p95 = pct(&costs, 0.95);
    println!("\nSLO cost p95   ${p95:.5} vs ${:.2} budget   {}",
        budget.cost_p95_usd,
        if p95 <= budget.cost_p95_usd { "PASS" } else { "FAIL" });

    // Sensitivity: the tokenizer is a proxy, so report the budget verdict
    // across a band rather than at a single point.
    println!("\nsensitivity to tokenizer error (this uses a wordpiece proxy)");
    for adj in [0.8f64, 0.9, 1.0, 1.1, 1.2] {
        let v = p95 * adj;
        println!(
            "  tokens {:+3.0}%   p95 ${v:.5}   {}",
            (adj - 1.0) * 100.0,
            if v <= budget.cost_p95_usd { "PASS" } else { "FAIL" }
        );
    }

    // Prompt caching only pays if the stable prefix is long enough to cache.
    println!("\nprompt caching");
    for m in [&models.standard, &models.escalated] {
        let min = min_cacheable_prefix(m);
        if system_tokens < min {
            println!(
                "  {m}: system prefix is {system_tokens} tokens, minimum cacheable is {min}.",
            );
            println!("     The cache_control marker is SILENTLY IGNORED -- no error is raised");
            println!("     and cache_read_input_tokens stays at zero.");
        } else {
            let saving = pct(&costs, 0.95) - pct(&costs_cached, 0.95);
            println!("  {m}: cacheable, p95 saving ${saving:.5}");
        }
    }
    Ok(())
}
