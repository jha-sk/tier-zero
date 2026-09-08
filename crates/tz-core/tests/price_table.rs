//! The shipped price table must parse and must price every model the system
//! can route to. A missing entry is a request that silently costs nothing,
//! which is how a cost SLO reports passing while it is not.

use tz_core::cost::{CacheTtl, PriceTable, Usage};

fn table() -> PriceTable {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/prices.toml");
    PriceTable::load(p).expect("config/prices.toml must parse")
}

#[test]
fn shipped_price_table_parses_and_is_dated() {
    let t = table();
    assert!(!t.version.is_empty());
    assert!(!t.fetched_at.is_empty(), "an undated price table is not reproducible");
    assert!(t.source.starts_with("https://"));
    assert_eq!(t.currency, "USD");
}

#[test]
fn every_routable_model_has_a_price() {
    let t = table();
    for m in ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"] {
        t.price_for(m).unwrap_or_else(|e| panic!("{m}: {e}"));
    }
}

#[test]
fn cache_multipliers_match_anthropics_published_terms() {
    let t = table();
    for m in ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"] {
        let p = t.price_for(m).unwrap();
        let rel = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(rel(p.cache_write_5m_per_mtok, p.input_per_mtok * 1.25), "{m} 5m write");
        assert!(rel(p.cache_write_1h_per_mtok, p.input_per_mtok * 2.00), "{m} 1h write");
        assert!(rel(p.cache_read_per_mtok, p.input_per_mtok * 0.10), "{m} read");
        assert!(rel(p.output_per_mtok, p.input_per_mtok * 5.0), "{m} output is 5x input");
    }
}

/// The arithmetic that forced the tiered architecture. If this test ever fails
/// because prices moved, the routing thresholds need re-tuning, not the test.
#[test]
fn the_six_cent_budget_is_what_rules_out_opus_as_a_default() {
    let t = table();
    let typical = Usage { input_tokens: 20_000, output_tokens: 1_000, ..Default::default() };
    let budget = tz_core::SloBudget::default().cost_p95_usd;

    let haiku = t.cost_of("claude-haiku-4-5", &typical, CacheTtl::FiveMinutes).unwrap().total_usd();
    let sonnet = t.cost_of("claude-sonnet-5", &typical, CacheTtl::FiveMinutes).unwrap().total_usd();
    let opus = t.cost_of("claude-opus-5", &typical, CacheTtl::FiveMinutes).unwrap().total_usd();

    assert!(haiku < budget * 0.5, "haiku {haiku} leaves room for retries");
    assert!(sonnet < budget, "sonnet {sonnet} fits");
    assert!(opus > budget, "opus {opus} does not fit at this context size");
}

/// Prompt caching is the lever that lets a stronger model stay inside budget.
#[test]
fn caching_brings_opus_back_under_budget_at_the_same_context_size() {
    let t = table();
    let cached = Usage {
        input_tokens: 2_000,
        output_tokens: 1_000,
        cache_read_input_tokens: 18_000,
        ..Default::default()
    };
    let c = t.cost_of("claude-opus-5", &cached, CacheTtl::FiveMinutes).unwrap();
    assert!(
        c.total_usd() < tz_core::SloBudget::default().cost_p95_usd,
        "cached opus {} should fit",
        c.total_usd()
    );
}

#[test]
fn local_embedding_is_priced_at_zero_dollars_deliberately() {
    let t = table();
    let p = t.price_for("bge-small-en-v1.5-int8").unwrap();
    assert_eq!(p.input_per_mtok, 0.0, "in-process ONNX costs latency and RAM, not dollars");
}
