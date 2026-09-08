//! Sweeps encoder / thread-count / sequence-length combinations and reports
//! query-embedding latency for each.
//!
//! This exists because the first honest measurement of the default
//! configuration came in at 11.5ms p95 for a single short query -- 46% of the
//! entire 25ms retrieval budget, for a 33M-parameter model on a ten-token
//! input. That is not a number to accept; it is a number to explain.
//!
//! Run: cargo run --release -p tz-embed --example encoder_sweep

use std::time::Instant;
use tz_embed::{EmbedderConfig, Encoder, LatencyStats, Embedder};

const QUERIES: &[&str] = &[
    "nginx returns 502 bad gateway after upgrade",
    "how do I rotate systemd journal logs",
    "postgres connection pool exhausted under load",
    "kubernetes pod stuck in CrashLoopBackOff",
    "why is my TLS certificate chain incomplete",
    "apache mod_rewrite infinite redirect loop",
    "raid array degraded after reboot mdadm",
    "iptables rule not matching forwarded traffic",
];

const WARMUP: usize = 50;
const SAMPLES: usize = 300;

fn measure(cfg: EmbedderConfig) -> Option<(LatencyStats, f64)> {
    let load0 = Instant::now();
    let e = match Embedder::with_config(cfg) {
        Ok(e) => e,
        Err(err) => {
            println!("  !! {err}");
            return None;
        }
    };
    let load_ms = load0.elapsed().as_secs_f64() * 1000.0;

    for i in 0..WARMUP {
        let _ = e.embed_query(QUERIES[i % QUERIES.len()]);
    }
    // Discard warm-up: the first inferences pay for lazy allocation and kernel
    // selection, and folding those into a p95 would misreport steady state.
    let mut s = LatencyStats::new("q");
    for i in 0..SAMPLES {
        let q = QUERIES[i % QUERIES.len()];
        let t = Instant::now();
        if e.embed_query(q).is_err() {
            return None;
        }
        s.record(t.elapsed().as_micros() as u64);
    }
    Some((s, load_ms))
}

fn main() {
    println!("TierZero encoder sweep -- query embedding latency");
    println!("{} warmup + {} sampled single-query embeds per row\n", WARMUP, SAMPLES);
    println!(
        "{:<26} {:>7} {:>6} {:>8} {:>8} {:>8} {:>8} {:>9}",
        "encoder", "threads", "maxlen", "p50 ms", "p95 ms", "p99 ms", "max ms", "load ms"
    );
    println!("{}", "-".repeat(88));

    let encoders = [
        ("bge-small-fp32", Encoder::BgeSmallEnV15),
        ("bge-small-int8", Encoder::BgeSmallEnV15Int8),
        ("minilm-l6-fp32", Encoder::AllMiniLmL6V2),
    ];
    let threads = [Some(1usize), Some(2), Some(4), None];
    let lengths = [64usize, 512];

    let mut best: Option<(f64, String)> = None;

    for (label, enc) in encoders {
        for t in threads {
            for ml in lengths {
                let cfg = EmbedderConfig {
                    encoder: enc,
                    intra_threads: t,
                    max_length: ml,
                    batch_size: 1,
                };
                let Some((s, load_ms)) = measure(cfg) else { continue };
                let tname = t.map(|n| n.to_string()).unwrap_or_else(|| "default".into());
                println!(
                    "{:<26} {:>7} {:>6} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>9.0}",
                    label,
                    tname,
                    ml,
                    s.p50_ms(),
                    s.p95_ms(),
                    s.p99_ms(),
                    s.max_ms(),
                    load_ms
                );
                let tag = format!("{label} threads={tname} maxlen={ml}");
                if best.as_ref().is_none_or(|(b, _)| s.p95_ms() < *b) {
                    best = Some((s.p95_ms(), tag));
                }
            }
        }
    }

    if let Some((p95, tag)) = best {
        println!("\nbest p95: {p95:.2}ms  [{tag}]");
        println!(
            "retrieval budget is 25ms total; this stage should stay under ~9ms \
             to leave room for search, fusion and payload fetch."
        );
    }
}
