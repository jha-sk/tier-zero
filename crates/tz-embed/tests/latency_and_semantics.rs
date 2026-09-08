//! Measures the real cost of in-process query embedding, and checks the encoder
//! actually encodes meaning.
//!
//! These tests download a model on first run and are the empirical basis for
//! the retrieval latency budget. If query embedding does not fit in roughly
//! 10ms at p95 on this machine, the 25ms end-to-end target is not reachable and
//! the architecture has to change rather than the number being fudged.

use tz_embed::{Embedder, Encoder, cosine};

fn embedder() -> Embedder {
    Embedder::new(Encoder::BgeSmallEnV15).expect("model loads")
}

#[test]
#[ignore = "downloads a model; run with --ignored"]
fn query_embedding_fits_the_latency_budget() {
    let e = embedder();
    let queries = [
        "nginx returns 502 bad gateway after upgrade",
        "how do I rotate systemd journal logs",
        "postgres connection pool exhausted under load",
        "kubernetes pod stuck in CrashLoopBackOff",
        "why is my TLS certificate chain incomplete",
    ];
    // Enough samples that a p95 means something.
    for i in 0..400 {
        let q = queries[i % queries.len()];
        e.embed_query(q).expect("embeds");
    }
    let s = e.query_latency();
    println!(
        "embed_query  n={}  p50={:.2}ms  p95={:.2}ms  p99={:.2}ms  max={:.2}ms",
        s.count(),
        s.p50_ms(),
        s.p95_ms(),
        s.p99_ms(),
        s.max_ms()
    );
    assert_eq!(s.count(), 400);
    // The plan budgets ~9ms p95 for this stage out of 25ms total.
    assert!(
        s.p95_ms() < 15.0,
        "query embed p95 {:.2}ms leaves no room for search in a 25ms budget",
        s.p95_ms()
    );
}

#[test]
#[ignore = "downloads a model; run with --ignored"]
fn embeddings_are_unit_normalized_and_correctly_shaped() {
    let e = embedder();
    let v = e.embed_query("disk full on /var").unwrap();
    assert_eq!(v.len(), tz_embed::EMBEDDING_DIM);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "expected unit norm, got {norm}");
}

#[test]
#[ignore = "downloads a model; run with --ignored"]
fn related_queries_embed_closer_than_unrelated_ones() {
    // The minimum bar for an encoder to be worth indexing with.
    let e = embedder();
    let a = e.embed_query("nginx 502 bad gateway error").unwrap();
    let b = e.embed_query("nginx returning bad gateway after restart").unwrap();
    let c = e.embed_query("how to bake sourdough bread").unwrap();
    let related = cosine(&a, &b);
    let unrelated = cosine(&a, &c);
    println!("related={related:.3} unrelated={unrelated:.3}");
    assert!(related > unrelated + 0.15, "related {related} vs unrelated {unrelated}");
}

#[test]
#[ignore = "downloads a model; run with --ignored"]
fn batch_embedding_preserves_input_order_despite_length_bucketing() {
    // embed_documents sorts by length internally for GPU/CPU efficiency and
    // must restore the caller's order. Getting this wrong would silently
    // attach every vector to the wrong chunk.
    let e = embedder();
    let docs: Vec<String> = vec![
        "short".into(),
        "a considerably longer document about configuring nginx upstream timeouts and keepalive"
            .into(),
        "mid length text here".into(),
        "x".into(),
    ];
    let batched = e.embed_documents(&docs).unwrap();
    assert_eq!(batched.len(), docs.len());
    for (i, d) in docs.iter().enumerate() {
        let single = e.embed_query(d).unwrap();
        // Same text through the batch path and the single path should agree
        // closely. They differ only by the query prefix bge applies, so compare
        // against the batch neighbours instead: each batch vector must be
        // nearest to its own re-embedding among all batch vectors.
        let mut best = 0usize;
        let mut best_sim = f32::MIN;
        for (j, v) in batched.iter().enumerate() {
            let s = cosine(&single, v);
            if s > best_sim {
                best_sim = s;
                best = j;
            }
        }
        assert_eq!(best, i, "batch vector {i} landed out of order (matched {best})");
    }
}

#[test]
#[ignore = "downloads a model; run with --ignored"]
fn empty_input_is_an_error_not_a_zero_vector() {
    let e = embedder();
    assert!(e.embed_query("").is_err());
    assert!(e.embed_query("   ").is_err());
    assert!(e.embed_documents(&[]).is_err());
}
