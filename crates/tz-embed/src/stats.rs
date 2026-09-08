//! Latency recording.
//!
//! Percentiles are computed from an HDR histogram rather than by averaging,
//! because the claim this project makes is about a p95 and a mean cannot be
//! turned back into one.

use hdrhistogram::Histogram;

#[derive(Clone)]
pub struct LatencyStats {
    name: String,
    hist: Histogram<u64>,
}

impl std::fmt::Debug for LatencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LatencyStats")
            .field("name", &self.name)
            .field("count", &self.count())
            .field("p50_ms", &self.p50_ms())
            .field("p95_ms", &self.p95_ms())
            .field("p99_ms", &self.p99_ms())
            .finish()
    }
}

impl LatencyStats {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            // 1us..60s at 3 significant figures.
            hist: Histogram::new_with_bounds(1, 60_000_000, 3).expect("valid histogram bounds"),
        }
    }

    pub fn record(&mut self, micros: u64) {
        // Saturate rather than drop: an out-of-range sample is a real outlier
        // and silently discarding it would flatter the tail.
        self.hist.saturating_record(micros);
    }

    pub fn reset(&mut self) {
        self.hist.reset();
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn count(&self) -> u64 {
        self.hist.len()
    }

    fn q_ms(&self, q: f64) -> f64 {
        self.hist.value_at_quantile(q) as f64 / 1000.0
    }
    pub fn p50_ms(&self) -> f64 {
        self.q_ms(0.50)
    }
    pub fn p95_ms(&self) -> f64 {
        self.q_ms(0.95)
    }
    pub fn p99_ms(&self) -> f64 {
        self.q_ms(0.99)
    }
    pub fn max_ms(&self) -> f64 {
        self.hist.max() as f64 / 1000.0
    }
    pub fn mean_ms(&self) -> f64 {
        self.hist.mean() / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_track_the_recorded_distribution() {
        let mut s = LatencyStats::new("t");
        for i in 1..=1000 {
            s.record(i * 100); // 0.1ms .. 100ms
        }
        assert_eq!(s.count(), 1000);
        assert!((s.p50_ms() - 50.0).abs() < 1.0, "p50 {}", s.p50_ms());
        assert!((s.p95_ms() - 95.0).abs() < 1.0, "p95 {}", s.p95_ms());
        assert!(s.p99_ms() > s.p95_ms());
        assert!(s.p95_ms() > s.p50_ms());
    }

    #[test]
    fn the_tail_is_not_hidden_by_the_mean() {
        // 99 fast samples and one very slow one: the mean stays small while
        // p99 shows the outlier. This is the whole reason for the histogram.
        let mut s = LatencyStats::new("t");
        for _ in 0..99 {
            s.record(1_000); // 1ms
        }
        s.record(5_000_000); // 5s
        assert!(s.mean_ms() < 100.0, "mean {} hides it", s.mean_ms());
        assert!(s.max_ms() > 4_000.0, "max {} shows it", s.max_ms());
    }

    #[test]
    fn reset_clears_warmup_samples() {
        let mut s = LatencyStats::new("t");
        s.record(999_999);
        assert_eq!(s.count(), 1);
        s.reset();
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn out_of_range_samples_saturate_rather_than_vanish() {
        let mut s = LatencyStats::new("t");
        s.record(u64::MAX);
        assert_eq!(s.count(), 1, "an extreme outlier must still be counted");
    }
}
