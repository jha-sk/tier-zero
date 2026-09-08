//! Open-model load generation with corrected latency accounting.

use hdrhistogram::Histogram;
use std::time::{Duration, Instant};

/// A fixed arrival schedule, decided before the run starts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schedule {
    pub rate_rps: f64,
    pub duration: Duration,
    /// Requests sent before measurement begins, to warm caches and connections.
    pub warmup: usize,
}

impl Schedule {
    pub fn new(rate_rps: f64, duration: Duration) -> Self {
        Self { rate_rps, duration, warmup: 20 }
    }

    /// Number of requests this schedule will issue.
    pub fn request_count(&self) -> usize {
        (self.rate_rps * self.duration.as_secs_f64()) as usize
    }

    /// Intended send offset for request `i`, from the start of the run.
    ///
    /// Deterministic rather than sampled from an exponential distribution: a
    /// fixed-rate schedule is reproducible run to run, which matters more here
    /// than modelling Poisson arrivals, and it removes a source of variance
    /// when comparing two builds.
    pub fn offset_for(&self, i: usize) -> Duration {
        Duration::from_secs_f64(i as f64 / self.rate_rps)
    }
}

/// Latency and error accounting for one load run.
pub struct LoadReport {
    /// Latency from *intended* send time. The honest number.
    corrected: Histogram<u64>,
    /// Latency from actual send time. Reported alongside so the gap between
    /// them is visible — that gap is the queueing the system imposed.
    uncorrected: Histogram<u64>,
    /// How late each request actually started.
    scheduling_delay: Histogram<u64>,
    pub sent: u64,
    pub errors: u64,
    pub schedule: Schedule,
}

impl LoadReport {
    pub fn new(schedule: Schedule) -> Self {
        let h = || Histogram::new_with_bounds(1, 300_000_000, 3).expect("valid bounds");
        Self {
            corrected: h(),
            uncorrected: h(),
            scheduling_delay: h(),
            sent: 0,
            errors: 0,
            schedule,
        }
    }

    /// Record one request.
    ///
    /// `intended` and `actual` are offsets from the run start; `service_us` is
    /// how long the request itself took.
    pub fn record(&mut self, intended: Duration, actual: Duration, service_us: u64) {
        let delay_us = actual.saturating_sub(intended).as_micros() as u64;
        self.uncorrected.saturating_record(service_us);
        // Corrected latency charges the request for time it spent waiting to be
        // sent. Without this term, a saturated system reports the latency of the
        // few requests it managed to start on time.
        self.corrected.saturating_record(service_us + delay_us);
        self.scheduling_delay.saturating_record(delay_us);
        self.sent += 1;
    }

    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    fn q(h: &Histogram<u64>, quantile: f64) -> f64 {
        h.value_at_quantile(quantile) as f64 / 1000.0
    }

    pub fn corrected_p50(&self) -> f64 {
        Self::q(&self.corrected, 0.50)
    }
    pub fn corrected_p95(&self) -> f64 {
        Self::q(&self.corrected, 0.95)
    }
    pub fn corrected_p99(&self) -> f64 {
        Self::q(&self.corrected, 0.99)
    }
    pub fn uncorrected_p95(&self) -> f64 {
        Self::q(&self.uncorrected, 0.95)
    }
    pub fn max_scheduling_delay_ms(&self) -> f64 {
        self.scheduling_delay.max() as f64 / 1000.0
    }

    /// True when the generator could not keep up with its own schedule, which
    /// means the offered rate exceeded what the system could absorb. Reporting
    /// a latency figure from such a run without saying so is the most common
    /// way a load test misleads.
    pub fn fell_behind(&self) -> bool {
        self.max_scheduling_delay_ms() > 50.0
    }

    pub fn error_rate(&self) -> f64 {
        if self.sent + self.errors == 0 {
            return 0.0;
        }
        self.errors as f64 / (self.sent + self.errors) as f64
    }

    pub fn render(&self) -> String {
        format!(
            "rate {:>6.0}/s  n={:<6} err={:<4} corrected p50/p95/p99 {:>7.2}/{:>7.2}/{:>7.2} ms   \
             uncorrected p95 {:>7.2} ms   max sched delay {:>8.2} ms{}",
            self.schedule.rate_rps,
            self.sent,
            self.errors,
            self.corrected_p50(),
            self.corrected_p95(),
            self.corrected_p99(),
            self.uncorrected_p95(),
            self.max_scheduling_delay_ms(),
            if self.fell_behind() { "  << GENERATOR FELL BEHIND" } else { "" }
        )
    }
}

/// Drives an async operation against a schedule.
pub struct LoadTest {
    pub schedule: Schedule,
}

impl LoadTest {
    pub fn new(schedule: Schedule) -> Self {
        Self { schedule }
    }

    /// Run `op` against the schedule. `op` receives the request index.
    pub async fn run<F, Fut>(&self, op: F) -> LoadReport
    where
        F: Fn(usize) -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<()>>,
    {
        let mut report = LoadReport::new(self.schedule);

        for i in 0..self.schedule.warmup {
            let _ = op(i).await;
        }

        let start = Instant::now();
        let n = self.schedule.request_count();
        for i in 0..n {
            let intended = self.schedule.offset_for(i);
            // Wait until this request's scheduled time. If we are already past
            // it, send immediately -- and the lateness is charged below.
            let now = start.elapsed();
            if intended > now {
                tokio::time::sleep(intended - now).await;
            }
            let actual = start.elapsed();
            let t = Instant::now();
            match op(i).await {
                Ok(()) => {
                    report.record(intended, actual, t.elapsed().as_micros() as u64);
                }
                Err(_) => report.record_error(),
            }
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_schedule_is_fixed_in_advance_and_evenly_spaced() {
        let s = Schedule::new(100.0, Duration::from_secs(2));
        assert_eq!(s.request_count(), 200);
        assert_eq!(s.offset_for(0), Duration::ZERO);
        assert_eq!(s.offset_for(100), Duration::from_secs(1));
        assert_eq!(s.offset_for(200), Duration::from_secs(2));
    }

    #[test]
    fn corrected_latency_charges_a_request_for_waiting_to_be_sent() {
        // This is the whole point of the module. A request that should have
        // been sent at t=0, was sent at t=100ms, and took 10ms to serve, was
        // 110ms late from the client's point of view -- not 10ms.
        let mut r = LoadReport::new(Schedule::new(10.0, Duration::from_secs(1)));
        r.record(Duration::ZERO, Duration::from_millis(100), 10_000);
        assert!((r.corrected_p50() - 110.0).abs() < 1.0, "got {}", r.corrected_p50());
        assert!((r.uncorrected_p95() - 10.0).abs() < 1.0, "got {}", r.uncorrected_p95());
    }

    #[test]
    fn the_two_measures_agree_when_the_generator_keeps_up() {
        let mut r = LoadReport::new(Schedule::new(10.0, Duration::from_secs(1)));
        for i in 0..100 {
            let t = Duration::from_millis(i * 100);
            r.record(t, t, 5_000);
        }
        assert!((r.corrected_p95() - r.uncorrected_p95()).abs() < 1.0);
        assert!(!r.fell_behind());
    }

    #[test]
    fn falling_behind_is_reported_rather_than_hidden_in_the_percentile() {
        let mut r = LoadReport::new(Schedule::new(1000.0, Duration::from_secs(1)));
        for i in 0..100 {
            // Every request starts 500ms late: the generator cannot keep up.
            let intended = Duration::from_millis(i);
            r.record(intended, intended + Duration::from_millis(500), 5_000);
        }
        assert!(r.fell_behind(), "a saturated run must say so");
        assert!(r.render().contains("FELL BEHIND"));
        // And the corrected number reflects it while the naive one does not.
        assert!(r.corrected_p95() > 500.0);
        assert!(r.uncorrected_p95() < 10.0);
    }

    #[test]
    fn errors_are_counted_separately_and_not_folded_into_latency() {
        // Computing percentiles only over successes hides timeouts entirely.
        let mut r = LoadReport::new(Schedule::new(10.0, Duration::from_secs(1)));
        r.record(Duration::ZERO, Duration::ZERO, 1_000);
        r.record_error();
        r.record_error();
        assert_eq!(r.sent, 1);
        assert_eq!(r.errors, 2);
        assert!((r.error_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn error_rate_is_zero_rather_than_nan_on_an_empty_run() {
        assert_eq!(LoadReport::new(Schedule::new(1.0, Duration::from_secs(1))).error_rate(), 0.0);
    }

    #[tokio::test]
    async fn the_driver_issues_the_scheduled_number_of_requests() {
        let lt = LoadTest::new(Schedule { rate_rps: 500.0, duration: Duration::from_millis(200), warmup: 2 });
        let report = lt.run(|_i| async { Ok(()) }).await;
        assert_eq!(report.sent, 100);
        assert_eq!(report.errors, 0);
    }
}
