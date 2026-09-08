//! Anthropic Messages API client.
//!
//! Raw HTTP rather than an SDK: there is no official Anthropic Rust SDK, and
//! raw HTTP is the sanctioned fallback for languages without one. That means
//! owning the retry policy, the SSE parsing and the usage accounting here
//! rather than inheriting them.
//!
//! Streaming is the default, not an option. Time to first token and total
//! generation time have different causes and different fixes -- TTFT is a
//! prefill problem, total is an output-length problem -- and a non-streaming
//! call cannot distinguish them.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tz_core::Usage;

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("no credentials: set ANTHROPIC_API_KEY")]
    NoCredentials,
    #[error("http {status}: {body}")]
    Status { status: u16, body: String },
    #[error("transport: {0}")]
    Transport(String),
    #[error("malformed response: {0}")]
    Malformed(String),
}

impl ApiError {
    /// Whether retrying could plausibly succeed.
    ///
    /// 429 and 5xx yes; 400 and 401 no. Retrying a malformed request burns
    /// budget and delays the user-visible failure without changing it.
    pub fn retryable(&self) -> bool {
        match self {
            ApiError::Status { status, .. } => *status == 429 || *status >= 500,
            ApiError::Transport(_) => true,
            _ => false,
        }
    }
}

/// A content block, optionally marked as a cache breakpoint.
#[derive(Debug, Clone, Serialize)]
pub struct Block {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: &'static str,
}

impl Block {
    pub fn text(s: impl Into<String>) -> Self {
        Self { kind: "text", text: s.into(), cache_control: None }
    }

    /// Mark this block as the end of a cacheable prefix.
    ///
    /// Everything up to and including this block is cached. Caching is a
    /// *prefix* mechanism, so anything volatile placed before a breakpoint
    /// invalidates the cache on every request and silently costs the 1.25x
    /// write premium without ever earning a read.
    pub fn cached(mut self) -> Self {
        self.cache_control = Some(CacheControl { kind: "ephemeral" });
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: &'static str,
    pub content: Vec<Block>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Request {
    pub model: String,
    pub max_tokens: u32,
    pub system: Vec<Block>,
    pub messages: Vec<Message>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

/// What a completed generation produced.
#[derive(Debug, Clone, Default)]
pub struct Completion {
    pub text: String,
    pub usage: Usage,
    pub model: String,
    pub stop_reason: Option<String>,
    /// Time from request start to the first token of output.
    pub ttft: Duration,
    /// Total wall time.
    pub total: Duration,
}

impl Completion {
    /// Output tokens per second during the decode phase.
    pub fn tokens_per_second(&self) -> f64 {
        let decode = self.total.saturating_sub(self.ttft).as_secs_f64();
        if decode <= 0.0 {
            return 0.0;
        }
        self.usage.output_tokens as f64 / decode
    }
}

/// Credentials resolved from the environment.
#[derive(Debug, Clone)]
pub enum Credential {
    ApiKey(String),
    /// OAuth bearer token; requires the oauth beta header.
    Bearer(String),
}

impl Credential {
    /// Resolve in the same order the official SDKs use.
    pub fn from_env() -> Result<Self, ApiError> {
        if let Ok(k) = std::env::var("ANTHROPIC_API_KEY")
            && !k.trim().is_empty()
        {
            return Ok(Credential::ApiKey(k));
        }
        if let Ok(t) = std::env::var("ANTHROPIC_AUTH_TOKEN")
            && !t.trim().is_empty()
        {
            return Ok(Credential::Bearer(t));
        }
        Err(ApiError::NoCredentials)
    }
}

pub struct Client {
    http: reqwest::Client,
    cred: Credential,
    max_retries: u32,
}

impl Client {
    pub fn from_env() -> Result<Self, ApiError> {
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .map_err(|e| ApiError::Transport(e.to_string()))?,
            cred: Credential::from_env()?,
            max_retries: 3,
        })
    }

    fn apply_auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.cred {
            Credential::ApiKey(k) => rb.header("x-api-key", k),
            // OAuth tokens go on Authorization, not x-api-key, and need the
            // beta header. Converting from an API key is a header change, not
            // a value swap.
            Credential::Bearer(t) => rb
                .header("authorization", format!("Bearer {t}"))
                .header("anthropic-beta", "oauth-2025-04-20"),
        }
    }

    /// Stream a completion, measuring TTFT.
    ///
    /// `on_token` is called with each text delta so a caller can render output
    /// as it arrives; the perceived-latency benefit of streaming only exists if
    /// something downstream actually consumes the deltas.
    pub async fn stream(
        &self,
        req: &Request,
        mut on_token: impl FnMut(&str),
    ) -> Result<Completion, ApiError> {
        let mut attempt = 0;
        loop {
            match self.stream_once(req, &mut on_token).await {
                Ok(c) => return Ok(c),
                Err(e) if e.retryable() && attempt < self.max_retries => {
                    // Full jitter: sleep uniformly in [0, base * 2^attempt].
                    // Equal or fixed backoff synchronises retries across
                    // concurrent callers and rebuilds the spike that caused
                    // the failure.
                    let cap = 8_000u64.min(500 * 2u64.pow(attempt));
                    let wait = fastrand_range(cap);
                    tokio::time::sleep(Duration::from_millis(wait)).await;
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn stream_once(
        &self,
        req: &Request,
        on_token: &mut impl FnMut(&str),
    ) -> Result<Completion, ApiError> {
        use futures::StreamExt;

        let t0 = Instant::now();
        let rb = self
            .http
            .post(API_URL)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(req);
        let resp = self
            .apply_auth(rb)
            .send()
            .await
            .map_err(|e| ApiError::Transport(e.to_string()))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ApiError::Status { status: status.as_u16(), body });
        }

        let mut out = Completion::default();
        let mut ttft: Option<Duration> = None;
        let mut buf = String::new();
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ApiError::Transport(e.to_string()))?;
            buf.push_str(&String::from_utf8_lossy(&chunk));

            // SSE frames are separated by a blank line; a frame may straddle
            // chunk boundaries, so only complete frames are consumed.
            while let Some(pos) = buf.find("\n\n") {
                let frame = buf[..pos].to_string();
                buf.drain(..pos + 2);
                for line in frame.lines() {
                    let Some(data) = line.strip_prefix("data: ") else { continue };
                    let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else { continue };
                    match v["type"].as_str().unwrap_or("") {
                        "message_start" => {
                            let u = &v["message"]["usage"];
                            out.usage.input_tokens = u["input_tokens"].as_u64().unwrap_or(0);
                            out.usage.cache_creation_input_tokens =
                                u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
                            out.usage.cache_read_input_tokens =
                                u["cache_read_input_tokens"].as_u64().unwrap_or(0);
                            out.model =
                                v["message"]["model"].as_str().unwrap_or_default().to_string();
                        }
                        "content_block_delta" => {
                            if let Some(t) = v["delta"]["text"].as_str() {
                                if ttft.is_none() {
                                    ttft = Some(t0.elapsed());
                                }
                                out.text.push_str(t);
                                on_token(t);
                            }
                        }
                        "message_delta" => {
                            if let Some(n) = v["usage"]["output_tokens"].as_u64() {
                                out.usage.output_tokens = n;
                            }
                            if let Some(r) = v["delta"]["stop_reason"].as_str() {
                                out.stop_reason = Some(r.to_string());
                            }
                        }
                        "error" => {
                            return Err(ApiError::Malformed(
                                v["error"]["message"].as_str().unwrap_or("stream error").into(),
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }

        out.ttft = ttft.unwrap_or_default();
        out.total = t0.elapsed();
        if out.text.is_empty() && out.stop_reason.is_none() {
            return Err(ApiError::Malformed("empty stream".into()));
        }
        Ok(out)
    }
}

/// Small uniform RNG so the crate does not pull in `rand` for one call site.
fn fastrand_range(cap: u64) -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    if cap == 0 {
        return 0;
    }
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    (n as u64).wrapping_mul(6364136223846793005).rotate_left(17) % cap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cache_breakpoint_serializes_where_the_api_expects_it() {
        let b = Block::text("system prompt").cached();
        let j = serde_json::to_value(&b).unwrap();
        assert_eq!(j["type"], "text");
        assert_eq!(j["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn an_uncached_block_omits_cache_control_entirely() {
        // Serializing `null` is not the same as omitting the field; the API
        // rejects an explicit null here.
        let j = serde_json::to_value(Block::text("hi")).unwrap();
        assert!(j.get("cache_control").is_none(), "got {j}");
    }

    #[test]
    fn retryable_classification_matches_what_is_worth_retrying() {
        let e = |s: u16| ApiError::Status { status: s, body: String::new() };
        assert!(e(429).retryable(), "rate limit is retryable");
        assert!(e(500).retryable());
        assert!(e(529).retryable());
        assert!(!e(400).retryable(), "a malformed request will not fix itself");
        assert!(!e(401).retryable(), "bad credentials will not fix themselves");
        assert!(ApiError::Transport("reset".into()).retryable());
        assert!(!ApiError::NoCredentials.retryable());
    }

    #[test]
    fn missing_credentials_are_an_explicit_error_not_a_silent_empty_key() {
        // Guard against sending an empty x-api-key and getting a confusing 401.
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("ANTHROPIC_AUTH_TOKEN");
        }
        assert!(matches!(Credential::from_env(), Err(ApiError::NoCredentials)));
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "   ") };
        assert!(
            matches!(Credential::from_env(), Err(ApiError::NoCredentials)),
            "whitespace is not a credential"
        );
        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
    }

    #[test]
    fn tokens_per_second_measures_decode_not_total() {
        // Including prefill in the rate understates decode speed, sometimes by
        // an order of magnitude on a long prompt.
        let c = Completion {
            usage: Usage { output_tokens: 300, ..Default::default() },
            ttft: Duration::from_millis(800),
            total: Duration::from_millis(3800),
            ..Default::default()
        };
        assert!((c.tokens_per_second() - 100.0).abs() < 0.1, "got {}", c.tokens_per_second());
    }

    #[test]
    fn tokens_per_second_is_zero_rather_than_infinite_on_a_degenerate_timing() {
        let c = Completion {
            usage: Usage { output_tokens: 10, ..Default::default() },
            ttft: Duration::from_millis(100),
            total: Duration::from_millis(100),
            ..Default::default()
        };
        assert_eq!(c.tokens_per_second(), 0.0);
        assert!(!c.tokens_per_second().is_nan());
    }

    #[test]
    fn a_request_serializes_to_the_documented_shape() {
        let r = Request {
            model: "claude-haiku-4-5".into(),
            max_tokens: 1024,
            system: vec![Block::text("You are a support assistant.").cached()],
            messages: vec![Message { role: "user", content: vec![Block::text("why 502?")] }],
            stream: true,
            temperature: None,
        };
        let j = serde_json::to_value(&r).unwrap();
        assert_eq!(j["model"], "claude-haiku-4-5");
        assert_eq!(j["stream"], true);
        assert_eq!(j["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(j["messages"][0]["role"], "user");
        assert!(j.get("temperature").is_none(), "None must be omitted, not null");
    }

    #[test]
    fn jitter_stays_within_its_cap() {
        for cap in [1u64, 100, 8000] {
            for _ in 0..50 {
                assert!(fastrand_range(cap) < cap);
            }
        }
        assert_eq!(fastrand_range(0), 0);
    }
}
