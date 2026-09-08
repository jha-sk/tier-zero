//! Tracing and per-request cost attribution.
//!
//! Attribute names follow the OpenTelemetry GenAI semantic conventions. Those
//! conventions are **not fully stable**: client spans settled in early 2026 but
//! the agent and MCP conventions are still in Development. The names are
//! therefore pinned in one place here rather than scattered as string literals,
//! so that tracking a spec revision is one edit and a changelog entry.
//!
//! The unit of cost is the **request**, not the model call. An agent turn that
//! routes, generates, and samples a judge is one thing the user asked for and
//! one budget.

pub mod attr;
pub mod export;
pub mod record;

pub use attr::*;
pub use export::{Metrics, init_meter_provider};
pub use record::{CacheOutcome, RequestRecord, StageTiming};
