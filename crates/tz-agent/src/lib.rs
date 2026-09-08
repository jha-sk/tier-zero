//! The agent layer: routing, generation, and cost enforcement.

pub mod anthropic;
pub mod answer;
pub mod budget;
pub mod router;

pub use anthropic::{ApiError, Client, Completion};
pub use answer::{Answer, AnswerPipeline};
pub use budget::{BudgetError, BudgetGuard};
pub use router::{Route, TierModels, route};
