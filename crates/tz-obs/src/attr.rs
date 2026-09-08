//! OpenTelemetry GenAI attribute names, pinned in one place.
//!
//! Spec revision tracked: `1.30.0` (2026). Client span attributes are stable;
//! the `gen_ai.agent.*` names are still Development and may change. Anything
//! this project invents is namespaced `tierzero.*` so it is never mistaken for
//! a standard attribute.

/// Conventions revision these names were taken from.
pub const OTEL_GENAI_SCHEMA: &str = "1.30.0";

// --- stable client-span attributes ---
pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
pub const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";
pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
pub const GEN_AI_RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";
pub const GEN_AI_SYSTEM: &str = "gen_ai.system";

// --- project attributes ---
// The conventions have no name for cached-token classes or for cost, and both
// are load-bearing here, so they are explicitly ours.
pub const TZ_CACHE_READ_TOKENS: &str = "tierzero.usage.cache_read_tokens";
pub const TZ_CACHE_WRITE_TOKENS: &str = "tierzero.usage.cache_write_tokens";
pub const TZ_COST_USD: &str = "tierzero.cost.usd";
/// Which price table produced `tierzero.cost.usd`. Without this a historical
/// cost cannot be recomputed after prices move.
pub const TZ_PRICE_TABLE_VERSION: &str = "tierzero.cost.price_table_version";
pub const TZ_ROUTE_TIER: &str = "tierzero.route.tier";
pub const TZ_TENANT_ID: &str = "tierzero.tenant.id";
pub const TZ_CACHE_OUTCOME: &str = "tierzero.cache.outcome";
pub const TZ_CACHE_SIMILARITY: &str = "tierzero.cache.similarity";
pub const TZ_DEGRADED: &str = "tierzero.degraded";
pub const TZ_RETRIEVAL_CANDIDATES: &str = "tierzero.retrieval.candidates";

/// Span names for the retrieval path, so dashboards can slice by stage.
pub mod span {
    pub const RETRIEVE: &str = "tierzero.retrieve";
    pub const EMBED_QUERY: &str = "tierzero.embed_query";
    pub const VECTOR_SEARCH: &str = "tierzero.vector_search";
    pub const FUSE: &str = "tierzero.fuse";
    pub const GENERATE: &str = "gen_ai.generate_content";
    pub const ROUTE: &str = "tierzero.route";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_attributes_use_the_gen_ai_namespace() {
        for a in [
            GEN_AI_OPERATION_NAME,
            GEN_AI_REQUEST_MODEL,
            GEN_AI_USAGE_INPUT_TOKENS,
            GEN_AI_USAGE_OUTPUT_TOKENS,
        ] {
            assert!(a.starts_with("gen_ai."), "{a} is not a conventions name");
        }
    }

    #[test]
    fn invented_attributes_are_namespaced_so_they_cannot_be_mistaken_for_standard() {
        for a in [TZ_COST_USD, TZ_CACHE_READ_TOKENS, TZ_ROUTE_TIER, TZ_PRICE_TABLE_VERSION] {
            assert!(a.starts_with("tierzero."), "{a} would be mistaken for a spec attribute");
        }
    }

    #[test]
    fn the_conventions_revision_is_recorded() {
        // The agent conventions are still in Development; pinning the revision
        // is what makes a later break traceable rather than mysterious.
        assert!(!OTEL_GENAI_SCHEMA.is_empty());
    }
}
