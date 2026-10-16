//! Global shared server state.

use crate::config::ServiceLimits;
use crate::engine::{EngineHandle, ModelInfo};
use crate::metrics::Metrics;

#[derive(Clone)]
pub struct AppState {
    pub engine: EngineHandle,
    pub model_info: ModelInfo,
    pub model_name: String,
    pub limits: ServiceLimits,
    /// Prometheus metrics registry; `None` when `--metrics` is disabled.
    pub metrics: Option<Metrics>,
    /// Accepted `Authorization: Bearer` API keys; empty disables authentication.
    pub api_keys: Vec<String>,
}

impl AppState {
    pub fn new(
        engine: EngineHandle,
        model_info: ModelInfo,
        model_name: String,
        limits: ServiceLimits,
        metrics: Option<Metrics>,
        api_keys: Vec<String>,
    ) -> Self {
        Self {
            engine,
            model_info,
            model_name,
            limits,
            metrics,
            api_keys,
        }
    }
}
