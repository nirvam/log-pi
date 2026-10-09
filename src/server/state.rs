//! Global shared server state.

use crate::config::ServiceLimits;
use crate::engine::{EngineHandle, ModelInfo};

#[derive(Clone)]
pub struct AppState {
    pub engine: EngineHandle,
    pub model_info: ModelInfo,
    pub model_name: String,
    pub limits: ServiceLimits,
}

impl AppState {
    pub fn new(
        engine: EngineHandle,
        model_info: ModelInfo,
        model_name: String,
        limits: ServiceLimits,
    ) -> Self {
        Self {
            engine,
            model_info,
            model_name,
            limits,
        }
    }
}
