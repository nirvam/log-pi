//! Server module definitions.

pub mod auth;
pub mod cors;
pub mod routes;
pub mod state;

pub use crate::metrics::{Metrics, Outcome};
pub use auth::require_api_key;
pub use cors::cors_layer;
pub use routes::create_router;
pub use state::AppState;
