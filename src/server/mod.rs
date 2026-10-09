//! Server module definitions.

pub mod cors;
pub mod routes;
pub mod state;

pub use cors::cors_layer;
pub use routes::create_router;
pub use state::AppState;
