//! HTTP API route handlers.

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tracing::{info, instrument};
use uuid::Uuid;

use crate::errors::AppError;
use crate::protocol::{SystemOneRequest, SystemOneResponse, Usage};
use crate::server::state::AppState;

/// Creates the router configuring all API endpoints.
pub fn create_router(state: AppState) -> Router {
    let max_body_bytes = state.limits.max_body_bytes;
    Router::new()
        .route("/v1/systemone", post(handle_systemone))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        .route("/", get(handle_health))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

/// Core Jev Wire Protocol endpoint: `POST /v1/systemone`.
#[instrument(skip_all, fields(questions_count = req.questions.len()))]
async fn handle_systemone(
    State(state): State<AppState>,
    Json(req): Json<SystemOneRequest>,
) -> Result<Json<SystemOneResponse>, AppError> {
    req.validate(state.limits.max_questions)?;

    // Inference runs on the dedicated engine thread; this task only awaits the result.
    let (answers, total_tokens) = state
        .engine
        .solve_batch(req.state.to_prompt_string(), req.questions)
        .await?;

    let id = format!("sys1_{}", Uuid::new_v4().simple());
    let response = SystemOneResponse {
        id,
        model: state.model_name.clone(),
        answers,
        usage: Usage::new(total_tokens),
    };

    info!(
        model = %state.model_name,
        tokens = total_tokens,
        "SystemOne request processed successfully"
    );

    Ok(Json(response))
}

/// Metadata endpoint: `GET /v1/models`.
async fn handle_models(State(state): State<AppState>) -> Json<Value> {
    let info = state.model_info;

    Json(json!({
        "object": "list",
        "data": [
            {
                "id": state.model_name,
                "object": "model",
                "created": info.created,
                "owned_by": "log-pi",
                "context_window": info.n_ctx,
                "vocab_size": info.vocab_size,
                "n_params": info.n_params,
                "size_bytes": info.size_bytes
            }
        ]
    }))
}

/// Health and banner probe: `GET /health` or `GET /`.
///
/// Reports `503` once the engine thread has died, so orchestrators can restart the service.
async fn handle_health(State(state): State<AppState>) -> (StatusCode, Json<Value>) {
    let (code, status) = if state.engine.is_available() {
        (StatusCode::OK, "ok")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "unavailable")
    };

    (
        code,
        Json(json!({
            "status": status,
            "service": "log-pi",
            "logo": "log(π)",
            "version": env!("CARGO_PKG_VERSION")
        })),
    )
}
