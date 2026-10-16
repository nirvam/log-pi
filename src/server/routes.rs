//! HTTP API route handlers.

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tracing::{info, instrument};
use uuid::Uuid;

use crate::errors::AppError;
use crate::protocol::{SystemOneRequest, SystemOneResponse, Usage};
use crate::server::state::AppState;

/// Creates the router configuring all API endpoints.
///
/// - `POST /v1/systemone` and `GET /v1/models` are protected by `--api-key` auth
///   (when keys are configured).
/// - `/health`, `/` and (when enabled) `GET /metrics` are always open, so
///   orchestrators and Prometheus can probe without a key.
/// - `GET /metrics` is registered only when metrics are enabled (`--metrics`),
///   so disabling them also hides the endpoint.
pub fn create_router(state: AppState) -> Router {
    let max_body_bytes = state.limits.max_body_bytes;
    let metrics_enabled = state.metrics.is_some();

    // Data endpoints: require a valid API key when authentication is configured.
    let api_keys = state.api_keys.clone();
    let mut router = Router::new()
        .route("/v1/systemone", post(handle_systemone))
        .route("/v1/models", get(handle_models))
        .route_layer(axum::middleware::from_fn_with_state(
            api_keys,
            crate::server::auth::require_api_key,
        ))
        .layer(DefaultBodyLimit::max(max_body_bytes));

    // Always-open endpoints: health probes and metrics must not need a key.
    let open = Router::new()
        .route("/health", get(handle_health))
        .route("/", get(handle_health));
    router = router.merge(open);
    if metrics_enabled {
        router = router.route("/metrics", get(handle_metrics));
    }

    router.with_state(state)
}

/// Core Jev Wire Protocol endpoint: `POST /v1/systemone`.
#[instrument(skip_all, fields(questions_count = req.questions.len()))]
async fn handle_systemone(
    State(state): State<AppState>,
    Json(req): Json<SystemOneRequest>,
) -> Result<Json<SystemOneResponse>, AppError> {
    req.validate(state.limits.max_questions)?;

    // Track the request as in-flight (queued + being inferred). The pure
    // inference duration is recorded on the engine thread, so recording the
    // round-trip here would double-count; we only measure in-flight and outcome.
    let metrics = state.metrics.clone();
    if let Some(m) = &metrics {
        m.request_started();
    }

    // Inference runs on the dedicated engine thread; this task only awaits the result.
    let result = state
        .engine
        .solve_batch(req.state.to_prompt_string(), req.questions)
        .await;

    if let Some(m) = &metrics {
        match &result {
            Ok(_) => m.record_request(crate::metrics::Outcome::Ok),
            Err(err) => m.record_request(err.outcome()),
        }
        m.request_finished();
    }

    let (answers, total_tokens) = result?;

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

/// Prometheus metrics endpoint: `GET /metrics`.
///
/// Only reachable when `--metrics` was passed (the route is registered
/// conditionally in [`create_router`]); the handler is never a no-op because a
/// disabled endpoint simply 404s.
async fn handle_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let body = state
        .metrics
        .as_ref()
        .map(|m| m.render())
        .unwrap_or_default();
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}
