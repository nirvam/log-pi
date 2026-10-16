//! Domain error types and Axum response mapping.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Failed to load model from path '{path}': {message}")]
    ModelLoadError { path: String, message: String },

    #[error("Failed to create inference context: {0}")]
    ContextCreationError(String),

    #[error("Tokenization failed: {0}")]
    TokenizeError(String),

    #[error("Model decode failed with error code: {0}")]
    DecodeError(i32),

    #[error("Failed to extract logits from context: {0}")]
    LogitsError(String),

    #[error("Option labels must map to single, distinct vocabulary tokens: {0}")]
    LabelTokenError(String),

    #[error("Chat template error: {0}")]
    ChatTemplateError(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Server is overloaded: the inference queue is full")]
    Overloaded,

    #[error("Inference engine is unavailable")]
    EngineUnavailable,

    #[error("Request timed out after {0}s")]
    Timeout(u64),
}

impl AppError {
    pub fn tokenize_error(text: &str) -> Self {
        const MAX_SNIPPET: usize = 48;
        let char_count = text.chars().count();
        let snippet: String = text.chars().take(MAX_SNIPPET).collect();
        let ellipsis = if char_count > MAX_SNIPPET { "..." } else { "" };
        AppError::TokenizeError(format!(
            "{char_count} chars input starting with '{snippet}{ellipsis}'"
        ))
    }

    /// Maps the error to a request-outcome label for metrics.
    ///
    /// Only the engine-admission outcomes (overload/timeout/unavailable) carry a
    /// dedicated bucket; every other failure is an internal/request error and is
    /// reported together so the outcome label set stays small.
    pub fn outcome(&self) -> crate::metrics::Outcome {
        match self {
            AppError::Overloaded => crate::metrics::Outcome::Overloaded,
            AppError::Timeout(_) => crate::metrics::Outcome::Timeout,
            AppError::EngineUnavailable => crate::metrics::Outcome::Unavailable,
            _ => crate::metrics::Outcome::Ok,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, err_type) = match &self {
            AppError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            AppError::TokenizeError(_) => (StatusCode::BAD_REQUEST, "tokenize_error"),
            AppError::DecodeError(1) => (StatusCode::BAD_REQUEST, "context_length_exceeded"),
            AppError::DecodeError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "decode_error"),
            AppError::ChatTemplateError(_) => (StatusCode::BAD_REQUEST, "template_error"),
            AppError::ModelLoadError { .. } => (StatusCode::INTERNAL_SERVER_ERROR, "model_error"),
            AppError::ContextCreationError(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "context_error")
            }
            AppError::LabelTokenError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "model_error"),
            AppError::LogitsError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "logits_error"),
            AppError::Overloaded => (StatusCode::SERVICE_UNAVAILABLE, "overloaded_error"),
            AppError::EngineUnavailable => (StatusCode::SERVICE_UNAVAILABLE, "api_error"),
            AppError::Timeout(_) => (StatusCode::GATEWAY_TIMEOUT, "timeout_error"),
        };

        let body = axum::Json(json!({
            "error": {
                "message": self.to_string(),
                "type": err_type,
            }
        }));

        (status, body).into_response()
    }
}
