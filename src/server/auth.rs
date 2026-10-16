//! Optional Bearer-token authentication for the API.
//!
//! Mirrors llama-server's `--api-key`: clients present `Authorization: Bearer
//! <key>`. When no key is configured the middleware is a no-op passthrough, so
//! the authentication layer adds no overhead or coupling when disabled.
//!
//! Only the data endpoints (`/v1/systemone`, `/v1/models`) are protected;
//! `/health`, `/` and `/metrics` stay open so orchestrators and Prometheus can
//! probe and scrape without a key.

use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Axum middleware that rejects requests without a valid configured API key.
///
/// The `State` is the list of accepted keys (an empty list disables auth). Wire
/// it up with `from_fn_with_state(keys.clone(), require_api_key)`.
///
/// The `Err` variant must stay a bare `Response`: axum's `FromFn` middleware
/// requires the handler's error type to satisfy `IntoResponse`, which `Box`
/// does not. The `result_large_err` size advisory is therefore irrelevant here.
#[allow(clippy::result_large_err)]
pub async fn require_api_key(
    State(keys): State<Vec<String>>,
    req: Request,
    next: Next,
) -> Result<Response, Response> {
    if keys.is_empty() {
        return Ok(next.run(req).await);
    }

    let authorized = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .map(|token| keys.iter().any(|key| key == token))
        .unwrap_or(false);

    if authorized {
        Ok(next.run(req).await)
    } else {
        Err(unauthorized().into_response())
    }
}

fn unauthorized() -> impl IntoResponse {
    (
        StatusCode::UNAUTHORIZED,
        axum::Json(json!({
            "error": {
                "message": "Missing or invalid API key",
                "type": "unauthorized",
            }
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::get;
    use tower::ServiceExt;

    fn app(keys: Vec<String>) -> axum::Router {
        axum::Router::new()
            .route("/protected", get(|| async { "ok" }))
            .route_layer(axum::middleware::from_fn_with_state(
                keys.clone(),
                require_api_key,
            ))
            .with_state(keys)
    }

    #[tokio::test]
    async fn rejects_missing_and_wrong_key() {
        let app = app(vec!["secret".to_string()]);
        for header_value in [None, Some("Bearer wrong")] {
            let mut req = Request::builder().uri("/protected");
            if let Some(h) = header_value {
                req = req.header(AUTHORIZATION, h);
            }
            let resp = app
                .clone()
                .oneshot(req.body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        }
    }

    #[tokio::test]
    async fn accepts_valid_bearer_key() {
        let app = app(vec!["secret".to_string()]);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .header(AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn passes_through_when_auth_disabled() {
        let app = app(vec![]);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn accepts_any_of_multiple_keys() {
        let app = app(vec!["a".to_string(), "b".to_string()]);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .header(AUTHORIZATION, "Bearer b")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
