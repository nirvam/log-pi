//! Opt-in CORS policy.

use axum::http::header::{InvalidHeaderValue, AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderValue, Method};
use tower_http::cors::{AllowOrigin, CorsLayer};

/// Builds the CORS layer for the configured origins.
///
/// An empty list disables CORS entirely (no layer); `"*"` allows any origin.
pub fn cors_layer(origins: &[String]) -> Result<Option<CorsLayer>, InvalidHeaderValue> {
    if origins.is_empty() {
        return Ok(None);
    }

    let allow_origin = if origins.iter().any(|origin| origin == "*") {
        AllowOrigin::any()
    } else {
        AllowOrigin::list(
            origins
                .iter()
                .map(|origin| HeaderValue::from_str(origin))
                .collect::<Result<Vec<_>, _>>()?,
        )
    };

    Ok(Some(
        CorsLayer::new()
            .allow_origin(allow_origin)
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([CONTENT_TYPE, AUTHORIZATION]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn origins(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn empty_origins_disable_cors() {
        assert!(cors_layer(&[]).expect("valid").is_none());
    }

    #[test]
    fn wildcard_and_explicit_origins_enable_cors() {
        assert!(cors_layer(&origins(&["*"])).expect("valid").is_some());
        assert!(
            cors_layer(&origins(&["https://a.example", "https://b.example"]))
                .expect("valid")
                .is_some()
        );
    }

    #[test]
    fn invalid_origin_is_rejected() {
        assert!(cors_layer(&origins(&["bad\norigin"])).is_err());
    }
}
