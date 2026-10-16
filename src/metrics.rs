//! Prometheus-compatible runtime metrics.
//!
//! This module is the **single owner** of all exported metrics. Business code
//! (engine worker, HTTP handlers) calls the semantic helpers below and never
//! touches `prometheus_client` types directly, so adding a new metric is a
//! one-file change: add a field, register it in [`Metrics::new`], and expose a
//! recording method.
//!
//! Rendering is decoupled from recording: [`Metrics::render`] serialises the
//! whole registry in the Prometheus OpenMetrics text format (the format
//! Prometheus scrapes). The `GET /metrics` handler is a thin shell over it.
//!
//! It lives at the crate root (not under `server`) so both the engine worker
//! and the HTTP layer can reference it without an engine→server layering
//! inversion.

use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{exponential_buckets, Histogram};
use prometheus_client::registry::Registry;
use std::fmt::Write as _;
use std::sync::Arc;

/// Outcome of a `POST /v1/systemone` request, used as the label of the
/// request-rate counter. Add variants here to track new failure modes.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Outcome {
    /// Inference ran to completion and a response was returned.
    Ok,
    /// The engine queue was full at admission; the request was rejected (503).
    Overloaded,
    /// The request exceeded its deadline while waiting or running (504).
    Timeout,
    /// The engine thread died; the request could not be served (503).
    Unavailable,
}

impl EncodeLabelValue for Outcome {
    fn encode(&self, encoder: &mut LabelValueEncoder<'_>) -> Result<(), std::fmt::Error> {
        let value = match self {
            Outcome::Ok => "ok",
            Outcome::Overloaded => "overloaded",
            Outcome::Timeout => "timeout",
            Outcome::Unavailable => "unavailable",
        };
        encoder.write_str(value)
    }
}

/// Label set for [`Metrics::requests_total`].
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RequestLabels {
    outcome: Outcome,
}

/// Thread-safe handle to the process-wide metric registry.
///
/// Cheaply cloneable so it can be embedded in [`crate::server::state::AppState`]
/// and handed to both the engine worker and the HTTP handlers. `None` on the
/// app state means metrics are disabled (`--metrics` not passed); recording
/// calls then no-op.
#[derive(Clone)]
pub struct Metrics {
    /// Registry is immutable after construction; `Arc` makes [`Metrics`] cheap to
    /// clone (required by axum's `State` / the app state) even though
    /// `prometheus_client::Registry` itself is not `Clone`.
    registry: Arc<Registry>,
    /// Total requests admitted to `POST /v1/systemone`, labelled by outcome.
    requests_total: Family<RequestLabels, Counter>,
    /// Requests currently being inferred (queued + running on the engine).
    requests_in_flight: Gauge,
    /// Number of jobs waiting in the engine queue right now (0 when idle).
    queue_depth: Gauge,
    /// Queue-wait time for a job to be picked up by the engine thread.
    queue_wait_seconds: Histogram,
    /// Pure inference time spent decoding all questions of a batch.
    infer_seconds: Histogram,
}

impl Metrics {
    /// Creates the registry and registers every metric once.
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let requests_total = Family::<RequestLabels, Counter>::default();
        let requests_in_flight = Gauge::default();
        let queue_depth = Gauge::default();
        let queue_wait_seconds = Histogram::new(exponential_buckets(0.001, 2.0, 16));
        let infer_seconds = Histogram::new(exponential_buckets(0.001, 2.0, 16));

        registry.register(
            "logpi_requests",
            "Total SystemOne requests, labelled by outcome",
            requests_total.clone(),
        );
        registry.register(
            "logpi_requests_in_flight",
            "Requests currently being inferred (queued or running)",
            requests_in_flight.clone(),
        );
        registry.register(
            "logpi_engine_queue_depth",
            "Number of jobs waiting in the engine queue",
            queue_depth.clone(),
        );
        registry.register(
            "logpi_queue_wait_seconds",
            "Time a job waited for the engine to pick it up",
            queue_wait_seconds.clone(),
        );
        registry.register(
            "logpi_infer_seconds",
            "Time spent inferring a batch of questions",
            infer_seconds.clone(),
        );

        Self {
            registry: Arc::new(registry),
            requests_total,
            requests_in_flight,
            queue_depth,
            queue_wait_seconds,
            infer_seconds,
        }
    }

    /// Bumps the outcome counter, e.g. once per completed `POST /v1/systemone`.
    pub fn record_request(&self, outcome: Outcome) {
        self.requests_total
            .get_or_create(&RequestLabels { outcome })
            .inc();
    }

    /// Tracks a request entering/leaving the inference pipeline.
    pub fn request_started(&self) {
        self.requests_in_flight.inc();
    }

    pub fn request_finished(&self) {
        self.requests_in_flight.dec();
    }

    /// Snapshot of the engine queue depth, written from the engine thread.
    pub fn set_queue_depth(&self, depth: usize) {
        self.queue_depth.set(depth as i64);
    }

    /// Records how long a job sat in the queue before the engine picked it up.
    pub fn observe_queue_wait(&self, seconds: f64) {
        self.queue_wait_seconds.observe(seconds);
    }

    /// Records the pure inference time of a batch.
    pub fn observe_infer(&self, seconds: f64) {
        self.infer_seconds.observe(seconds);
    }

    /// Serialises all metrics in the Prometheus OpenMetrics text format.
    pub fn render(&self) -> String {
        let mut buf = String::new();
        let _ = encode(&mut buf, &self.registry);
        buf
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_emits_registered_metrics_with_values() {
        let m = Metrics::new();
        // Record at least one value on every metric so the encoder does not
        // elide empty metrics from the exposition.
        m.record_request(Outcome::Ok);
        m.request_started();
        m.request_finished();
        m.set_queue_depth(3);
        m.observe_queue_wait(0.25);
        m.observe_infer(0.5);

        let text = m.render();
        for name in [
            "logpi_requests",
            "logpi_requests_in_flight",
            "logpi_engine_queue_depth",
            "logpi_queue_wait_seconds",
            "logpi_infer_seconds",
        ] {
            assert!(
                text.contains(name),
                "expected metric {name} in render, got:\n{text}"
            );
        }
        // Ends with the OpenMetrics EOF marker.
        assert!(text.contains("# EOF"));
    }

    #[test]
    fn record_request_attributes_to_the_right_outcome_label() {
        let m = Metrics::new();
        m.record_request(Outcome::Ok);
        m.record_request(Outcome::Overloaded);
        let text = m.render();
        // Counter base name "logpi_requests" is exposed with the "_total" suffix.
        assert!(
            text.contains("logpi_requests_total{outcome=\"ok\"} 1"),
            "got:\n{text}"
        );
        assert!(
            text.contains("logpi_requests_total{outcome=\"overloaded\"} 1"),
            "got:\n{text}"
        );
    }

    #[test]
    fn in_flight_tracks_started_minus_finished() {
        let m = Metrics::new();
        m.request_started();
        m.request_started();
        m.request_finished();
        let text = m.render();
        assert!(text.contains("logpi_requests_in_flight 1"), "got:\n{text}");
    }

    #[test]
    fn histogram_and_gauge_reflect_observations() {
        let m = Metrics::new();
        m.observe_queue_wait(0.5);
        m.observe_infer(1.25);
        m.set_queue_depth(7);

        // test-util feature gates `count`/`sum` accessors.
        assert_eq!(m.queue_wait_seconds.count(), 1);
        assert_eq!(m.infer_seconds.count(), 1);
        let text = m.render();
        // Gauges render as `name value` with no braces.
        assert!(text.contains("logpi_engine_queue_depth 7"), "got:\n{text}");
    }
}
