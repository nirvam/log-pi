//! Dedicated OS thread that owns the llama.cpp engine.
//!
//! Inference is a long, blocking FFI call. Running it on a Tokio worker thread starves every
//! other task (health probes included), so the engine lives on its own thread and is reached
//! through a bounded queue:
//!
//! - **Backpressure:** the queue is bounded; a full queue is rejected immediately (HTTP 503)
//!   instead of piling up unbounded latency.
//! - **Deadlines:** callers wait at most `timeout` (queue wait + inference). FFI calls cannot be
//!   cancelled, so a running job finishes; a job still queued when its caller gave up is skipped.
//! - **Ownership:** the engine is never shared, so no lock is needed.

use std::thread::{self, JoinHandle};
use std::time::Duration;

use indexmap::IndexMap;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info};

use crate::engine::LlamaEngine;
use crate::errors::AppError;
use crate::protocol::{Answer, Question};

/// Answers keyed by question id, plus the number of tokens evaluated.
pub type BatchOutput = (IndexMap<String, Answer>, usize);

/// Anything able to answer a batch of questions about one shared state.
pub trait BatchSolver {
    fn solve_batch(
        &mut self,
        state: &str,
        questions: &IndexMap<String, Question>,
    ) -> Result<BatchOutput, AppError>;
}

impl BatchSolver for LlamaEngine {
    fn solve_batch(
        &mut self,
        state: &str,
        questions: &IndexMap<String, Question>,
    ) -> Result<BatchOutput, AppError> {
        LlamaEngine::solve_batch(self, state, questions)
    }
}

struct Job {
    state: String,
    questions: IndexMap<String, Question>,
    reply: oneshot::Sender<Result<BatchOutput, AppError>>,
}

/// Cheap, cloneable client side of the engine thread.
#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<Job>,
    timeout: Duration,
}

/// Owner of the engine thread; joined on shutdown so the GPU context is freed cleanly.
pub struct EngineWorker {
    thread: JoinHandle<()>,
}

/// Starts the engine thread.
///
/// `queue_depth` bounds the jobs waiting behind the one currently running.
pub fn spawn_engine_worker<S>(
    solver: S,
    queue_depth: usize,
    timeout: Duration,
) -> std::io::Result<(EngineHandle, EngineWorker)>
where
    S: BatchSolver + Send + 'static,
{
    let (tx, rx) = mpsc::channel(queue_depth.max(1));
    let thread = thread::Builder::new()
        .name("log-pi-engine".to_string())
        .spawn(move || run(solver, rx))?;

    Ok((EngineHandle { tx, timeout }, EngineWorker { thread }))
}

fn run<S: BatchSolver>(mut solver: S, mut rx: mpsc::Receiver<Job>) {
    while let Some(job) = rx.blocking_recv() {
        if job.reply.is_closed() {
            debug!("Skipping job abandoned by its caller before it started");
            continue;
        }

        let result = solver.solve_batch(&job.state, &job.questions);
        if job.reply.send(result).is_err() {
            debug!("Caller gave up before the result was delivered");
        }
    }
    info!("Engine worker stopped");
}

impl EngineHandle {
    /// Queues a batch and waits for its result, bounded by the configured timeout.
    pub async fn solve_batch(
        &self,
        state: String,
        questions: IndexMap<String, Question>,
    ) -> Result<BatchOutput, AppError> {
        let (reply, response) = oneshot::channel();
        self.tx
            .try_send(Job {
                state,
                questions,
                reply,
            })
            .map_err(|err| match err {
                TrySendError::Full(_) => AppError::Overloaded,
                TrySendError::Closed(_) => AppError::EngineUnavailable,
            })?;

        match tokio::time::timeout(self.timeout, response).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(AppError::EngineUnavailable),
            Err(_) => Err(AppError::Timeout(self.timeout.as_secs())),
        }
    }

    /// Whether the engine thread is still accepting work.
    pub fn is_available(&self) -> bool {
        !self.tx.is_closed()
    }
}

impl EngineWorker {
    /// Blocks until the thread exits, which happens once every [`EngineHandle`] is dropped.
    pub fn join(self) {
        if self.thread.join().is_err() {
            error!("Engine worker thread panicked");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc as std_mpsc, Arc};

    use super::*;

    /// Solver that reports when a call starts and blocks until the test releases it.
    struct GatedSolver {
        started: std_mpsc::Sender<()>,
        release: std_mpsc::Receiver<()>,
        calls: Arc<AtomicUsize>,
    }

    impl BatchSolver for GatedSolver {
        fn solve_batch(
            &mut self,
            _state: &str,
            _questions: &IndexMap<String, Question>,
        ) -> Result<BatchOutput, AppError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.send(()).expect("test is listening");
            self.release.recv().expect("test releases the solver");
            Ok((IndexMap::new(), 7))
        }
    }

    struct PanickingSolver;

    impl BatchSolver for PanickingSolver {
        fn solve_batch(
            &mut self,
            _state: &str,
            _questions: &IndexMap<String, Question>,
        ) -> Result<BatchOutput, AppError> {
            panic!("simulated engine crash");
        }
    }

    struct Gate {
        started: std_mpsc::Receiver<()>,
        release: std_mpsc::Sender<()>,
        calls: Arc<AtomicUsize>,
    }

    fn gated_solver() -> (GatedSolver, Gate) {
        let (started_tx, started_rx) = std_mpsc::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let solver = GatedSolver {
            started: started_tx,
            release: release_rx,
            calls: calls.clone(),
        };
        let gate = Gate {
            started: started_rx,
            release: release_tx,
            calls,
        };
        (solver, gate)
    }

    async fn submit(handle: &EngineHandle) -> Result<BatchOutput, AppError> {
        handle
            .solve_batch("state".to_string(), IndexMap::new())
            .await
    }

    #[tokio::test]
    async fn returns_the_solver_result() {
        let (solver, gate) = gated_solver();
        let (handle, worker) =
            spawn_engine_worker(solver, 1, Duration::from_secs(5)).expect("spawn");
        gate.release.send(()).expect("pre-release");

        let (answers, tokens) = submit(&handle).await.expect("solved");
        assert!(answers.is_empty());
        assert_eq!(tokens, 7);

        drop(handle);
        worker.join();
    }

    #[tokio::test]
    async fn rejects_when_queue_is_full() {
        let (solver, gate) = gated_solver();
        let (handle, worker) =
            spawn_engine_worker(solver, 1, Duration::from_secs(5)).expect("spawn");

        // Job 1 occupies the engine; wait until it is actually running.
        let running = tokio::spawn({
            let handle = handle.clone();
            async move { submit(&handle).await }
        });
        let started = gate.started;
        let started = tokio::task::spawn_blocking(move || {
            started.recv().expect("job 1 started");
            started
        })
        .await
        .expect("join");

        // Job 2 fills the single queue slot; one yield lets it run up to its first await.
        let queued = tokio::spawn({
            let handle = handle.clone();
            async move { submit(&handle).await }
        });
        tokio::task::yield_now().await;

        assert!(matches!(submit(&handle).await, Err(AppError::Overloaded)));

        gate.release.send(()).expect("release job 1");
        gate.release.send(()).expect("release job 2");
        assert!(running.await.expect("join").is_ok());
        assert!(queued.await.expect("join").is_ok());
        drop(started);

        drop(handle);
        worker.join();
    }

    #[tokio::test]
    async fn times_out_and_skips_abandoned_queued_jobs() {
        let (solver, gate) = gated_solver();
        let (handle, worker) =
            spawn_engine_worker(solver, 2, Duration::from_millis(100)).expect("spawn");

        // Job 1 runs (and blocks); job 2 waits in the queue. Both callers time out.
        let (first, second) = tokio::join!(submit(&handle), submit(&handle));
        assert!(matches!(first, Err(AppError::Timeout(_))));
        assert!(matches!(second, Err(AppError::Timeout(_))));

        // Unblock job 1; job 2 must be skipped because its caller is gone.
        gate.release.send(()).expect("release job 1");
        drop(handle);
        worker.join();

        assert_eq!(gate.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn reports_unavailable_after_worker_crash() {
        let (handle, worker) =
            spawn_engine_worker(PanickingSolver, 1, Duration::from_secs(5)).expect("spawn");
        assert!(handle.is_available());

        assert!(matches!(
            submit(&handle).await,
            Err(AppError::EngineUnavailable)
        ));

        // The receiver is dropped while the thread unwinds; give it a moment.
        for _ in 0..100 {
            if !handle.is_available() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!handle.is_available());
        assert!(matches!(
            submit(&handle).await,
            Err(AppError::EngineUnavailable)
        ));

        drop(handle);
        worker.join();
    }
}
