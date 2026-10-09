//! log-pi: System One Decision Engine in Rust
//!
//! Outer Brand: log(π)
//! Kernel Mathematical Formula: log(P_i) = z_i - ln \sum_j e^{z_j}

pub mod cli;
pub mod config;
pub mod engine;
pub mod errors;
pub mod protocol;
pub mod server;
