//! Server configuration derived from CLI and environment.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

use crate::cli::{Cli, FlashAttnSetting};
use crate::engine::ContextParams;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Invalid bind address '{address}': {source}")]
    InvalidAddress {
        address: String,
        #[source]
        source: std::net::AddrParseError,
    },

    #[error("Required argument '--model <MODEL>' was not provided")]
    MissingModel,

    #[error("GGUF model file not found at path: {}", .0.display())]
    ModelNotFound(PathBuf),

    #[error("Context size (--ctx-size) must be greater than zero, got {0}")]
    InvalidContextSize(u32),

    #[error("CPU threads (--threads) cannot be negative, got {0}")]
    InvalidThreads(i32),

    #[error("Physical batch size (--ubatch-size={ubatch}) cannot exceed logical batch size (--batch-size={batch})")]
    UbatchExceedsBatch { batch: u32, ubatch: u32 },

    #[error("Cannot read chat template file '{path}': {source}")]
    ChatTemplateRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Per-request resource limits that keep one client from monopolising the engine.
#[derive(Debug, Clone, Copy)]
pub struct ServiceLimits {
    pub max_body_bytes: usize,
    pub max_questions: usize,
    pub queue_depth: usize,
    pub request_timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub model_path: PathBuf,
    pub bind_addr: SocketAddr,
    pub ctx_size: u32,
    pub threads: i32,
    pub gpu_layers: i32,
    pub device: Option<String>,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub flash_attn: FlashAttnSetting,
    pub model_name: String,
    pub max_sequences: u32,
    /// Contents of `--chat-template-file`, if given.
    pub chat_template: Option<String>,
    pub limits: ServiceLimits,
    pub cors_origins: Vec<String>,
    /// Whether to expose `GET /metrics` (default: false, aligned with llama-server).
    pub metrics_enabled: bool,
    /// Accepted bearer API keys; empty disables authentication.
    pub api_keys: Vec<String>,
}

impl ServerConfig {
    pub fn from_cli(cli: Cli) -> Result<Self, ConfigError> {
        let addr_str = format!("{}:{}", cli.host, cli.port);
        let bind_addr: SocketAddr =
            addr_str
                .parse()
                .map_err(|source| ConfigError::InvalidAddress {
                    address: addr_str,
                    source,
                })?;

        let model_str = cli.model.ok_or(ConfigError::MissingModel)?;
        let model_path = PathBuf::from(&model_str);
        if !model_path.exists() {
            return Err(ConfigError::ModelNotFound(model_path));
        }

        if cli.ctx_size == 0 {
            return Err(ConfigError::InvalidContextSize(cli.ctx_size));
        }

        if cli.threads < 0 {
            return Err(ConfigError::InvalidThreads(cli.threads));
        }

        if cli.ubatch_size > cli.batch_size {
            return Err(ConfigError::UbatchExceedsBatch {
                batch: cli.batch_size,
                ubatch: cli.ubatch_size,
            });
        }

        // If a device is specified and gpu_layers is 0, default to offloading all layers (-1)
        let gpu_layers = if cli.device.is_some() && cli.gpu_layers == 0 {
            -1
        } else {
            cli.gpu_layers
        };

        let chat_template = cli
            .chat_template_file
            .map(|path| {
                std::fs::read_to_string(&path)
                    .map_err(|source| ConfigError::ChatTemplateRead { path, source })
            })
            .transpose()?;

        Ok(Self {
            model_path,
            bind_addr,
            ctx_size: cli.ctx_size,
            threads: cli.threads,
            gpu_layers,
            device: cli.device,
            batch_size: cli.batch_size,
            ubatch_size: cli.ubatch_size,
            flash_attn: cli.flash_attn,
            model_name: cli.model_name,
            max_sequences: cli.max_sequences.max(1),
            chat_template,
            limits: ServiceLimits {
                max_body_bytes: cli.max_body_bytes as usize,
                max_questions: cli.max_questions as usize,
                queue_depth: cli.queue_depth as usize,
                request_timeout: Duration::from_secs(u64::from(cli.request_timeout_secs)),
            },
            cors_origins: cli.cors_origins,
            metrics_enabled: cli.metrics,
            api_keys: cli.api_key,
        })
    }

    #[must_use]
    pub fn context_params(&self) -> ContextParams {
        ContextParams {
            n_ctx: self.ctx_size,
            n_threads: self.threads,
            n_seq_max: self.max_sequences,
            n_batch: self.batch_size,
            n_ubatch: self.ubatch_size,
            flash_attn: self.flash_attn,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_cli(model: Option<String>) -> Cli {
        Cli {
            model,
            device: None,
            list_devices: false,
            port: 8080,
            host: "127.0.0.1".to_string(),
            ctx_size: 4096,
            threads: 4,
            gpu_layers: 0,
            batch_size: 2048,
            ubatch_size: 512,
            flash_attn: FlashAttnSetting::Auto,
            model_name: "test".to_string(),
            max_sequences: 16,
            chat_template_file: None,
            max_body_bytes: 1048576,
            max_questions: 64,
            queue_depth: 32,
            request_timeout_secs: 60,
            cors_origins: vec![],
            metrics: false,
            api_key: vec![],
        }
    }

    #[test]
    fn test_missing_model() {
        let cli = dummy_cli(None);
        let err = ServerConfig::from_cli(cli).unwrap_err();
        assert!(matches!(err, ConfigError::MissingModel));
    }

    #[test]
    fn test_invalid_address() {
        let mut cli = dummy_cli(Some("nonexistent.gguf".to_string()));
        cli.host = "invalid host address".to_string();
        let err = ServerConfig::from_cli(cli).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidAddress { .. }));
    }

    #[test]
    fn test_nonexistent_model() {
        let cli = dummy_cli(Some("/nonexistent/path/model.gguf".to_string()));
        let err = ServerConfig::from_cli(cli).unwrap_err();
        assert!(matches!(err, ConfigError::ModelNotFound(_)));
    }

    #[test]
    fn test_invalid_ctx_size_and_threads() {
        // Create a temporary file to satisfy path.exists()
        let mut cli = dummy_cli(Some(file!().to_string()));

        cli.ctx_size = 0;
        let err = ServerConfig::from_cli(cli.clone()).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidContextSize(0)));

        cli.ctx_size = 4096;
        cli.threads = -1;
        let err = ServerConfig::from_cli(cli).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidThreads(-1)));
    }

    #[test]
    fn test_ubatch_exceeds_batch() {
        let mut cli = dummy_cli(Some(file!().to_string()));
        cli.batch_size = 256;
        cli.ubatch_size = 512;
        let err = ServerConfig::from_cli(cli).unwrap_err();
        assert!(matches!(
            err,
            ConfigError::UbatchExceedsBatch {
                batch: 256,
                ubatch: 512
            }
        ));
    }
}
