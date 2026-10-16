//! Command line argument parsing.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "log-pi",
    author = "nirvam",
    version,
    about = "System One Decision Engine in Rust - 100% TypeSafe Jev Wire Protocol Compatible",
    after_help = "Brand Identity: log(π) | Mathematical Kernel: log(P_i) = z_i - ln Σ e^{z_j}"
)]
pub struct Cli {
    /// Path to GGUF model file
    #[arg(short, long, env = "LOGPI_MODEL")]
    pub model: Option<String>,

    /// Comma-separated list of devices to use for offloading (e.g. Vulkan0, CUDA0, none)
    #[arg(
        short = 'd',
        long = "device",
        visible_alias = "dev",
        env = "LOGPI_DEVICE"
    )]
    pub device: Option<String>,

    /// Print list of available GGML devices and exit
    #[arg(long = "list-devices")]
    pub list_devices: bool,

    /// Serving HTTP port
    #[arg(short, long, default_value_t = 8080, env = "LOGPI_PORT")]
    pub port: u16,

    /// HTTP binding address
    #[arg(long, default_value = "0.0.0.0", env = "LOGPI_HOST")]
    pub host: String,

    /// Context window size in tokens
    #[arg(long, default_value_t = 4096, env = "LOGPI_CTX_SIZE")]
    pub ctx_size: u32,

    /// Number of CPU threads for inference (0 = auto)
    #[arg(short, long, default_value_t = 4, env = "LOGPI_THREADS")]
    pub threads: i32,

    /// Number of layers to offload to GPU (-1 = all, 0 = none)
    #[arg(
        short = 'g',
        long = "gpu-layers",
        visible_alias = "ngl",
        default_value_t = 0,
        allow_hyphen_values = true,
        env = "LOGPI_GPU_LAYERS"
    )]
    pub gpu_layers: i32,

    /// Logical maximum batch size (default: 2048)
    #[arg(
        short = 'b',
        long = "batch-size",
        default_value_t = 2048,
        env = "LOGPI_BATCH_SIZE",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub batch_size: u32,

    /// Physical maximum batch size (default: 512)
    #[arg(
        long = "ubatch-size",
        visible_alias = "ub",
        default_value_t = 512,
        env = "LOGPI_UBATCH_SIZE",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub ubatch_size: u32,

    /// Set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
    #[arg(
        long = "flash-attn",
        visible_alias = "fa",
        default_value = "auto",
        env = "LOGPI_FLASH_ATTN"
    )]
    pub flash_attn: FlashAttnSetting,

    /// Model identifier to report in API responses
    #[arg(long, default_value = "log-pi", env = "LOGPI_MODEL_NAME")]
    pub model_name: String,

    /// Maximum concurrent sequences for multi-branch evaluation (default 16)
    #[arg(
        long = "max-sequences",
        visible_alias = "nseq",
        default_value_t = 16,
        env = "LOGPI_MAX_SEQUENCES"
    )]
    pub max_sequences: u32,

    /// Jinja chat template file overriding the one embedded in the GGUF
    #[arg(long, env = "LOGPI_CHAT_TEMPLATE_FILE")]
    pub chat_template_file: Option<PathBuf>,

    /// Maximum request body size in bytes
    #[arg(
        long,
        default_value_t = 1_048_576,
        env = "LOGPI_MAX_BODY_BYTES",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub max_body_bytes: u32,

    /// Maximum number of questions per request
    #[arg(
        long,
        default_value_t = 64,
        env = "LOGPI_MAX_QUESTIONS",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub max_questions: u32,

    /// Requests allowed to wait behind the one being evaluated; beyond this, HTTP 503
    #[arg(
        long,
        default_value_t = 32,
        env = "LOGPI_QUEUE_DEPTH",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub queue_depth: u32,

    /// Per-request deadline in seconds (queue wait + inference)
    #[arg(
        long,
        default_value_t = 300,
        env = "LOGPI_REQUEST_TIMEOUT_SECS",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub request_timeout_secs: u32,

    /// Comma-separated CORS origins allowed to call the API ('*' = any); empty disables CORS
    #[arg(long, value_delimiter = ',', env = "LOGPI_CORS_ORIGINS")]
    pub cors_origins: Vec<String>,

    /// Expose a Prometheus-compatible metrics endpoint at GET /metrics
    #[arg(long, env = "LOGPI_METRICS")]
    pub metrics: bool,

    /// API key(s) for `Authorization: Bearer` auth; comma-separated for multiple.
    /// Empty disables authentication. Data endpoints only; /health and /metrics stay open.
    #[arg(long, value_delimiter = ',', env = "LOGPI_API_KEY")]
    pub api_key: Vec<String>,
}

/// Flash Attention mode matching llama-server (`-fa, --flash-attn [on|off|auto]`).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    clap::ValueEnum,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum FlashAttnSetting {
    #[default]
    #[value(name = "auto")]
    Auto,
    #[value(name = "on")]
    On,
    #[value(name = "off")]
    Off,
}

impl FlashAttnSetting {
    /// Maps to `enum llama_flash_attn_type` in `llama.h`.
    #[must_use]
    pub const fn to_c_int(self) -> i32 {
        match self {
            Self::Auto => -1,
            Self::Off => 0,
            Self::On => 1,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

impl std::fmt::Display for FlashAttnSetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Pre-normalizes CLI arguments before passing to Clap to support single-dash aliases
/// commonly used in llama.cpp / llama-server (`-ngl`, `-fa`, `-ub`).
pub fn normalize_cli_args<I, T>(args: I) -> Vec<String>
where
    I: IntoIterator<Item = T>,
    T: Into<String>,
{
    args.into_iter()
        .map(|item| {
            let s = item.into();
            if s == "-ngl" {
                "--ngl".to_string()
            } else if let Some(rest) = s.strip_prefix("-ngl=") {
                format!("--ngl={rest}")
            } else if s == "-fa" {
                "--flash-attn".to_string()
            } else if let Some(rest) = s.strip_prefix("-fa=") {
                format!("--flash-attn={rest}")
            } else if s == "-ub" {
                "--ubatch-size".to_string()
            } else if let Some(rest) = s.strip_prefix("-ub=") {
                format!("--ubatch-size={rest}")
            } else {
                s
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attn_mapping() {
        assert_eq!(FlashAttnSetting::Auto.to_c_int(), -1);
        assert_eq!(FlashAttnSetting::Off.to_c_int(), 0);
        assert_eq!(FlashAttnSetting::On.to_c_int(), 1);

        assert_eq!(FlashAttnSetting::Auto.to_string(), "auto");
        assert_eq!(FlashAttnSetting::Off.to_string(), "off");
        assert_eq!(FlashAttnSetting::On.to_string(), "on");
    }

    #[test]
    fn test_normalize_cli_args() {
        let raw = vec![
            "log-pi".to_string(),
            "-m".to_string(),
            "model.gguf".to_string(),
            "-ngl".to_string(),
            "33".to_string(),
            "-fa".to_string(),
            "off".to_string(),
            "-b".to_string(),
            "1024".to_string(),
            "-ub".to_string(),
            "128".to_string(),
        ];
        let normalized = normalize_cli_args(raw);
        assert_eq!(
            normalized,
            vec![
                "log-pi",
                "-m",
                "model.gguf",
                "--ngl",
                "33",
                "--flash-attn",
                "off",
                "-b",
                "1024",
                "--ubatch-size",
                "128",
            ]
        );

        let parsed = Cli::parse_from(normalized);
        assert_eq!(parsed.gpu_layers, 33);
        assert_eq!(parsed.flash_attn, FlashAttnSetting::Off);
        assert_eq!(parsed.batch_size, 1024);
        assert_eq!(parsed.ubatch_size, 128);
    }

    #[test]
    fn test_cli_default_batch_and_fa() {
        let args = vec!["log-pi", "-m", "model.gguf"];
        let parsed = Cli::parse_from(args);
        assert_eq!(parsed.batch_size, 2048);
        assert_eq!(parsed.ubatch_size, 512);
        assert_eq!(parsed.flash_attn, FlashAttnSetting::Auto);
    }
}
