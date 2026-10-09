//! log-pi: System One Decision Engine in Rust
//! Brand Identity: log(π)
//! Kernel Mathematical Formula: log(P_i) = z_i - ln \sum_j e^{z_j}

use std::sync::Arc;

use clap::Parser;
use log_pi::cli::Cli;
use log_pi::config::ServerConfig;
use log_pi::engine::template::ChatTemplate;
use log_pi::engine::{spawn_engine_worker, LlamaEngine, LlamaModel};
use log_pi::server::{cors_layer, create_router, AppState};
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

const BANNER: &str = r#"
  ++                    ##              @@
  ==                  **    ########%%    %%
  ==    ==      ++++  **      ##  ##      %%
  --  ==  ==  ==  ++  ++      **  ****    ##
  --  --  ==  ==  ==  ++      **  ***     ##
  ::    --      ====    ==              **
                  --
              ::--
"#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize structured logging
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "log_pi=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    eprintln!("{}", BANNER);

    let args = log_pi::cli::normalize_cli_args(std::env::args());
    let cli = Cli::parse_from(args);
    if cli.list_devices {
        LlamaModel::print_devices();
        return Ok(());
    }

    let config = match ServerConfig::from_cli(cli) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("Configuration error: {err}");
            std::process::exit(1);
        }
    };

    info!(
        model = %config.model_path.display(),
        bind = %config.bind_addr,
        ctx_size = config.ctx_size,
        threads = config.threads,
        gpu_layers = config.gpu_layers,
        device = ?config.device,
        batch_size = config.batch_size,
        ubatch_size = config.ubatch_size,
        flash_attn = %config.flash_attn,
        "Initializing log(π) System One Decision Engine..."
    );

    let model = Arc::new(LlamaModel::load(
        config.model_path.to_str().expect("valid utf-8 path"),
        config.gpu_layers,
        config.device.as_deref(),
    )?);

    let template = ChatTemplate::for_model(&model, config.chat_template.as_deref())?;
    let engine = LlamaEngine::new(model, template, config.context_params())?;
    let model_info = engine.model_info();
    let (engine_handle, engine_worker) = spawn_engine_worker(
        engine,
        config.limits.queue_depth,
        config.limits.request_timeout,
    )?;
    let app_state = AppState::new(
        engine_handle,
        model_info,
        config.model_name.clone(),
        config.limits,
    );

    let mut app = create_router(app_state).layer(TraceLayer::new_for_http());
    match cors_layer(&config.cors_origins)? {
        Some(cors) => {
            info!(origins = ?config.cors_origins, "CORS enabled");
            app = app.layer(cors);
        }
        None => info!("CORS disabled (set --cors-origins to enable)"),
    }

    let listener = TcpListener::bind(config.bind_addr).await?;
    info!("Server listening on http://{}", config.bind_addr);
    info!(
        "Endpoint POST http://{}/v1/systemone ready for requests",
        config.bind_addr
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // `serve` consumed the router, so every engine handle is gone and the worker can drain.
    engine_worker.join();
    info!("log-pi shutdown cleanly");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received Ctrl+C, initiating shutdown..."),
        _ = terminate => info!("Received SIGTERM, initiating shutdown..."),
    }
}
