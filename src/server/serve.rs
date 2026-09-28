//! Startup path corresponding to `minisgl/server/serve.py`.

use std::{fmt, io, net::SocketAddr};

use tokio::net::TcpListener;

use crate::{
    engine::{EngineError, ServerArgs},
    logging::LoggingConfig,
    scheduler::SchedulerError,
    tokenizer::TokenizerWorkerError,
};

use super::{
    api,
    manager::{FrontendManager, ManagerError},
};

/// HTTP binding and engine settings for one model server.
#[derive(Debug, Clone)]
pub struct ServeArgs {
    pub engine: ServerArgs,
    pub bind: SocketAddr,
    pub logging: LoggingConfig,
}

impl ServeArgs {
    pub fn new(engine: ServerArgs) -> Self {
        Self {
            engine,
            bind: SocketAddr::from(([127, 0, 0, 1], 8000)),
            logging: LoggingConfig::default(),
        }
    }
}

#[derive(Debug)]
pub enum ServeError {
    Engine(EngineError),
    Scheduler(SchedulerError),
    Tokenizer(TokenizerWorkerError),
    Io(io::Error),
    Frontend(ManagerError),
    UnsupportedModel {
        model_type: String,
        architectures: Vec<String>,
    },
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Engine(error) => write!(f, "engine 初始化失败: {error}"),
            Self::Scheduler(error) => write!(f, "scheduler 初始化失败: {error}"),
            Self::Tokenizer(error) => write!(f, "tokenizer 初始化失败: {error}"),
            Self::Io(error) => write!(f, "HTTP 服务失败: {error}"),
            Self::Frontend(error) => write!(f, "frontend 启动失败: {error}"),
            Self::UnsupportedModel {
                model_type,
                architectures,
            } => write!(
                f,
                "不支持的模型: model_type={model_type}, architectures={architectures:?}"
            ),
        }
    }
}

impl std::error::Error for ServeError {}

/// Initialize the model on the scheduler thread, then serve the inference API.
pub async fn serve(args: ServeArgs) -> Result<(), ServeError> {
    let frontend = FrontendManager::start(args.clone()).map_err(ServeError::Frontend)?;
    let app = api::router(frontend);
    let listener = TcpListener::bind(args.bind).await.map_err(ServeError::Io)?;
    tracing::info!(address = %listener.local_addr().map_err(ServeError::Io)?, "HTTP server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            match tokio::signal::ctrl_c().await {
                Ok(()) => tracing::info!("shutdown requested"),
                Err(error) => tracing::error!(error = %error, "unable to listen for Ctrl+C"),
            }
        })
        .await
        .map_err(ServeError::Io)
}
