//! Startup path corresponding to `minisgl/server/serve.py`.

use std::{fmt, io, net::SocketAddr};

use tokio::net::TcpListener;

use crate::{
    engine::{Engine, EngineError, ModelArgs, ServerArgs},
    logging::LoggingConfig,
    models::Qwen3Factory,
    scheduler::{Scheduler, SchedulerError},
    tokenizer::{TokenizerWorker, TokenizerWorkerError},
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

/// The initialized components handed to the HTTP frontend.
pub struct ServeComponents {
    pub scheduler: Scheduler,
    pub tokenizer: TokenizerWorker,
}

#[derive(Debug)]
pub enum ServeError {
    Engine(EngineError),
    Scheduler(SchedulerError),
    Tokenizer(TokenizerWorkerError),
    Io(io::Error),
    Frontend(ManagerError),
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Engine(error) => write!(f, "engine 初始化失败: {error}"),
            Self::Scheduler(error) => write!(f, "scheduler 初始化失败: {error}"),
            Self::Tokenizer(error) => write!(f, "tokenizer 初始化失败: {error}"),
            Self::Io(error) => write!(f, "HTTP 服务失败: {error}"),
            Self::Frontend(error) => write!(f, "frontend 启动失败: {error}"),
        }
    }
}

impl std::error::Error for ServeError {}

/// Load model metadata, tokenizer and weights, then create the scheduler.
pub fn build_components(args: &ServeArgs) -> Result<ServeComponents, ServeError> {
    let model_args =
        ModelArgs::from_pretrained(&args.engine.model_path).map_err(ServeError::Engine)?;
    let tokenizer = TokenizerWorker::new(&args.engine.model_path, args.engine.trust_remote_code)
        .map_err(ServeError::Tokenizer)?;
    let mut engine = Engine::new(args.engine.clone(), model_args, 0).map_err(ServeError::Engine)?;
    engine
        .build_model(&Qwen3Factory)
        .map_err(ServeError::Engine)?;
    engine.load_model_weights().map_err(ServeError::Engine)?;
    let scheduler = Scheduler::new(engine).map_err(ServeError::Scheduler)?;
    Ok(ServeComponents {
        scheduler,
        tokenizer,
    })
}

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
