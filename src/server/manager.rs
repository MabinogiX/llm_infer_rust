//! Owns the non-Send scheduler on a dedicated thread and routes its results.

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

use tokio::sync::{mpsc as async_mpsc, oneshot};

use crate::{
    engine::SamplingParams,
    scheduler::{FinishReason, OutputToken, RequestId, Scheduler},
    tokenizer::{TokenizerWorker, TokenizerWorkerError},
};

use super::{
    ServeArgs, build_components,
    output::{ChatOutputParser, ChatOutputParserConstructor},
};

#[derive(Debug)]
pub enum ManagerError {
    Startup(String),
    Closed,
    Scheduler(String),
}

impl fmt::Display for ManagerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Startup(message) => write!(f, "frontend 初始化失败: {message}"),
            Self::Closed => write!(f, "scheduler 线程已关闭"),
            Self::Scheduler(message) => write!(f, "scheduler 请求失败: {message}"),
        }
    }
}

impl std::error::Error for ManagerError {}

enum Command {
    Submit {
        input_ids: Vec<i64>,
        sampling_params: SamplingParams,
        reply: oneshot::Sender<Result<Submission, String>>,
    },
    Abort(RequestId),
}

struct Submission {
    uid: RequestId,
    output: async_mpsc::UnboundedReceiver<OutputToken>,
}

/// Sendable handle used by all HTTP request handlers.
#[derive(Clone)]
pub struct FrontendManager {
    commands: mpsc::Sender<Command>,
    tokenizer: Arc<TokenizerWorker>,
    output_parser_constructor: ChatOutputParserConstructor,
}

impl FrontendManager {
    /// Build all model state inside the thread that will run the scheduler.
    pub fn start(args: ServeArgs) -> Result<Self, ManagerError> {
        let (commands, command_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("minisgl-scheduler".to_owned())
            .spawn(move || match build_components(&args) {
                Ok(components) => {
                    let tokenizer = components.tokenizer;
                    tracing::info!("scheduler and model initialized");
                    if ready_tx
                        .send(Ok((
                            tokenizer.clone(),
                            components.output_parser_constructor,
                        )))
                        .is_ok()
                    {
                        run_scheduler(components.scheduler, command_rx);
                    }
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error.to_string()));
                }
            })
            .map_err(|error| ManagerError::Startup(error.to_string()))?;
        let (tokenizer, output_parser_constructor) = ready_rx
            .recv()
            .map_err(|_| ManagerError::Startup("scheduler 线程启动失败".to_owned()))?
            .map_err(ManagerError::Startup)?;
        Ok(Self {
            commands,
            tokenizer: Arc::new(tokenizer),
            output_parser_constructor,
        })
    }

    pub fn tokenizer(&self) -> &TokenizerWorker {
        &self.tokenizer
    }

    pub fn shared_tokenizer(&self) -> Arc<TokenizerWorker> {
        self.tokenizer.clone()
    }

    pub fn new_chat_output_parser(&self, uid: RequestId) -> Box<dyn ChatOutputParser> {
        (self.output_parser_constructor)(uid)
    }

    pub async fn submit_request(
        &self,
        input_ids: Vec<i64>,
        sampling_params: SamplingParams,
    ) -> Result<RequestHandle, ManagerError> {
        let (reply, response) = oneshot::channel();
        self.commands
            .send(Command::Submit {
                input_ids,
                sampling_params,
                reply,
            })
            .map_err(|_| ManagerError::Closed)?;
        let submitted = response
            .await
            .map_err(|_| ManagerError::Closed)?
            .map_err(ManagerError::Scheduler)?;
        Ok(RequestHandle {
            uid: submitted.uid,
            output: submitted.output,
            commands: self.commands.clone(),
        })
    }
}

/// Dropping this handle cancels an unfinished generation, including on disconnect.
pub struct RequestHandle {
    uid: RequestId,
    output: async_mpsc::UnboundedReceiver<OutputToken>,
    commands: mpsc::Sender<Command>,
}

impl RequestHandle {
    pub fn uid(&self) -> RequestId {
        self.uid
    }

    pub async fn recv(&mut self) -> Option<OutputToken> {
        self.output.recv().await
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Abort(self.uid));
    }
}

fn run_scheduler(mut scheduler: Scheduler, commands: mpsc::Receiver<Command>) {
    let mut results: HashMap<RequestId, async_mpsc::UnboundedSender<OutputToken>> = HashMap::new();
    tracing::info!("scheduler event loop started");
    loop {
        if scheduler.is_idle() {
            match commands.recv_timeout(Duration::from_millis(10)) {
                Ok(command) => handle_command(command, &mut scheduler, &mut results),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        for _ in 0..64 {
            match commands.try_recv() {
                Ok(command) => handle_command(command, &mut scheduler, &mut results),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        }
        if scheduler.is_idle() {
            continue;
        }
        match scheduler.step() {
            Ok(outputs) => {
                if let Some(error) = scheduler.last_step_error() {
                    tracing::error!(error = %error, "model forward or sampling failed");
                }
                for output in outputs {
                    let uid = output.uid;
                    if let Some(sender) = results.get(&uid) {
                        let disconnected = sender.send(output.clone()).is_err();
                        if disconnected || output.finished {
                            results.remove(&uid);
                        }
                        if disconnected {
                            scheduler.abort_request(uid);
                        }
                    }
                }
            }
            Err(error) => {
                tracing::error!(error = %error, "scheduler step failed");
                for (uid, sender) in results.drain() {
                    let _ = sender.send(OutputToken {
                        uid,
                        token_id: 0,
                        finished: true,
                        finish_reason: Some(FinishReason::Error),
                    });
                    scheduler.abort_request(uid);
                }
            }
        }
    }
    tracing::info!("scheduler event loop stopped");
}

fn handle_command(
    command: Command,
    scheduler: &mut Scheduler,
    results: &mut HashMap<RequestId, async_mpsc::UnboundedSender<OutputToken>>,
) {
    match command {
        Command::Submit {
            input_ids,
            sampling_params,
            reply,
        } => match scheduler.add_request(input_ids, sampling_params) {
            Ok(uid) => {
                let (sender, output) = async_mpsc::unbounded_channel();
                results.insert(uid, sender);
                if reply.send(Ok(Submission { uid, output })).is_err() {
                    results.remove(&uid);
                    scheduler.abort_request(uid);
                }
            }
            Err(error) => {
                let _ = reply.send(Err(error.to_string()));
            }
        },
        Command::Abort(uid) => {
            scheduler.abort_request(uid);
            if let Some(sender) = results.remove(&uid) {
                let _ = sender.send(OutputToken {
                    uid,
                    token_id: 0,
                    finished: true,
                    finish_reason: Some(FinishReason::Abort),
                });
            }
        }
    }
}

/// Re-decodes the accumulated sequence so partial UTF-8 bytes stay buffered.
pub struct IncrementalDetokenizer {
    tokenizer: Arc<TokenizerWorker>,
    token_ids: Vec<i64>,
    text: String,
}

impl IncrementalDetokenizer {
    pub fn new(tokenizer: Arc<TokenizerWorker>) -> Self {
        Self {
            tokenizer,
            token_ids: Vec::new(),
            text: String::new(),
        }
    }

    pub fn add_token(&mut self, token_id: i64) -> Result<String, TokenizerWorkerError> {
        self.token_ids.push(token_id);
        let decoded = self.tokenizer.decode(&self.token_ids, true)?;
        let stable = decoded.strip_suffix('\u{fffd}').unwrap_or(&decoded);
        let delta = stable.strip_prefix(&self.text).unwrap_or(stable).to_owned();
        self.text = stable.to_owned();
        Ok(delta)
    }
}
