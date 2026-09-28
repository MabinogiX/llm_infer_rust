//! OpenAI-compatible streaming response construction.

use std::{
    collections::VecDeque,
    convert::Infallible,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::response::sse::Event;
use futures_util::stream;
use serde_json::{Value, json};

use crate::{logging::format_duration, scheduler::FinishReason, tokenizer::TokenizerWorker};

use super::{
    manager::{IncrementalDetokenizer, RequestHandle},
    output::ChatOutputParser,
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone, Copy)]
pub enum ApiKind {
    Chat,
    Completion,
}

impl ApiKind {
    pub fn id(self, uid: u64) -> String {
        match self {
            Self::Chat => format!("chatcmpl-{uid}"),
            Self::Completion => format!("cmpl-{uid}"),
        }
    }

    pub fn object(self, streaming: bool) -> &'static str {
        match (self, streaming) {
            (Self::Chat, true) => "chat.completion.chunk",
            (Self::Chat, false) => "chat.completion",
            (Self::Completion, _) => "text_completion",
        }
    }
}

pub fn reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Abort => "abort",
        FinishReason::Error => "error",
    }
}

pub fn chat_reason(reason: FinishReason, has_tool_calls: bool) -> &'static str {
    if has_tool_calls && reason == FinishReason::Stop {
        "tool_calls"
    } else {
        self::reason(reason)
    }
}

pub fn created() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|time| time.as_secs())
        .unwrap_or(0)
}

pub fn usage(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

fn chat_delta_chunk(uid: u64, model: &str, delta: Value) -> Value {
    json!({
        "id": ApiKind::Chat.id(uid),
        "object": ApiKind::Chat.object(true),
        "created": created(),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
    })
}

fn completion_chunk(uid: u64, model: &str, content: &str) -> Value {
    json!({
        "id": ApiKind::Completion.id(uid),
        "object": ApiKind::Completion.object(true),
        "created": created(),
        "model": model,
        "choices": [{"index": 0, "text": content, "finish_reason": null}],
    })
}

fn finish_chunk(
    kind: ApiKind,
    uid: u64,
    model: &str,
    finish_reason: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Value {
    let choice = match kind {
        ApiKind::Chat => json!({"index": 0, "delta": {}, "finish_reason": finish_reason}),
        ApiKind::Completion => {
            json!({"index": 0, "text": "", "finish_reason": finish_reason})
        }
    };
    json!({
        "id": kind.id(uid),
        "object": kind.object(true),
        "created": created(),
        "model": model,
        "choices": [choice],
        "usage": usage(prompt_tokens, completion_tokens),
    })
}

fn error_chunk(message: &str) -> Value {
    json!({"error": {"message": message}})
}

struct StreamState {
    handle: RequestHandle,
    detokenizer: IncrementalDetokenizer,
    kind: ApiKind,
    model: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    started_at: Instant,
    chat_output: Option<Box<dyn ChatOutputParser>>,
    events: VecDeque<Event>,
    done: bool,
}

impl Drop for StreamState {
    fn drop(&mut self) {
        if !self.done {
            tracing::warn!(
                uid = self.handle.uid(),
                duration_ms = %format_duration(self.started_at.elapsed()),
                "stream disconnected before completion"
            );
        }
    }
}

/// Streaming state is dropped on client disconnect; `RequestHandle` aborts it.
pub fn response_stream(
    handle: RequestHandle,
    tokenizer: Arc<TokenizerWorker>,
    kind: ApiKind,
    model: String,
    prompt_tokens: usize,
    chat_output: Option<Box<dyn ChatOutputParser>>,
    started_at: Instant,
) -> impl futures_util::Stream<Item = Result<Event, Infallible>> + Send + 'static {
    let mut events = VecDeque::new();
    if matches!(kind, ApiKind::Chat) {
        events.push_back(event(chat_delta_chunk(
            handle.uid(),
            &model,
            json!({"role": "assistant"}),
        )));
    }
    let state = StreamState {
        chat_output,
        handle,
        detokenizer: IncrementalDetokenizer::new(tokenizer),
        kind,
        model,
        prompt_tokens,
        completion_tokens: 0,
        started_at,
        events,
        done: false,
    };
    stream::unfold(state, |mut state| async move {
        if let Some(next) = state.events.pop_front() {
            return Some((Ok(next), state));
        }
        if state.done {
            return None;
        }
        loop {
            let next = tokio::time::timeout(REQUEST_TIMEOUT, state.handle.recv()).await;
            let token = match next {
                Ok(Some(token)) => token,
                Ok(None) => {
                    tracing::warn!(
                        uid = state.handle.uid(),
                        duration_ms = %format_duration(state.started_at.elapsed()),
                        "stream ended because scheduler closed"
                    );
                    state
                        .events
                        .push_back(event(error_chunk("Scheduler closed")));
                    state.done = true;
                    break;
                }
                Err(_) => {
                    tracing::warn!(
                        uid = state.handle.uid(),
                        duration_ms = %format_duration(state.started_at.elapsed()),
                        "stream timed out waiting for a token"
                    );
                    state
                        .events
                        .push_back(event(error_chunk("Timed out waiting for the next token")));
                    state.done = true;
                    break;
                }
            };
            match token.finish_reason {
                Some(FinishReason::Abort) => {
                    tracing::warn!(
                        uid = state.handle.uid(),
                        duration_ms = %format_duration(state.started_at.elapsed()),
                        "stream request aborted"
                    );
                    state
                        .events
                        .push_back(event(error_chunk("Request aborted by the scheduler")));
                    state.done = true;
                    break;
                }
                Some(FinishReason::Error) => {
                    tracing::error!(
                        uid = state.handle.uid(),
                        duration_ms = %format_duration(state.started_at.elapsed()),
                        "stream generation failed"
                    );
                    state
                        .events
                        .push_back(event(error_chunk("Request failed during generation")));
                    state.done = true;
                    break;
                }
                _ => {}
            }
            state.completion_tokens += 1;
            let content = match state.detokenizer.add_token(token.token_id) {
                Ok(content) => content,
                Err(error) => {
                    tracing::error!(
                        uid = state.handle.uid(),
                        duration_ms = %format_duration(state.started_at.elapsed()),
                        error = %error,
                        "stream detokenization failed"
                    );
                    state
                        .events
                        .push_back(event(error_chunk(&error.to_string())));
                    state.done = true;
                    break;
                }
            };
            if let Some(output) = &mut state.chat_output {
                for delta in output.push(&content) {
                    state.events.push_back(event(chat_delta_chunk(
                        state.handle.uid(),
                        &state.model,
                        delta,
                    )));
                }
            } else if !content.is_empty() {
                state.events.push_back(event(completion_chunk(
                    state.handle.uid(),
                    &state.model,
                    &content,
                )));
            }
            if token.finished {
                let finish_reason = token.finish_reason.unwrap_or(FinishReason::Stop);
                let finish_reason = if let Some(output) = &mut state.chat_output {
                    for delta in output.finish() {
                        state.events.push_back(event(chat_delta_chunk(
                            state.handle.uid(),
                            &state.model,
                            delta,
                        )));
                    }
                    chat_reason(finish_reason, output.has_tool_calls())
                } else {
                    reason(finish_reason)
                };
                tracing::info!(
                    uid = state.handle.uid(),
                    completion_tokens = state.completion_tokens,
                    finish_reason,
                    duration_ms = %format_duration(state.started_at.elapsed()),
                    "stream generation completed"
                );
                state.events.push_back(event(finish_chunk(
                    state.kind,
                    state.handle.uid(),
                    &state.model,
                    finish_reason,
                    state.prompt_tokens,
                    state.completion_tokens,
                )));
                state.done = true;
            }
            if !state.events.is_empty() {
                break;
            }
        }
        if state.done {
            state.events.push_back(Event::default().data("[DONE]"));
        }
        Some((Ok(state.events.pop_front().unwrap()), state))
    })
}

fn event(value: Value) -> Event {
    Event::default().data(value.to_string())
}
