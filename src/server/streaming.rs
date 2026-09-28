//! OpenAI-compatible streaming response construction.

use std::{convert::Infallible, sync::Arc, time::Duration};

use axum::response::sse::Event;
use futures_util::stream;
use serde_json::{Value, json};

use crate::{scheduler::FinishReason, tokenizer::TokenizerWorker};

use super::manager::{IncrementalDetokenizer, RequestHandle};

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

fn content_chunk(kind: ApiKind, uid: u64, model: &str, content: &str) -> Value {
    let choice = match kind {
        ApiKind::Chat => json!({"index": 0, "delta": {"content": content}, "finish_reason": null}),
        ApiKind::Completion => json!({"index": 0, "text": content, "finish_reason": null}),
    };
    json!({
        "id": kind.id(uid),
        "object": kind.object(true),
        "created": created(),
        "model": model,
        "choices": [choice],
    })
}

fn finish_chunk(
    kind: ApiKind,
    uid: u64,
    model: &str,
    finish_reason: FinishReason,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Value {
    let choice = match kind {
        ApiKind::Chat => json!({"index": 0, "delta": {}, "finish_reason": reason(finish_reason)}),
        ApiKind::Completion => {
            json!({"index": 0, "text": "", "finish_reason": reason(finish_reason)})
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
    phase: StreamPhase,
}

enum StreamPhase {
    Running,
    FinishPending(FinishReason),
    DonePending,
    Complete,
}

/// Streaming state is dropped on client disconnect; `RequestHandle` aborts it.
pub fn response_stream(
    handle: RequestHandle,
    tokenizer: Arc<TokenizerWorker>,
    kind: ApiKind,
    model: String,
    prompt_tokens: usize,
) -> impl futures_util::Stream<Item = Result<Event, Infallible>> + Send + 'static {
    let state = StreamState {
        handle,
        detokenizer: IncrementalDetokenizer::new(tokenizer),
        kind,
        model,
        prompt_tokens,
        completion_tokens: 0,
        phase: StreamPhase::Running,
    };
    stream::unfold(state, |mut state| async move {
        match state.phase {
            StreamPhase::FinishPending(reason) => {
                state.phase = StreamPhase::DonePending;
                let item = finish_chunk(
                    state.kind,
                    state.handle.uid(),
                    &state.model,
                    reason,
                    state.prompt_tokens,
                    state.completion_tokens,
                );
                return Some((Ok(event(item)), state));
            }
            StreamPhase::DonePending => {
                state.phase = StreamPhase::Complete;
                return Some((Ok(Event::default().data("[DONE]")), state));
            }
            StreamPhase::Complete => return None,
            StreamPhase::Running => {}
        }
        loop {
            let next = tokio::time::timeout(REQUEST_TIMEOUT, state.handle.recv()).await;
            let token = match next {
                Ok(Some(token)) => token,
                Ok(None) => {
                    state.phase = StreamPhase::DonePending;
                    return Some((Ok(event(error_chunk("Scheduler closed"))), state));
                }
                Err(_) => {
                    state.phase = StreamPhase::DonePending;
                    return Some((
                        Ok(event(error_chunk("Timed out waiting for the next token"))),
                        state,
                    ));
                }
            };
            match token.finish_reason {
                Some(FinishReason::Abort) => {
                    state.phase = StreamPhase::DonePending;
                    return Some((
                        Ok(event(error_chunk("Request aborted by the scheduler"))),
                        state,
                    ));
                }
                Some(FinishReason::Error) => {
                    state.phase = StreamPhase::DonePending;
                    return Some((
                        Ok(event(error_chunk("Request failed during generation"))),
                        state,
                    ));
                }
                _ => {}
            }
            state.completion_tokens += 1;
            let content = match state.detokenizer.add_token(token.token_id) {
                Ok(content) => content,
                Err(error) => {
                    state.phase = StreamPhase::DonePending;
                    return Some((Ok(event(error_chunk(&error.to_string()))), state));
                }
            };
            if token.finished {
                let finish_reason = token.finish_reason.unwrap_or(FinishReason::Stop);
                // Emit the final text delta before the terminal chunk.
                if !content.is_empty() {
                    let item = event(content_chunk(
                        state.kind,
                        state.handle.uid(),
                        &state.model,
                        &content,
                    ));
                    state.phase = StreamPhase::FinishPending(finish_reason);
                    return Some((Ok(item), state));
                }
                state.phase = StreamPhase::DonePending;
                return Some((
                    Ok(event(finish_chunk(
                        state.kind,
                        state.handle.uid(),
                        &state.model,
                        finish_reason,
                        state.prompt_tokens,
                        state.completion_tokens,
                    ))),
                    state,
                ));
            }
            if !content.is_empty() {
                return Some((
                    Ok(event(content_chunk(
                        state.kind,
                        state.handle.uid(),
                        &state.model,
                        &content,
                    ))),
                    state,
                ));
            }
        }
    })
}

fn event(value: Value) -> Event {
    Event::default().data(value.to_string())
}
