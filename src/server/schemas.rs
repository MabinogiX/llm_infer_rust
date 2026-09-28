//! OpenAI-compatible request shapes accepted by the HTTP frontend.

use serde::Deserialize;

use crate::{engine::SamplingParams, tokenizer::ChatMessage};

fn default_model() -> String {
    "default".to_owned()
}

fn default_top_p() -> f64 {
    1.0
}

fn default_top_k() -> i64 {
    -1
}

fn default_max_tokens() -> i64 {
    1024
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_top_k")]
    pub top_k: i64,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i64,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub ignore_eos: bool,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_top_k")]
    pub top_k: i64,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i64,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub ignore_eos: bool,
}

pub fn sampling_params(
    temperature: f64,
    top_p: f64,
    top_k: i64,
    max_tokens: i64,
    ignore_eos: bool,
) -> SamplingParams {
    SamplingParams {
        temperature,
        top_p,
        top_k,
        max_tokens: usize::try_from(max_tokens).unwrap_or(1),
        ignore_eos,
    }
    .normalized()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_clamps_http_sampling() {
        let request: CompletionRequest = serde_json::from_str(r#"{"prompt":"hello"}"#).unwrap();
        assert_eq!(request.model, "default");
        assert_eq!(request.max_tokens, 1024);
        assert!(!request.stream);
        let params = sampling_params(0.0, 2.0, -1, -4, false);
        assert_eq!(params.max_tokens, 1);
        assert_eq!(params.top_p, 1.0);
    }
}
