//! OpenAI-compatible request shapes accepted by the HTTP frontend.

use serde::Deserialize;
use serde_json::{Map, Value};

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
    pub tools: Vec<Value>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub chat_template_kwargs: Map<String, Value>,
    #[serde(default)]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_top_k")]
    pub top_k: i64,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i64,
    #[serde(default)]
    pub max_completion_tokens: Option<i64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub ignore_eos: bool,
}

impl ChatCompletionRequest {
    /// The newer chat API limit takes precedence over the legacy `max_tokens`.
    pub fn sampling_params(&self) -> SamplingParams {
        sampling_params(
            self.temperature,
            self.top_p,
            self.top_k,
            self.max_completion_tokens.unwrap_or(self.max_tokens),
            self.ignore_eos,
        )
    }
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

    #[test]
    fn resolves_chat_token_limits_into_sampling_params() {
        for (fields, expected) in [
            (serde_json::json!({}), 1024),
            (serde_json::json!({"max_tokens": 7}), 7),
            (serde_json::json!({"max_completion_tokens": 32}), 32),
            (
                serde_json::json!({"max_tokens": 7, "max_completion_tokens": 32}),
                32,
            ),
            (
                serde_json::json!({"max_tokens": 7, "max_completion_tokens": null}),
                7,
            ),
            (serde_json::json!({"max_completion_tokens": null}), 1024),
            (
                serde_json::json!({"max_tokens": 7, "max_completion_tokens": 0}),
                1,
            ),
            (
                serde_json::json!({"max_tokens": 7, "max_completion_tokens": -4}),
                1,
            ),
        ] {
            let mut payload = fields.clone();
            payload["messages"] = serde_json::json!([{"role": "user", "content": "hi"}]);
            payload["temperature"] = serde_json::json!(0.5);
            payload["top_p"] = serde_json::json!(0.9);
            payload["top_k"] = serde_json::json!(10);
            payload["ignore_eos"] = serde_json::json!(true);
            let request: ChatCompletionRequest = serde_json::from_value(payload).unwrap();
            let params = request.sampling_params();
            assert_eq!(params.max_tokens, expected, "{fields}");
            assert_eq!(params.temperature, 0.5);
            assert_eq!(params.top_p, 0.9);
            assert_eq!(params.top_k, 10);
            assert!(params.ignore_eos);
        }
    }

    #[test]
    fn preserves_tools_tool_calls_and_template_kwargs() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages": [{
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "checking",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"city": "北京"}}
                    }],
                    "tool_call_id": "previous_call"
                }],
                "tools": [{"type": "function", "function": {
                    "name": "lookup", "parameters": {"type": "object"}
                }}],
                "chat_template_kwargs": {"enable_thinking": false, "custom_key": "value"}
            }"#,
        )
        .unwrap();
        assert_eq!(request.messages[0].content, None);
        assert_eq!(
            request.messages[0].reasoning_content.as_deref(),
            Some("checking")
        );
        assert_eq!(
            request.messages[0].tool_calls.as_ref().unwrap()[0]["id"],
            "call_1"
        );
        assert_eq!(request.messages[0].extra["tool_call_id"], "previous_call");
        assert_eq!(request.tools[0]["function"]["name"], "lookup");
        assert_eq!(request.chat_template_kwargs["enable_thinking"], false);
        assert_eq!(request.chat_template_kwargs["custom_key"], "value");
    }
}
