//! Hugging Face `tokenizer.json` backed tokenizer worker.
//!
//! The worker is deliberately synchronous: the underlying `tokenizers` crate
//! performs the CPU-bound work and is safe to share through Axum state when a
//! future controller needs it.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

mod qwen3_template;

/// A chat message accepted by [`TokenizerWorker::apply_chat_template`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: None,
            extra: Map::new(),
        }
    }
}

/// Additional values passed to a model's chat template.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatTemplateOptions<'a> {
    pub add_generation_prompt: bool,
    pub tools: &'a [Value],
    pub enable_thinking: Option<bool>,
    pub kwargs: Option<&'a Map<String, Value>>,
}

impl ChatTemplateOptions<'_> {
    fn effective_enable_thinking(self) -> Option<bool> {
        self.enable_thinking.or_else(|| {
            self.kwargs
                .and_then(|kwargs| kwargs.get("enable_thinking"))
                .and_then(Value::as_bool)
        })
    }
}

/// Failures while loading or using a [`TokenizerWorker`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerWorkerError {
    LoadTokenizer { path: PathBuf, message: String },
    ReadConfig { path: PathBuf, message: String },
    InvalidTokenId(i64),
    UnsupportedRemoteCode,
    UnsupportedChatTemplateFormat,
    RenderChatTemplate(String),
}

impl fmt::Display for TokenizerWorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadTokenizer { path, message } => {
                write!(f, "无法加载 tokenizer 文件 {}: {message}", path.display())
            }
            Self::ReadConfig { path, message } => {
                write!(f, "无法读取 tokenizer 配置 {}: {message}", path.display())
            }
            Self::InvalidTokenId(id) => write!(f, "token id {id} 不在 u32 范围内"),
            Self::UnsupportedRemoteCode => write!(
                f,
                "未实现：Rust tokenizers 不执行 Hugging Face trust_remote_code"
            ),
            Self::UnsupportedChatTemplateFormat => write!(
                f,
                "未实现：仅支持 tokenizer_config.json 中字符串形式的 chat_template"
            ),
            Self::RenderChatTemplate(message) => write!(f, "渲染 chat_template 失败: {message}"),
        }
    }
}

impl std::error::Error for TokenizerWorkerError {}

pub type Result<T> = std::result::Result<T, TokenizerWorkerError>;

/// Rust equivalent of mini-sglang's in-process tokenizer worker.
///
/// `model_path` may point to a Hugging Face model directory (containing
/// `tokenizer.json`) or directly to a `tokenizer.json` file.
#[derive(Clone, Debug)]
pub struct TokenizerWorker {
    tokenizer: Tokenizer,
    chat_template: Option<String>,
}

impl TokenizerWorker {
    /// Loads a tokenizer serialized by Hugging Face as `tokenizer.json`.
    ///
    /// Native Rust tokenizers deserialize tokenizer data only. They cannot
    /// safely reproduce Python's `trust_remote_code` extension mechanism, so
    /// requesting it returns an explicit error instead of silently ignoring it.
    pub fn new(model_path: impl AsRef<Path>, trust_remote_code: bool) -> Result<Self> {
        if trust_remote_code {
            return Err(TokenizerWorkerError::UnsupportedRemoteCode);
        }

        let (tokenizer_path, model_dir) = resolve_tokenizer_path(model_path.as_ref());
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            TokenizerWorkerError::LoadTokenizer {
                path: tokenizer_path.clone(),
                message: error.to_string(),
            }
        })?;

        let chat_template = load_chat_template(&model_dir)?;
        Ok(Self {
            tokenizer,
            chat_template,
        })
    }

    /// Tokenizes text and includes any configured special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self.tokenizer.encode(text, true).map_err(|error| {
            TokenizerWorkerError::LoadTokenizer {
                path: PathBuf::from("<in-memory tokenizer>"),
                message: error.to_string(),
            }
        })?;
        Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    /// Decodes a sequence of token IDs.
    pub fn decode(&self, token_ids: &[i64], skip_special_tokens: bool) -> Result<String> {
        let ids = token_ids
            .iter()
            .copied()
            .map(|id| u32::try_from(id).map_err(|_| TokenizerWorkerError::InvalidTokenId(id)))
            .collect::<Result<Vec<_>>>()?;
        self.tokenizer
            .decode(&ids, skip_special_tokens)
            .map_err(|error| TokenizerWorkerError::LoadTokenizer {
                path: PathBuf::from("<in-memory tokenizer>"),
                message: error.to_string(),
            })
    }

    /// Convenience variant for incremental decoding of a single token.
    pub fn decode_token(&self, token_id: i64, skip_special_tokens: bool) -> Result<String> {
        self.decode(&[token_id], skip_special_tokens)
    }

    /// Renders the model's Hugging Face chat template when one is configured.
    ///
    /// Models without `chat_template` keep mini-sglang's original fallback:
    /// concatenate non-empty message contents separated by blank lines.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        self.apply_chat_template_with_options(
            messages,
            ChatTemplateOptions {
                add_generation_prompt,
                ..Default::default()
            },
        )
    }

    pub fn apply_chat_template_with_options(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions<'_>,
    ) -> Result<String> {
        let Some(template_source) = &self.chat_template else {
            return Ok(messages
                .iter()
                .filter_map(|message| {
                    message
                        .content
                        .as_deref()
                        .filter(|content| !content.is_empty())
                })
                .collect::<Vec<_>>()
                .join("\n\n"));
        };

        // Qwen3's Hugging Face template uses Python string methods and a
        // mutable Jinja namespace, neither of which Minijinja implements.
        // Render Qwen3's full tool and reasoning branches directly.
        if template_source.contains("namespace(multi_step_tool=true")
            && template_source.contains("<|im_start|>")
        {
            return qwen3_template::render(messages, options);
        }

        let mut environment = Environment::new();
        environment
            .add_template("chat", template_source)
            .map_err(|error| TokenizerWorkerError::RenderChatTemplate(error.to_string()))?;
        let mut template_context = options.kwargs.cloned().unwrap_or_default();
        template_context.insert(
            "messages".to_owned(),
            serde_json::to_value(messages)
                .map_err(|error| TokenizerWorkerError::RenderChatTemplate(error.to_string()))?,
        );
        template_context.insert("tools".to_owned(), Value::Array(options.tools.to_vec()));
        template_context.insert(
            "add_generation_prompt".to_owned(),
            Value::Bool(options.add_generation_prompt),
        );
        if let Some(enable_thinking) = options.effective_enable_thinking() {
            template_context.insert("enable_thinking".to_owned(), Value::Bool(enable_thinking));
        }
        environment
            .get_template("chat")
            .expect("template was added immediately before lookup")
            .render(template_context)
            .map_err(|error| TokenizerWorkerError::RenderChatTemplate(error.to_string()))
    }
}

fn resolve_tokenizer_path(model_path: &Path) -> (PathBuf, PathBuf) {
    if model_path.is_dir() {
        (model_path.join("tokenizer.json"), model_path.to_path_buf())
    } else {
        let model_dir = model_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        (model_path.to_path_buf(), model_dir)
    }
}

fn load_chat_template(model_dir: &Path) -> Result<Option<String>> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.is_file() {
        return Ok(None);
    }

    let config_text =
        fs::read_to_string(&config_path).map_err(|error| TokenizerWorkerError::ReadConfig {
            path: config_path.clone(),
            message: error.to_string(),
        })?;
    let config: serde_json::Value =
        serde_json::from_str(&config_text).map_err(|error| TokenizerWorkerError::ReadConfig {
            path: config_path,
            message: error.to_string(),
        })?;

    match config.get("chat_template") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(template)) => Ok(Some(template.clone())),
        Some(_) => Err(TokenizerWorkerError::UnsupportedChatTemplateFormat),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(chat_template: Option<&str>) -> TokenizerWorker {
        let tokenizer: Tokenizer = serde_json::from_str(
            r#"{
                "version":"1.0",
                "truncation":null,
                "padding":null,
                "added_tokens":[],
                "normalizer":null,
                "pre_tokenizer":{"type":"WhitespaceSplit"},
                "post_processor":null,
                "decoder":null,
                "model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"world":2},"unk_token":"<unk>"}
            }"#,
        )
        .expect("test tokenizer must deserialize");
        TokenizerWorker {
            tokenizer,
            chat_template: chat_template.map(str::to_owned),
        }
    }

    #[test]
    fn encodes_and_decodes() {
        let worker = worker(None);

        assert_eq!(worker.encode("hello world").unwrap(), vec![1, 2]);
        assert_eq!(worker.decode(&[1, 2], true).unwrap(), "hello world");
        assert_eq!(worker.decode_token(1, true).unwrap(), "hello");
    }

    #[test]
    fn falls_back_to_joined_message_content() {
        let worker = worker(None);
        let messages = [
            ChatMessage::new("system", ""),
            ChatMessage::new("user", "hello"),
            ChatMessage::new("assistant", "world"),
        ];

        assert_eq!(
            worker.apply_chat_template(&messages, true).unwrap(),
            "hello\n\nworld"
        );
    }

    #[test]
    fn renders_a_basic_hugging_face_chat_template() {
        let worker = worker(Some(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}",
        ));
        let messages = [ChatMessage::new("user", "hello")];

        assert_eq!(
            worker.apply_chat_template(&messages, true).unwrap(),
            "user: hello\nassistant: "
        );
    }

    #[test]
    fn rejects_out_of_range_token_ids() {
        let error = worker(None).decode(&[-1], true).unwrap_err();
        assert_eq!(error, TokenizerWorkerError::InvalidTokenId(-1));
    }

    #[test]
    fn renders_qwen3_role_content_template() {
        let worker = worker(Some(
            "{% set ns = namespace(multi_step_tool=true) %}<|im_start|>",
        ));
        let messages = [
            ChatMessage::new("system", "Helpful."),
            ChatMessage::new("user", "Say hi"),
        ];
        assert_eq!(
            worker.apply_chat_template(&messages, true).unwrap(),
            "<|im_start|>system\nHelpful.<|im_end|>\n<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn passes_extended_kwargs_to_generic_template() {
        let worker = worker(Some("{{ custom_key }}: {{ tools[0].function.name }}"));
        let tools = [serde_json::json!({"type": "function", "function": {"name": "lookup"}})];
        let kwargs = serde_json::from_str(r#"{"custom_key":"ready"}"#).unwrap();
        let rendered = worker
            .apply_chat_template_with_options(
                &[ChatMessage::new("user", "hi")],
                ChatTemplateOptions {
                    add_generation_prompt: true,
                    tools: &tools,
                    enable_thinking: None,
                    kwargs: Some(&kwargs),
                },
            )
            .unwrap();
        assert_eq!(rendered, "ready: lookup");
    }
}
