//! Qwen3's Hugging Face Jinja template rendered without a Python runtime.

use serde_json::Value;

use super::{ChatMessage, ChatTemplateOptions, Result, TokenizerWorkerError};

const TOOL_INSTRUCTIONS: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const TOOL_SUFFIX: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";

pub(super) fn render(messages: &[ChatMessage], options: ChatTemplateOptions<'_>) -> Result<String> {
    if messages.is_empty() {
        return Err(template_error("messages 不能为空"));
    }
    let mut rendered = String::new();
    if !options.tools.is_empty() {
        rendered.push_str("<|im_start|>system\n");
        if messages[0].role == "system" {
            rendered.push_str(content(&messages[0]));
            rendered.push_str("\n\n");
        }
        rendered.push_str(TOOL_INSTRUCTIONS);
        for tool in options.tools {
            rendered.push('\n');
            write_json(&mut rendered, tool)?;
        }
        rendered.push_str(TOOL_SUFFIX);
    } else if messages[0].role == "system" {
        write_message(&mut rendered, "system", content(&messages[0]));
    }

    let last_query = messages
        .iter()
        .rposition(|message| {
            message.role == "user"
                && message.content.as_deref().is_some_and(|content| {
                    !(content.starts_with("<tool_response>")
                        && content.ends_with("</tool_response>"))
                })
        })
        .unwrap_or(messages.len() - 1);

    for (index, message) in messages.iter().enumerate() {
        match message.role.as_str() {
            "system" if index == 0 => {}
            "system" | "user" => write_message(&mut rendered, &message.role, content(message)),
            "assistant" => {
                write_assistant(&mut rendered, message, index, last_query, messages.len())?
            }
            "tool" => {
                if index == 0 || messages[index - 1].role != "tool" {
                    rendered.push_str("<|im_start|>user");
                }
                rendered.push_str("\n<tool_response>\n");
                rendered.push_str(content(message));
                rendered.push_str("\n</tool_response>");
                if index + 1 == messages.len() || messages[index + 1].role != "tool" {
                    rendered.push_str("<|im_end|>\n");
                }
            }
            role => return Err(template_error(format!("不支持的 Qwen3 消息角色: {role}"))),
        }
    }

    if options.add_generation_prompt {
        rendered.push_str("<|im_start|>assistant\n");
        if options.effective_enable_thinking() == Some(false) {
            rendered.push_str("<think>\n\n</think>\n\n");
        }
    }
    Ok(rendered)
}

fn content(message: &ChatMessage) -> &str {
    message.content.as_deref().unwrap_or("")
}

fn write_message(rendered: &mut String, role: &str, content: &str) {
    rendered.push_str("<|im_start|>");
    rendered.push_str(role);
    rendered.push('\n');
    rendered.push_str(content);
    rendered.push_str("<|im_end|>\n");
}

fn write_assistant(
    rendered: &mut String,
    message: &ChatMessage,
    index: usize,
    last_query: usize,
    message_count: usize,
) -> Result<()> {
    let mut assistant_content = content(message);
    let extracted_reasoning;
    let reasoning = if let Some(reasoning) = message.reasoning_content.as_deref() {
        reasoning
    } else if assistant_content.contains("</think>") {
        let before = assistant_content.split("</think>").next().unwrap_or("");
        extracted_reasoning = before
            .trim_end_matches('\n')
            .rsplit("<think>")
            .next()
            .unwrap_or("")
            .trim_start_matches('\n');
        assistant_content = assistant_content
            .rsplit("</think>")
            .next()
            .unwrap_or("")
            .trim_start_matches('\n');
        extracted_reasoning
    } else {
        ""
    };

    rendered.push_str("<|im_start|>assistant\n");
    if index > last_query && (index + 1 == message_count || !reasoning.is_empty()) {
        rendered.push_str("<think>\n");
        rendered.push_str(reasoning.trim_matches('\n'));
        rendered.push_str("\n</think>\n\n");
        rendered.push_str(assistant_content.trim_start_matches('\n'));
    } else {
        rendered.push_str(assistant_content);
    }

    if let Some(calls) = &message.tool_calls {
        for (call_index, call) in calls.iter().enumerate() {
            if (call_index == 0 && !assistant_content.is_empty()) || call_index > 0 {
                rendered.push('\n');
            }
            write_tool_call(rendered, call)?;
        }
    }
    rendered.push_str("<|im_end|>\n");
    Ok(())
}

fn write_tool_call(rendered: &mut String, call: &Value) -> Result<()> {
    let call = call
        .get("function")
        .filter(|function| {
            function
                .as_object()
                .is_some_and(|object| !object.is_empty())
        })
        .unwrap_or(call);
    let name = call
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| template_error("tool_call 缺少字符串 name"))?;
    let arguments = call
        .get("arguments")
        .ok_or_else(|| template_error("tool_call 缺少 arguments"))?;
    rendered.push_str("<tool_call>\n{\"name\": \"");
    rendered.push_str(name);
    rendered.push_str("\", \"arguments\": ");
    if let Some(arguments) = arguments.as_str() {
        rendered.push_str(arguments);
    } else {
        write_json(rendered, arguments)?;
    }
    rendered.push_str("}\n</tool_call>");
    Ok(())
}

/// Transformers' `tojson` filter uses insertion order and spaced separators.
fn write_json(rendered: &mut String, value: &Value) -> Result<()> {
    match value {
        Value::Array(items) => {
            rendered.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    rendered.push_str(", ");
                }
                write_json(rendered, item)?;
            }
            rendered.push(']');
        }
        Value::Object(entries) => {
            rendered.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    rendered.push_str(", ");
                }
                rendered.push_str(
                    &serde_json::to_string(key)
                        .map_err(|error| template_error(error.to_string()))?,
                );
                rendered.push_str(": ");
                write_json(rendered, item)?;
            }
            rendered.push('}');
        }
        _ => rendered.push_str(
            &serde_json::to_string(value).map_err(|error| template_error(error.to_string()))?,
        ),
    }
    Ok(())
}

fn template_error(message: impl Into<String>) -> TokenizerWorkerError {
    TokenizerWorkerError::RenderChatTemplate(message.into())
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Deserialize)]
    struct GoldenCase {
        name: String,
        messages: Vec<ChatMessage>,
        tools: Vec<Value>,
        enable_thinking: Option<bool>,
        add_generation_prompt: bool,
        expected: String,
    }

    #[test]
    fn matches_hugging_face_qwen3_chat_template() {
        let cases: Vec<GoldenCase> =
            serde_json::from_str(include_str!("../../tests/fixtures/qwen3_chat_golden.json"))
                .unwrap();
        for case in cases {
            let actual = render(
                &case.messages,
                ChatTemplateOptions {
                    tools: &case.tools,
                    enable_thinking: case.enable_thinking,
                    add_generation_prompt: case.add_generation_prompt,
                    kwargs: None,
                },
            )
            .unwrap();
            assert_eq!(actual, case.expected, "{}", case.name);
        }
    }

    #[test]
    fn rejects_malformed_tool_call_instead_of_silently_omitting_it() {
        let mut assistant = ChatMessage::new("assistant", "");
        assistant.tool_calls = Some(vec![serde_json::json!({"arguments": {}})]);
        let error = render(
            &[ChatMessage::new("user", "call"), assistant],
            ChatTemplateOptions::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("name"));
    }

    #[test]
    fn reads_enable_thinking_from_template_kwargs() {
        let kwargs = serde_json::from_str(r#"{"enable_thinking":false}"#).unwrap();
        let rendered = render(
            &[ChatMessage::new("user", "hi")],
            ChatTemplateOptions {
                add_generation_prompt: true,
                kwargs: Some(&kwargs),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }
}
