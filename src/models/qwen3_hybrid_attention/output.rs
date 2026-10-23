//! Parse Qwen3's generated reasoning and tool-call tags into chat fields.

use serde_json::{Value, json};

use crate::server::output::ChatOutputParser;

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_START: &str = "<tool_call>";
const TOOL_END: &str = "</tool_call>";

#[derive(Clone, Copy)]
enum Mode {
    Content,
    Reasoning,
    ToolCall,
}

pub fn new_parser(uid: u64) -> Box<dyn ChatOutputParser> {
    Box::new(HybridOutputParser::new(uid))
}

struct HybridOutputParser {
    uid: u64,
    mode: Mode,
    pending: String,
    content: String,
    reasoning: Option<String>,
    tool_calls: Vec<Value>,
}

impl HybridOutputParser {
    fn new(uid: u64) -> Self {
        Self {
            uid,
            mode: Mode::Content,
            pending: String::new(),
            content: String::new(),
            reasoning: None,
            tool_calls: Vec::new(),
        }
    }
}

impl ChatOutputParser for HybridOutputParser {
    fn set_prompt(&mut self, prompt: &str) {
        if prompt.trim_end().ends_with("<think>") {
            self.mode = Mode::Reasoning;
            self.reasoning = Some(String::new());
        }
    }
    /// Returns OpenAI chat delta objects. A tool call is emitted when its
    /// closing tag arrives, since its JSON arguments are not valid before then.
    fn push(&mut self, text: &str) -> Vec<Value> {
        self.pending.push_str(text);
        let mut deltas = Vec::new();
        loop {
            match self.mode {
                Mode::Content => {
                    if let Some((position, marker)) = next_marker(&self.pending) {
                        let before = self.pending[..position].to_owned();
                        self.push_content(&before, &mut deltas);
                        self.pending.drain(..position + marker.len());
                        self.mode = if marker == THINK_START {
                            self.reasoning.get_or_insert_with(String::new);
                            Mode::Reasoning
                        } else {
                            Mode::ToolCall
                        };
                        continue;
                    }
                    let safe = safe_prefix_len(&self.pending, &[THINK_START, TOOL_START]);
                    if safe == 0 {
                        break;
                    }
                    let before = self.pending[..safe].to_owned();
                    self.pending.drain(..safe);
                    self.push_content(&before, &mut deltas);
                }
                Mode::Reasoning => {
                    if let Some(position) = self.pending.find(THINK_END) {
                        let before = self.pending[..position].to_owned();
                        self.push_reasoning(&before, &mut deltas);
                        self.pending.drain(..position + THINK_END.len());
                        self.mode = Mode::Content;
                        continue;
                    }
                    let safe = safe_prefix_len(&self.pending, &[THINK_END]);
                    if safe == 0 {
                        break;
                    }
                    let before = self.pending[..safe].to_owned();
                    self.pending.drain(..safe);
                    self.push_reasoning(&before, &mut deltas);
                }
                Mode::ToolCall => {
                    let Some(position) = self.pending.find(TOOL_END) else {
                        break;
                    };
                    let body = self.pending[..position].to_owned();
                    self.pending.drain(..position + TOOL_END.len());
                    if let Some(call) = parse_tool_call(&body, self.uid, self.tool_calls.len()) {
                        deltas.push(json!({"tool_calls": [{
                            "index": self.tool_calls.len(),
                            "id": call["id"],
                            "type": "function",
                            "function": call["function"],
                        }]}));
                        self.tool_calls.push(call);
                    } else {
                        self.push_content(&format!("{TOOL_START}{body}{TOOL_END}"), &mut deltas);
                    }
                    self.mode = Mode::Content;
                    continue;
                }
            }
            break;
        }
        deltas
    }

    fn finish(&mut self) -> Vec<Value> {
        let pending = std::mem::take(&mut self.pending);
        let mut deltas = Vec::new();
        match self.mode {
            Mode::Content => self.push_content(&pending, &mut deltas),
            Mode::Reasoning => self.push_reasoning(&pending, &mut deltas),
            Mode::ToolCall => self.push_content(&format!("{TOOL_START}{pending}"), &mut deltas),
        }
        deltas
    }

    fn message(&self) -> Value {
        let mut message = json!({
            "role": "assistant",
            "content": if self.content.is_empty() { None } else { Some(&self.content) },
        });
        if let Some(reasoning) = &self.reasoning {
            message["reasoning_content"] = json!(reasoning);
        }
        if !self.tool_calls.is_empty() {
            message["tool_calls"] = json!(self.tool_calls);
        }
        message
    }

    fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

impl HybridOutputParser {
    fn push_content(&mut self, text: &str, deltas: &mut Vec<Value>) {
        if !text.is_empty() {
            self.content.push_str(text);
            deltas.push(json!({"content": text}));
        }
    }

    fn push_reasoning(&mut self, text: &str, deltas: &mut Vec<Value>) {
        if !text.is_empty() {
            self.reasoning
                .get_or_insert_with(String::new)
                .push_str(text);
            deltas.push(json!({"reasoning_content": text}));
        }
    }
}

fn next_marker(text: &str) -> Option<(usize, &'static str)> {
    [THINK_START, TOOL_START]
        .into_iter()
        .filter_map(|marker| text.find(marker).map(|position| (position, marker)))
        .min_by_key(|(position, _)| *position)
}

fn safe_prefix_len(text: &str, markers: &[&str]) -> usize {
    let hold = markers
        .iter()
        .flat_map(|marker| (1..marker.len()).map(move |length| &marker[..length]))
        .filter(|prefix| text.ends_with(*prefix))
        .map(str::len)
        .max()
        .unwrap_or(0);
    text.len() - hold
}

fn parse_tool_call(body: &str, uid: u64, index: usize) -> Option<Value> {
    let body = body.trim();
    let inner = body.strip_prefix("<function=")?;
    let (name, rest) = inner.split_once('>')?;
    let mut rest = rest.trim_start();
    let mut arguments = serde_json::Map::new();
    while let Some(inner) = rest.strip_prefix("<parameter=") {
        let (key, value) = inner.split_once('>')?;
        let (value, tail) = value.split_once("</parameter>")?;
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        arguments.insert(
            key.to_owned(),
            serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned())),
        );
        rest = tail.trim_start();
    }
    if name.is_empty() || rest.trim() != "</function>" {
        return None;
    }
    let arguments = serde_json::to_string(&arguments).ok()?;
    Some(json!({
        "id": format!("call_{uid}_{index}"),
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompt_opened_reasoning_and_xml_tools_survive_fragmentation() {
        let mut parser = new_parser(5);
        parser.set_prompt("<|im_start|>assistant\n<think>\n");
        for fragment in [
            "checking",
            "</th",
            "ink>answer<tool_call>\n<function=weather>\n<parameter=city>\n北京\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_",
            "call>",
        ] {
            parser.push(fragment);
        }
        parser.finish();
        let message = parser.message();
        assert_eq!(message["reasoning_content"], "checking");
        assert_eq!(message["content"], "answer");
        assert_eq!(message["tool_calls"][0]["function"]["name"], "weather");
        let args: Value = serde_json::from_str(
            message["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(args, json!({"city":"北京","days":3}));
    }
    #[test]
    fn disabled_thinking_is_content_and_invalid_tools_are_preserved() {
        let mut parser = new_parser(6);
        parser.set_prompt("<think>\n\n</think>\n\n");
        parser.push("hello<tool_call>invalid</tool_call>");
        parser.finish();
        assert_eq!(
            parser.message()["content"],
            "hello<tool_call>invalid</tool_call>"
        );
        assert!(!parser.has_tool_calls());
    }
}
