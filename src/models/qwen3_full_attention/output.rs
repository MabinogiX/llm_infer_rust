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

pub fn new_qwen3_output_parser(uid: u64) -> Box<dyn ChatOutputParser> {
    Box::new(Qwen3OutputParser::new(uid))
}

struct Qwen3OutputParser {
    uid: u64,
    mode: Mode,
    pending: String,
    content: String,
    reasoning: Option<String>,
    tool_calls: Vec<Value>,
}

impl Qwen3OutputParser {
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

impl ChatOutputParser for Qwen3OutputParser {
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

impl Qwen3OutputParser {
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
    let payload: Value = serde_json::from_str(body.trim()).ok()?;
    let name = payload.get("name")?.as_str()?;
    let arguments = payload.get("arguments")?;
    let arguments = match arguments {
        Value::String(text) => text.clone(),
        _ => serde_json::to_string(arguments).ok()?,
    };
    Some(json!({
        "id": format!("call_{uid}_{index}"),
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(parts: &[&str]) -> (Qwen3OutputParser, Vec<Value>) {
        let mut parser = Qwen3OutputParser::new(42);
        let mut deltas = Vec::new();
        for part in parts {
            deltas.extend(parser.push(part));
        }
        deltas.extend(parser.finish());
        (parser, deltas)
    }

    #[test]
    fn splits_reasoning_and_content_across_fragments() {
        let (parser, deltas) = parse(&["<thi", "nk>step", " one</th", "ink>answer"]);
        assert_eq!(
            parser.message(),
            json!({"role": "assistant", "content": "answer", "reasoning_content": "step one"})
        );
        assert_eq!(
            deltas,
            vec![
                json!({"reasoning_content": "step"}),
                json!({"reasoning_content": " one"}),
                json!({"content": "answer"}),
            ]
        );
    }

    #[test]
    fn emits_structured_tool_call_and_nullable_content() {
        let (parser, deltas) = parse(&[
            "<tool_",
            "call>\n{\"name\":\"weather\",\"arguments\":{\"city\":\"北京\"}}",
            "</tool_call>",
        ]);
        let message = parser.message();
        assert!(message["content"].is_null());
        assert!(parser.has_tool_calls());
        assert_eq!(message["tool_calls"][0]["function"]["name"], "weather");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"北京"}"#
        );
        assert_eq!(deltas[0]["tool_calls"][0]["index"], 0);
        assert_eq!(deltas[0]["tool_calls"][0]["id"], "call_42_0");
    }

    #[test]
    fn retains_malformed_or_incomplete_tool_markup_as_content() {
        let (parser, _) = parse(&["hello <tool_call>{bad}</tool_call> world"]);
        assert_eq!(
            parser.message()["content"],
            "hello <tool_call>{bad}</tool_call> world"
        );
        let (parser, _) = parse(&["<tool_call>{\"name\":\"weather\""]);
        assert_eq!(
            parser.message()["content"],
            "<tool_call>{\"name\":\"weather\""
        );
    }

    #[test]
    fn keeps_partial_reasoning_and_numbers_multiple_calls() {
        let (parser, _) = parse(&["<think>still working"]);
        assert!(parser.message()["content"].is_null());
        assert_eq!(parser.message()["reasoning_content"], "still working");

        let (parser, deltas) = parse(&[
            "<tool_call>{\"name\":\"first\",\"arguments\":{}}</tool_call>",
            "<tool_call>{\"name\":\"second\",\"arguments\":{}}</tool_call>",
        ]);
        assert_eq!(parser.message()["tool_calls"].as_array().unwrap().len(), 2);
        assert_eq!(deltas[0]["tool_calls"][0]["index"], 0);
        assert_eq!(deltas[1]["tool_calls"][0]["index"], 1);
        assert_ne!(
            parser.message()["tool_calls"][0]["id"],
            parser.message()["tool_calls"][1]["id"]
        );
    }

    #[test]
    fn constructor_creates_isolated_request_parsers() {
        let mut first = new_qwen3_output_parser(1);
        let mut second = new_qwen3_output_parser(2);
        first.push("<think>first</think>");
        second.push("second");
        first.finish();
        second.finish();
        assert_eq!(first.message()["reasoning_content"], "first");
        assert_eq!(second.message()["content"], "second");
        assert!(second.message().get("reasoning_content").is_none());
    }
}
