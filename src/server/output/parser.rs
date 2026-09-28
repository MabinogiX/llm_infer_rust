use serde_json::Value;

/// A parser belongs to one generation request. Its deltas and final message
/// must describe the same output even when markers cross token boundaries.
pub trait ChatOutputParser: Send {
    fn push(&mut self, text: &str) -> Vec<Value>;
    fn finish(&mut self) -> Vec<Value>;
    fn message(&self) -> Value;
    fn has_tool_calls(&self) -> bool;
}

/// Selected once during model initialization and copied into the frontend.
pub type ChatOutputParserConstructor = fn(u64) -> Box<dyn ChatOutputParser>;
