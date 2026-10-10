//! Model definition only; runtime assembly belongs to server startup.

use serde_json::Value;
use tch::Kind;

use super::{
    Qwen3Config, Qwen3Factory, matches_chat_template, new_qwen3_output_parser, render_chat_template,
};
use crate::models::registry::{ModelDefinition, ModelRegistration};

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    model_type: "qwen3",
    architecture: "Qwen3ForCausalLM",
    parse,
    template_probe: Some(matches_chat_template),
    template_renderer: Some(render_chat_template),
    output_parser_constructor: new_qwen3_output_parser,
};

fn parse(raw: &Value, checkpoint_kind: Kind) -> Result<ModelDefinition, String> {
    let config = Qwen3Config::parse(raw)?;
    Ok(ModelDefinition {
        runtime: config.runtime(checkpoint_kind),
        factory: Box::new(Qwen3Factory { config }),
    })
}
