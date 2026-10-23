//! Qwen3.5 / Qwen3.8 dense 模型的纯文本推理。
mod config;
mod model;
mod output;
use crate::models::registry::{ModelDefinition, ModelRegistration};
pub use config::Config;
pub use model::Factory;
pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    model_type: "qwen3_5",
    architecture: "Qwen3_5ForConditionalGeneration",
    parse: |raw, kind| {
        let config = Config::parse(raw)?;
        Ok(ModelDefinition {
            runtime: config.runtime(kind),
            factory: Box::new(Factory { config }),
        })
    },
    recurrent_state: true,
    template_probe: None,
    template_renderer: None,
    output_parser_constructor: output::new_parser,
};
