//! Startup model registration and standalone chat-template detection.

use serde_json::Value;
use tch::Kind;

use crate::{
    engine::{ModelFactory, RuntimeModelConfig},
    server::output::ChatOutputParserConstructor,
    tokenizer::ChatTemplateRenderFn,
};

use super::qwen3;

pub(crate) struct ModelDefinition {
    pub runtime: RuntimeModelConfig,
    pub factory: Box<dyn ModelFactory>,
}

pub(crate) struct ModelRegistration {
    pub model_type: &'static str,
    pub architecture: &'static str,
    pub parse: fn(&Value, Kind) -> Result<ModelDefinition, String>,
    pub template_probe: Option<fn(&str) -> bool>,
    pub template_renderer: Option<ChatTemplateRenderFn>,
    pub output_parser_constructor: ChatOutputParserConstructor,
}

const MODELS: &[ModelRegistration] = &[qwen3::REGISTRATION];

pub(crate) fn model_registration(
    model_type: &str,
    architectures: &[String],
) -> Option<&'static ModelRegistration> {
    MODELS.iter().find(|model| {
        model.model_type == model_type
            && architectures.iter().any(|name| name == model.architecture)
    })
}

pub(crate) fn detect_chat_template(source: &str) -> Option<ChatTemplateRenderFn> {
    MODELS.iter().find_map(|model| {
        model
            .template_probe
            .filter(|probe| probe(source))
            .and(model.template_renderer)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{ChatMessage, ChatTemplateOptions};

    #[test]
    fn selects_dense_qwen3_and_rejects_other_architectures() {
        assert!(model_registration("qwen3", &["Qwen3ForCausalLM".to_owned()]).is_some());
        for (model_type, architecture) in [
            ("qwen3_moe", "Qwen3MoeForCausalLM"),
            ("qwen3", "OtherForCausalLM"),
            ("llama", "LlamaForCausalLM"),
        ] {
            assert!(model_registration(model_type, &[architecture.to_owned()]).is_none());
        }
    }

    #[test]
    fn detects_qwen3_template() {
        let render =
            detect_chat_template("{% set ns = namespace(multi_step_tool=true) %}<|im_start|>")
                .expect("Qwen3 template should be registered");
        let rendered = render(
            &[ChatMessage::new("user", "Say hi")],
            ChatTemplateOptions {
                add_generation_prompt: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
