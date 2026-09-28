//! Startup model registration and standalone chat-template detection.

use crate::{
    server::{ServeArgs, ServeComponents, ServeError},
    tokenizer::ChatTemplateRenderFn,
};

use super::qwen3;

type ComponentBuilder = fn(&ServeArgs) -> Result<ServeComponents, ServeError>;

pub(crate) struct ModelRegistration {
    pub model_type: &'static str,
    pub architecture: &'static str,
    pub build: ComponentBuilder,
    pub template_probe: Option<fn(&str) -> bool>,
    pub template_renderer: Option<ChatTemplateRenderFn>,
}

const MODELS: &[ModelRegistration] = &[qwen3::REGISTRATION];

pub(crate) fn component_builder(
    model_type: &str,
    architectures: &[String],
) -> Option<ComponentBuilder> {
    MODELS
        .iter()
        .find(|model| {
            model.model_type == model_type
                && architectures.iter().any(|name| name == model.architecture)
        })
        .map(|model| model.build)
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
        assert!(component_builder("qwen3", &["Qwen3ForCausalLM".to_owned()]).is_some());
        for (model_type, architecture) in [
            ("qwen3_moe", "Qwen3MoeForCausalLM"),
            ("qwen3", "OtherForCausalLM"),
            ("llama", "LlamaForCausalLM"),
        ] {
            assert!(component_builder(model_type, &[architecture.to_owned()]).is_none());
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
