//! Assemble tokenizer, engine, scheduler, and output parser for dense Qwen3.

use crate::{
    engine::{Engine, ModelArgs},
    models::registry::ModelRegistration,
    scheduler::Scheduler,
    server::{ServeArgs, ServeComponents, ServeError},
    tokenizer::{ChatTemplateKind, TokenizerWorker},
};

use super::{Qwen3Factory, matches_chat_template, new_qwen3_output_parser, render_chat_template};

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    model_type: "qwen3",
    architecture: "Qwen3ForCausalLM",
    build,
    template_probe: Some(matches_chat_template),
    template_renderer: Some(render_chat_template),
};

pub(crate) fn build(args: &ServeArgs) -> Result<ServeComponents, ServeError> {
    let model_args =
        ModelArgs::from_pretrained(&args.engine.model_path).map_err(ServeError::Engine)?;
    let tokenizer = TokenizerWorker::new_with_chat_template_kind(
        &args.engine.model_path,
        args.engine.trust_remote_code,
        ChatTemplateKind::Custom(render_chat_template),
    )
    .map_err(ServeError::Tokenizer)?;
    let mut engine = Engine::new(args.engine.clone(), model_args, 0).map_err(ServeError::Engine)?;
    engine
        .build_model(&Qwen3Factory)
        .map_err(ServeError::Engine)?;
    engine.load_model_weights().map_err(ServeError::Engine)?;
    let scheduler = Scheduler::new(engine).map_err(ServeError::Scheduler)?;
    Ok(ServeComponents {
        scheduler,
        tokenizer,
        output_parser_constructor: new_qwen3_output_parser,
    })
}
