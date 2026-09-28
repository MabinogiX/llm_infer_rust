//! Assemble tokenizer, engine, scheduler, and output parser for dense Qwen3.

use crate::{
    engine::{Engine, ModelArgs},
    models::Qwen3Factory,
    scheduler::Scheduler,
    tokenizer::{ChatTemplateKind, TokenizerWorker},
};

use super::{
    super::{
        output::new_qwen3_output_parser,
        serve::{ServeArgs, ServeError},
    },
    ServeComponents,
};

pub(super) fn build(args: &ServeArgs) -> Result<ServeComponents, ServeError> {
    let model_args =
        ModelArgs::from_pretrained(&args.engine.model_path).map_err(ServeError::Engine)?;
    let tokenizer = TokenizerWorker::new_with_chat_template_kind(
        &args.engine.model_path,
        args.engine.trust_remote_code,
        ChatTemplateKind::Qwen3,
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
