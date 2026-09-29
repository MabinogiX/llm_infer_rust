//! Assemble tokenizer, engine, scheduler, and output parser for dense Qwen3.

use crate::{
    engine::{Engine, ModelArgs, validate_max_seq_len},
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
    validate_max_seq_len(&args.engine, model_args).map_err(ServeError::Engine)?;
    let tokenizer = TokenizerWorker::new_with_chat_template_kind(
        &args.engine.model_path,
        args.engine.trust_remote_code,
        ChatTemplateKind::Custom(render_chat_template),
    )
    .map_err(ServeError::Tokenizer)?;
    let engine = Engine::load_for_serving(args.engine.clone(), model_args, 0, &Qwen3Factory)
        .map_err(ServeError::Engine)?;
    let scheduler = Scheduler::new(engine).map_err(ServeError::Scheduler)?;
    Ok(ServeComponents {
        scheduler,
        tokenizer,
        output_parser_constructor: new_qwen3_output_parser,
    })
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::engine::{EngineError, ServerArgs};

    #[test]
    fn rejects_oversized_context_before_tokenizer_or_weights_load() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let model_dir = std::env::temp_dir().join(format!("sglang-qwen3-context-{nonce}"));
        fs::create_dir(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            r#"{"hidden_size":16,"num_attention_heads":4,"max_position_embeddings":40960}"#,
        )
        .unwrap();
        let mut args = ServeArgs::new(ServerArgs::new(&model_dir));
        args.engine.max_seq_len = 70000;

        let result = build(&args);
        fs::remove_dir_all(&model_dir).unwrap();
        assert!(
            matches!(result, Err(ServeError::Engine(EngineError::InvalidArgument(message)))
            if message.contains("--max-seq-len 70000") && message.contains("max_position_embeddings=40960"))
        );
    }
}
