//! Assemble a single model runtime from its registered definition.

use std::{fs, path::Path};

use serde::Deserialize;

use crate::{
    engine::{Engine, EngineError, validate_model_path},
    models::registry,
    scheduler::Scheduler,
    tokenizer::{ChatTemplateKind, TokenizerWorker},
};

use super::super::{
    output::ChatOutputParserConstructor,
    serve::{ServeArgs, ServeError},
};

/// Runtime pieces that vary with the loaded model.
pub struct ServeComponents {
    pub scheduler: Scheduler,
    pub tokenizer: TokenizerWorker,
    pub output_parser_constructor: ChatOutputParserConstructor,
}

#[derive(Debug, Deserialize)]
struct ModelIdentity {
    model_type: String,
    #[serde(default)]
    architectures: Vec<String>,
}

/// Parse and validate once, then initialize tokenizer, model, cache, graphs and scheduler.
pub fn build_components(args: &ServeArgs) -> Result<ServeComponents, ServeError> {
    validate_model_path(&args.engine.model_path).map_err(ServeError::Engine)?;
    let path = args.engine.model_path.join("config.json");
    let raw = read_model_config(&path)?;
    let identity: ModelIdentity = serde_json::from_value(raw.clone())
        .map_err(|error| invalid_config(&path, error.to_string()))?;
    let registration = select_model(&identity)?;
    let checkpoint_kind =
        checkpoint_kind(&raw).map_err(|message| invalid_config(&path, message))?;
    let definition = (registration.parse)(&raw, checkpoint_kind)
        .map_err(|message| invalid_config(&path, message))?;
    Engine::validate_for_serving(
        &args.engine,
        definition.runtime,
        0,
        definition.factory.as_ref(),
    )
    .map_err(ServeError::Engine)?;
    let eos = super::generation::load_eos_token_ids(
        &args.engine.model_path,
        &raw,
        definition.runtime.vocab_size,
    )
    .map_err(ServeError::Engine)?;
    let template = registration
        .template_renderer
        .map(ChatTemplateKind::Custom)
        .unwrap_or(ChatTemplateKind::Auto);
    let tokenizer = TokenizerWorker::new_with_chat_template_kind(
        &args.engine.model_path,
        args.engine.trust_remote_code,
        template,
    )
    .map_err(ServeError::Tokenizer)?;
    let engine = Engine::load_for_serving(
        args.engine.clone(),
        definition.runtime,
        0,
        definition.factory.as_ref(),
    )
    .map_err(ServeError::Engine)?;
    let scheduler = Scheduler::new(engine, eos).map_err(ServeError::Scheduler)?;
    Ok(ServeComponents {
        scheduler,
        tokenizer,
        output_parser_constructor: registration.output_parser_constructor,
    })
}

fn read_model_config(path: &Path) -> Result<serde_json::Value, ServeError> {
    let contents = fs::read(path).map_err(|error| invalid_config(path, error.to_string()))?;
    serde_json::from_slice(&contents).map_err(|error| invalid_config(path, error.to_string()))
}

fn checkpoint_kind(raw: &serde_json::Value) -> Result<tch::Kind, String> {
    match raw
        .get("torch_dtype")
        .filter(|value| !value.is_null())
        .or_else(|| raw.get("dtype").filter(|value| !value.is_null()))
    {
        None => Ok(tch::Kind::Float),
        Some(value) => match value.as_str() {
            Some("float32") => Ok(tch::Kind::Float),
            Some("float16") => Ok(tch::Kind::Half),
            Some("bfloat16") => Ok(tch::Kind::BFloat16),
            _ => Err(format!("unsupported or invalid checkpoint dtype: {value}")),
        },
    }
}

fn invalid_config(path: &Path, message: String) -> ServeError {
    ServeError::Engine(EngineError::InvalidModelConfig {
        path: path.to_owned(),
        message,
    })
}

fn select_model(
    identity: &ModelIdentity,
) -> Result<&'static registry::ModelRegistration, ServeError> {
    registry::model_registration(&identity.model_type, &identity.architectures).ok_or_else(|| {
        ServeError::UnsupportedModel {
            model_type: identity.model_type.clone(),
            architectures: identity.architectures.clone(),
        }
    })
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::*;
    use crate::engine::ServerArgs;

    #[test]
    fn build_components_rejects_unsupported_model_before_loading() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let model_path = std::env::temp_dir().join(format!("sglang-unsupported-model-{nonce}"));
        fs::create_dir(&model_path).unwrap();
        fs::write(
            model_path.join("config.json"),
            json!({
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
            })
            .to_string(),
        )
        .unwrap();
        let args = ServeArgs::new(ServerArgs::new(&model_path));

        let result = build_components(&args);
        fs::remove_dir_all(&model_path).unwrap();
        assert!(matches!(result, Err(ServeError::UnsupportedModel { .. })));
    }

    struct ModelDirectory(std::path::PathBuf);
    impl ModelDirectory {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "sglang-startup-{nonce}-{}",
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn write(&self, filename: &str, value: &serde_json::Value) {
            fs::write(self.0.join(filename), value.to_string()).unwrap();
        }
        fn args(&self) -> ServeArgs {
            let mut args = ServeArgs::new(ServerArgs::new(&self.0));
            args.engine.device = "cpu".into();
            args.engine.max_seq_len = 16;
            args.engine.max_running_req = 1;
            args.engine.page_size = 2;
            args.engine.cuda_graph_bs = Some(0);
            args.engine.prefill_cuda_graph_max_tokens = 0;
            args
        }
    }
    impl Drop for ModelDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn tiny_config() -> serde_json::Value {
        json!({"model_type":"qwen3", "architectures":["Qwen3ForCausalLM"],
            "hidden_size":4, "num_hidden_layers":1, "num_attention_heads":2,
            "num_key_value_heads":1, "head_dim":2, "intermediate_size":8,
            "vocab_size":8, "max_position_embeddings":16, "eos_token_id":2})
    }
    fn engine_message(result: Result<ServeComponents, ServeError>) -> String {
        match result {
            Err(ServeError::Engine(error)) => error.to_string(),
            Err(other) => panic!("expected an error before tokenizer/weights loading: {other}"),
            Ok(_) => panic!("expected startup to reject configuration"),
        }
    }

    #[test]
    fn invalid_model_and_runtime_combinations_fail_before_tokenizer_or_weights() {
        let dir = ModelDirectory::new();
        for (field, value, expected) in [
            ("rope_scaling", json!({}), "rope_scaling"),
            ("quantization_config", json!({}), "quantization_config"),
            ("attention_bias", json!(true), "attention_bias"),
            ("torch_dtype", json!("int8"), "dtype"),
            ("num_hidden_layers", json!(0), "num_hidden_layers"),
        ] {
            let mut raw = tiny_config();
            raw[field] = value;
            dir.write("config.json", &raw);
            assert!(engine_message(build_components(&dir.args())).contains(expected));
        }
        dir.write("config.json", &tiny_config());
        let mut args = dir.args();
        args.engine.max_seq_len = 17;
        assert!(engine_message(build_components(&args)).contains("--max-seq-len 17"));
        args = dir.args();
        args.engine.dtype = "int8".into();
        assert!(engine_message(build_components(&args)).contains("dtype"));
        args = dir.args();
        args.engine.tp_size = 2;
        assert!(engine_message(build_components(&args)).contains("张量并行"));
        args = dir.args();
        args.engine.attention_backend = "unknown".into();
        assert!(engine_message(build_components(&args)).contains("unknown attention backend"));
        args.engine.attention_backend = "flashinfer".into();
        assert!(engine_message(build_components(&args)).contains("FlashInfer"));
    }

    #[test]
    fn strict_eos_precedence_and_validation_happen_before_model_loading() {
        use super::super::generation::load_eos_token_ids;
        use std::collections::BTreeSet;
        let dir = ModelDirectory::new();
        let config = tiny_config();
        assert_eq!(
            load_eos_token_ids(&dir.0, &config, 8).unwrap(),
            BTreeSet::from([2])
        );
        dir.write("generation_config.json", &json!({"eos_token_id":[7, 6, 7]}));
        assert_eq!(
            load_eos_token_ids(&dir.0, &config, 8).unwrap(),
            BTreeSet::from([6, 7])
        );
        for value in [json!(null), json!([])] {
            dir.write("generation_config.json", &json!({"eos_token_id":value}));
            assert!(load_eos_token_ids(&dir.0, &config, 8).unwrap().is_empty());
        }
        dir.write("generation_config.json", &json!({}));
        assert_eq!(
            load_eos_token_ids(&dir.0, &config, 8).unwrap(),
            BTreeSet::from([2])
        );
        dir.write("config.json", &config);
        for value in [
            json!(-1),
            json!(8),
            json!("2"),
            json!([2, null]),
            json!([2, "3"]),
            json!({"token_id":2}),
        ] {
            dir.write("generation_config.json", &json!({"eos_token_id":value}));
            assert!(engine_message(build_components(&dir.args())).contains("eos_token_id"));
        }
        fs::write(dir.0.join("generation_config.json"), "{").unwrap();
        assert!(engine_message(build_components(&dir.args())).contains("generation_config.json"));
        fs::remove_file(dir.0.join("generation_config.json")).unwrap();
        let mut missing = config;
        missing.as_object_mut().unwrap().remove("eos_token_id");
        dir.write("config.json", &missing);
        assert!(engine_message(build_components(&dir.args())).contains("missing eos_token_id"));
    }

    #[test]
    fn unified_startup_loads_tiny_qwen3_and_generates_without_eos() {
        use crate::{engine::SamplingParams, scheduler::FinishReason};
        use tch::{Device, Kind, Tensor};
        let dir = ModelDirectory::new();
        dir.write("config.json", &tiny_config());
        dir.write("generation_config.json", &json!({"eos_token_id":[]}));
        dir.write("tokenizer.json", &json!({"version":"1.0", "added_tokens":[],
            "model":{"type":"WordLevel", "vocab":{"<unk>":0,"hello":1,"world":2}, "unk_token":"<unk>"}}));
        let shapes: &[(&str, &[i64])] = &[
            ("model.embed_tokens.weight", &[8, 4]),
            ("model.norm.weight", &[4]),
            ("lm_head.weight", &[8, 4]),
            ("model.layers.0.input_layernorm.weight", &[4]),
            ("model.layers.0.post_attention_layernorm.weight", &[4]),
            ("model.layers.0.self_attn.q_norm.weight", &[2]),
            ("model.layers.0.self_attn.k_norm.weight", &[2]),
            ("model.layers.0.self_attn.q_proj.weight", &[4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", &[2, 4]),
            ("model.layers.0.self_attn.v_proj.weight", &[2, 4]),
            ("model.layers.0.self_attn.o_proj.weight", &[4, 4]),
            ("model.layers.0.mlp.gate_proj.weight", &[8, 4]),
            ("model.layers.0.mlp.up_proj.weight", &[8, 4]),
            ("model.layers.0.mlp.down_proj.weight", &[4, 8]),
        ];
        let weights: Vec<_> = shapes
            .iter()
            .map(|(name, shape)| (*name, Tensor::ones(*shape, (Kind::Float, Device::Cpu))))
            .collect();
        Tensor::write_safetensors(&weights, dir.0.join("model.safetensors")).unwrap();
        let mut components = build_components(&dir.args()).unwrap();
        assert!(components.scheduler.eos_token_ids().is_empty());
        assert_eq!(components.tokenizer.encode("hello").unwrap(), vec![1]);
        components
            .scheduler
            .add_request(
                vec![1],
                SamplingParams {
                    max_tokens: 2,
                    temperature: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();
        let output = components.scheduler.step().unwrap();
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|token| token.token_id == 0));
        assert_eq!(
            output.last().unwrap().finish_reason,
            Some(FinishReason::Length)
        );
        assert!(components.scheduler.is_idle());
        // A no-EOS model must still emit an abort without panicking.
        components
            .scheduler
            .add_request(vec![1; 17], SamplingParams::default())
            .unwrap();
        assert_eq!(
            components.scheduler.step().unwrap()[0].finish_reason,
            Some(FinishReason::Abort)
        );
    }
}
