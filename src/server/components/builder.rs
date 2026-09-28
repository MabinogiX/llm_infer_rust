//! Select one model assembly path at server startup.

use std::{fs, path::Path};

use serde::Deserialize;

use crate::{engine::EngineError, scheduler::Scheduler, tokenizer::TokenizerWorker};

use super::{
    super::{
        output::ChatOutputParserConstructor,
        serve::{ServeArgs, ServeError},
    },
    qwen3,
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

#[derive(Debug, PartialEq, Eq)]
enum ModelKind {
    Qwen3,
}

/// Identify the model once, then delegate all model-specific assembly.
pub fn build_components(args: &ServeArgs) -> Result<ServeComponents, ServeError> {
    let identity = read_model_identity(&args.engine.model_path)?;
    match select_model(&identity)? {
        ModelKind::Qwen3 => qwen3::build(args),
    }
}

fn read_model_identity(model_path: &Path) -> Result<ModelIdentity, ServeError> {
    let path = model_path.join("config.json");
    let contents = fs::read(&path).map_err(|error| invalid_config(&path, error.to_string()))?;
    serde_json::from_slice(&contents).map_err(|error| invalid_config(&path, error.to_string()))
}

fn invalid_config(path: &Path, message: String) -> ServeError {
    ServeError::Engine(EngineError::InvalidModelConfig {
        path: path.to_owned(),
        message,
    })
}

fn select_model(identity: &ModelIdentity) -> Result<ModelKind, ServeError> {
    if identity.model_type == "qwen3"
        && identity
            .architectures
            .iter()
            .any(|architecture| architecture == "Qwen3ForCausalLM")
    {
        return Ok(ModelKind::Qwen3);
    }
    Err(ServeError::UnsupportedModel {
        model_type: identity.model_type.clone(),
        architectures: identity.architectures.clone(),
    })
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::*;
    use crate::engine::ServerArgs;

    #[test]
    fn selects_dense_qwen3_from_hugging_face_identity() {
        let identity: ModelIdentity = serde_json::from_value(json!({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }))
        .unwrap();
        assert_eq!(select_model(&identity).unwrap(), ModelKind::Qwen3);
    }

    #[test]
    fn rejects_other_architectures_before_loading_weights() {
        for (model_type, architecture) in [
            ("qwen3_moe", "Qwen3MoeForCausalLM"),
            ("qwen3", "OtherForCausalLM"),
            ("llama", "LlamaForCausalLM"),
        ] {
            let identity = ModelIdentity {
                model_type: model_type.to_owned(),
                architectures: vec![architecture.to_owned()],
            };
            let error = select_model(&identity).unwrap_err();
            assert!(matches!(error, ServeError::UnsupportedModel { .. }));
            assert!(error.to_string().contains(model_type));
        }
    }

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
}
