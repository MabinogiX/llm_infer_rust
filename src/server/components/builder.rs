//! Select one model assembly path at server startup.

use std::{fs, path::Path};

use serde::Deserialize;

use crate::{
    engine::EngineError, models::registry, scheduler::Scheduler, tokenizer::TokenizerWorker,
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

/// Identify the model once, then delegate all model-specific assembly.
pub fn build_components(args: &ServeArgs) -> Result<ServeComponents, ServeError> {
    let identity = read_model_identity(&args.engine.model_path)?;
    select_model(&identity)?(args)
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

fn select_model(
    identity: &ModelIdentity,
) -> Result<fn(&ServeArgs) -> Result<ServeComponents, ServeError>, ServeError> {
    registry::component_builder(&identity.model_type, &identity.architectures).ok_or_else(|| {
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
}
