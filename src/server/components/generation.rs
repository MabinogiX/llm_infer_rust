//! Generation metadata resolved before tokenizer or model allocation.

use crate::engine::{EngineError, Result};
use serde_json::Value;
use std::{collections::BTreeSet, fs, path::Path};

pub(super) fn load_eos_token_ids(
    model_path: &Path,
    model_config: &Value,
    vocab_size: usize,
) -> Result<BTreeSet<i64>> {
    let generation_path = model_path.join("generation_config.json");
    let generation = match fs::read(&generation_path) {
        Ok(bytes) => {
            let value: Value = serde_json::from_slice(&bytes)
                .map_err(|error| invalid(&generation_path, error.to_string()))?;
            if !value.is_object() {
                return Err(invalid(
                    &generation_path,
                    "generation config must be an object".into(),
                ));
            }
            Some(value)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(invalid(&generation_path, error.to_string())),
    };
    let model_path = model_path.join("config.json");
    let (path, raw) = if let Some(raw) = generation
        .as_ref()
        .and_then(|config| config.get("eos_token_id"))
    {
        (&generation_path, raw)
    } else {
        (
            &model_path,
            model_config.get("eos_token_id").ok_or_else(|| {
                invalid(
                    &model_path,
                    "missing eos_token_id; use explicit null or [] for a model without EOS".into(),
                )
            })?,
        )
    };
    let mut ids = BTreeSet::new();
    let mut insert = |value: &Value| -> Result<()> {
        let id = value
            .as_i64()
            .filter(|id| *id >= 0 && (*id as u64) < vocab_size as u64)
            .ok_or_else(|| {
                invalid(
                    path,
                    format!("invalid eos_token_id {value}; expected integer in [0, {vocab_size})"),
                )
            })?;
        ids.insert(id);
        Ok(())
    };
    match raw {
        Value::Null => {}
        Value::Array(values) => {
            for value in values {
                insert(value)?;
            }
        }
        _ => insert(raw)?,
    }
    Ok(ids)
}

fn invalid(path: &Path, message: String) -> EngineError {
    EngineError::InvalidModelConfig {
        path: path.to_owned(),
        message,
    }
}
