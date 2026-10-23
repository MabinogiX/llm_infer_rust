//! Dense Qwen3 computation configuration; parsed once at startup.

use serde::Deserialize;
use serde_json::Value;
use tch::Kind;

use crate::engine::RuntimeModelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    #[serde(rename = "num_hidden_layers")]
    pub num_layers: usize,
    pub num_attention_heads: usize,
    #[serde(rename = "num_key_value_heads")]
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

fn default_max_position_embeddings() -> usize {
    8192
}
fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_tie_word_embeddings() -> bool {
    false
}

// Preserve the established defaults for programmatically constructed test models.
impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            hidden_size: 0,
            num_layers: 0,
            num_attention_heads: 0,
            num_kv_heads: 0,
            intermediate_size: 0,
            vocab_size: 0,
            head_dim: 0,
            max_position_embeddings: default_max_position_embeddings(),
            rope_theta: default_rope_theta(),
            rms_norm_eps: default_rms_norm_eps(),
            tie_word_embeddings: default_tie_word_embeddings(),
        }
    }
}

impl Qwen3Config {
    pub(crate) fn parse(raw: &Value) -> Result<Self, String> {
        let mut normalized = raw.clone();
        let object = normalized
            .as_object_mut()
            .ok_or("config must be an object")?;
        // Only dimensions derived from other fields need normalization.
        // Serde handles fixed defaults; explicit null and invalid types fail.
        if !object.contains_key("num_key_value_heads") {
            let heads = object
                .get("num_attention_heads")
                .cloned()
                .ok_or("missing num_attention_heads")?;
            object.insert("num_key_value_heads".into(), heads);
        }
        if !object.contains_key("head_dim") {
            let hidden = object
                .get("hidden_size")
                .and_then(Value::as_u64)
                .ok_or("invalid hidden_size")?;
            let heads = object
                .get("num_attention_heads")
                .and_then(Value::as_u64)
                .filter(|n| *n > 0)
                .ok_or("invalid num_attention_heads")?;
            if hidden % heads != 0 {
                return Err(
                    "hidden_size must be divisible by num_attention_heads when head_dim is omitted"
                        .into(),
                );
            }
            object.insert("head_dim".into(), Value::from(hidden / heads));
        }
        for field in [
            "rope_scaling",
            "rope_parameters",
            "quantization_config",
            "compression_config",
        ] {
            if raw.get(field).is_some_and(|value| !value.is_null()) {
                return Err(format!("unsupported Qwen3 computation feature: {field}"));
            }
        }
        for field in ["attention_bias", "mlp_bias", "use_sliding_window"] {
            if let Some(value) = raw.get(field) {
                if value != &Value::Bool(false) {
                    return Err(format!("unsupported or invalid Qwen3 {field}: {value}"));
                }
            }
        }
        // A dormant sliding_window value is metadata when use_sliding_window=false.
        // Layer-specific attention changes must still be rejected.
        if raw.get("layer_types").is_some_and(|value| !value.is_null()) {
            let types = raw["layer_types"].as_array().ok_or("invalid layer_types")?;
            let count = raw["num_hidden_layers"]
                .as_u64()
                .ok_or("invalid num_hidden_layers")?;
            if types.len() as u64 != count
                || types
                    .iter()
                    .any(|value| value.as_str() != Some("full_attention"))
            {
                return Err("unsupported Qwen3 layer_types; only full_attention for every layer is supported".into());
            }
        }
        if let Some(value) = raw.get("hidden_act") {
            if value.as_str() != Some("silu") {
                return Err(format!("unsupported Qwen3 hidden_act: {value}"));
            }
        }
        if let Some(value) = raw.get("qk_norm") {
            if value != &Value::Bool(true) {
                return Err("Qwen3 requires qk_norm=true".into());
            }
        }
        let config: Self = serde_json::from_value(normalized).map_err(|error| error.to_string())?;
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn validate(self) -> Result<(), String> {
        for (name, size) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_kv_heads),
            ("intermediate_size", self.intermediate_size),
            ("vocab_size", self.vocab_size),
            ("head_dim", self.head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if size == 0 || i64::try_from(size).is_err() {
                return Err(format!("{name} must be positive and fit in i64"));
            }
        }
        if self.num_attention_heads % self.num_kv_heads != 0 {
            return Err("num_attention_heads must be divisible by num_key_value_heads".into());
        }
        if self.head_dim % 2 != 0 {
            return Err("Qwen3 RoPE requires an even head_dim".into());
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err("rope_theta must be positive and finite".into());
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err("rms_norm_eps must be positive and finite".into());
        }
        let qkv_width = self
            .num_kv_heads
            .checked_mul(2)
            .and_then(|kv| self.num_attention_heads.checked_add(kv))
            .and_then(|heads| heads.checked_mul(self.head_dim));
        for width in [
            qkv_width,
            self.num_attention_heads.checked_mul(self.head_dim),
            self.num_kv_heads.checked_mul(self.head_dim),
            self.intermediate_size.checked_mul(2),
        ] {
            if width.and_then(|n| i64::try_from(n).ok()).is_none() {
                return Err("Qwen3 projection width exceeds i64".into());
            }
        }
        Ok(())
    }

    pub(crate) fn runtime(self, checkpoint_kind: Kind) -> RuntimeModelConfig {
        RuntimeModelConfig {
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            checkpoint_kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn raw() -> Value {
        json!({"hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "intermediate_size": 32, "vocab_size": 64,
            "max_position_embeddings": 128, "rope_theta": 1000000.0,
            "rms_norm_eps": 0.00001, "tie_word_embeddings": true})
    }

    #[test]
    fn extracts_runtime_geometry_and_keeps_computation_fields_model_specific() {
        let config = Qwen3Config::parse(&raw()).unwrap();
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.rope_theta, 1000000.0);
        assert!(config.tie_word_embeddings);
        let runtime = config.runtime(Kind::BFloat16);
        assert_eq!(runtime.num_kv_heads, 2);
        assert_eq!(runtime.vocab_size, 64);
        assert_eq!(runtime.checkpoint_kind, Kind::BFloat16);
        // Qwen3 supports attention width independent of hidden_size.
        let mut raw = raw();
        raw["head_dim"] = json!(8);
        assert!(Qwen3Config::parse(&raw).is_ok());
    }

    #[test]
    fn rejects_unsupported_computation_and_malformed_fields() {
        for (field, value) in [
            ("rope_scaling", json!({"rope_type":"linear", "factor":2.0})),
            ("rope_parameters", json!({})),
            ("quantization_config", json!({})),
            ("attention_bias", json!(true)),
            ("mlp_bias", json!(true)),
            ("use_sliding_window", json!(true)),
            ("hidden_act", json!("gelu")),
            ("qk_norm", json!(false)),
            (
                "layer_types",
                json!(["full_attention", "sliding_attention"]),
            ),
            ("rms_norm_eps", json!(0.0)),
            ("rope_theta", json!(-1.0)),
            ("tie_word_embeddings", json!("true")),
            ("num_hidden_layers", json!(0)),
            ("num_key_value_heads", json!(3)),
            ("head_dim", json!(3)),
        ] {
            let mut raw = raw();
            raw[field] = value;
            assert!(
                Qwen3Config::parse(&raw).is_err(),
                "{field} must be rejected"
            );
        }
        for field in ["num_hidden_layers", "intermediate_size", "vocab_size"] {
            let mut raw = raw();
            raw.as_object_mut().unwrap().remove(field);
            assert!(
                Qwen3Config::parse(&raw).is_err(),
                "{field} must be required"
            );
        }
    }

    #[test]
    fn permits_metadata_and_disabled_sliding_window() {
        let mut raw = raw();
        raw["transformers_version"] = json!("metadata");
        raw["use_sliding_window"] = json!(false);
        raw["sliding_window"] = json!(4096);
        raw["layer_types"] = json!(["full_attention", "full_attention"]);
        raw["rope_scaling"] = Value::Null;
        assert!(Qwen3Config::parse(&raw).is_ok());
    }

    #[test]
    fn missing_head_dim_requires_exact_division() {
        let mut raw = raw();
        raw["hidden_size"] = json!(17);
        assert!(Qwen3Config::parse(&raw).is_err());
        raw["head_dim"] = json!(4);
        assert!(Qwen3Config::parse(&raw).is_ok());
    }

    #[test]
    fn defaults_only_apply_to_missing_fields_and_keep_dynamic_dimensions() {
        let fields = [
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "tie_word_embeddings",
        ];
        let mut raw = raw();
        for field in fields {
            raw.as_object_mut().unwrap().remove(field);
        }
        raw.as_object_mut().unwrap().remove("num_key_value_heads");
        let config = Qwen3Config::parse(&raw).unwrap();
        let defaults = Qwen3Config::default();
        assert_eq!(
            config.max_position_embeddings,
            defaults.max_position_embeddings
        );
        assert_eq!(config.rope_theta, defaults.rope_theta);
        assert_eq!(config.rms_norm_eps, defaults.rms_norm_eps);
        assert_eq!(config.tie_word_embeddings, defaults.tie_word_embeddings);
        assert_eq!(config.num_kv_heads, config.num_attention_heads);
        assert_eq!(config.head_dim, 4);
        for field in fields {
            let mut invalid = raw.clone();
            invalid[field] = Value::Null;
            assert!(
                Qwen3Config::parse(&invalid).is_err(),
                "{field}: null must be rejected"
            );
            invalid[field] = json!("invalid");
            assert!(
                Qwen3Config::parse(&invalid).is_err(),
                "{field}: wrong type must be rejected"
            );
        }
    }
}
