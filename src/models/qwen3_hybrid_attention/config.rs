//! Qwen3.5 混合文本解码器配置（包括 Qwen3.8 检查点）。
use crate::engine::RuntimeModelConfig;
use serde::Deserialize;
use serde_json::Value;
use tch::Kind;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub hidden_size: i64,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub head_dim: i64,
    pub intermediate_size: i64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: i64,
    pub linear_key_head_dim: i64,
    pub linear_value_head_dim: i64,
    pub linear_num_key_heads: i64,
    pub linear_num_value_heads: i64,
    pub rope_parameters: RopeConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}
#[derive(Clone, Debug, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
}
impl Config {
    pub fn parse(raw: &Value) -> Result<Self, String> {
        let text = raw.get("text_config").unwrap_or(raw);
        let config: Self = serde_json::from_value(text.clone()).map_err(|e| e.to_string())?;
        for field in ["quantization_config", "compression_config"] {
            if [raw, text]
                .iter()
                .any(|v| v.get(field).is_some_and(|x| !x.is_null()))
            {
                return Err(format!("暂不支持 Qwen3.5 {field}"));
            }
        }
        for field in ["attention_bias", "mlp_bias"] {
            if text.get(field).is_some_and(|x| x != &Value::Bool(false)) {
                return Err(format!("暂不支持 Qwen3.5 {field}"));
            }
        }
        if text.get("hidden_act").and_then(Value::as_str) != Some("silu")
            || text
                .pointer("/rope_parameters/rope_type")
                .and_then(Value::as_str)
                != Some("default")
        {
            return Err("Qwen3.5 仅支持 silu 和 default RoPE".into());
        }
        for n in [
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.linear_conv_kernel_dim,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
        ] {
            if n <= 0 || n > i32::MAX as i64 {
                return Err("模型维度必须为有效正整数".into());
            }
        }
        let rotary = config.rotary_dim();
        if config.num_hidden_layers == 0
            || config.layer_types.len() != config.num_hidden_layers
            || config.full_layers() == 0
            || config.vocab_size == 0
            || config.max_position_embeddings == 0
            || config
                .layer_types
                .iter()
                .any(|s| s != "full_attention" && s != "linear_attention")
            || config.num_attention_heads % config.num_key_value_heads != 0
            || config.linear_num_value_heads % config.linear_num_key_heads != 0
            || !config.rms_norm_eps.is_finite()
            || config.rms_norm_eps <= 0.0
            || !config.rope_parameters.rope_theta.is_finite()
            || config.rope_parameters.rope_theta <= 0.0
            || !config.rope_parameters.partial_rotary_factor.is_finite()
            || rotary <= 0
            || rotary > config.head_dim
            || rotary % 2 != 0
        {
            return Err("无效的 Qwen3.5 混合层或 RoPE 配置".into());
        }
        if config.tie_word_embeddings {
            return Err("暂不支持 tied Qwen3.5 embeddings".into());
        }
        if text
            .get("attn_output_gate")
            .is_some_and(|v| v != &Value::Bool(true))
        {
            return Err("Qwen3.5 需要 attn_output_gate=true".into());
        }
        if text.get("output_gate_type").is_some_and(|v| {
            !v.is_null() && v.as_str() != Some("swish") && v.as_str() != Some("silu")
        }) {
            return Err("Qwen3.5 仅支持 swish 线性注意力门控".into());
        }
        Ok(config)
    }
    pub fn rotary_dim(&self) -> i64 {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as i64
    }
    pub fn full_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|s| *s == "full_attention")
            .count()
    }
    pub fn runtime(&self, checkpoint_kind: Kind) -> RuntimeModelConfig {
        RuntimeModelConfig {
            num_layers: self.full_layers(),
            num_kv_heads: self.num_key_value_heads as usize,
            head_dim: self.head_dim as usize,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            checkpoint_kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw() -> Value {
        serde_json::from_str(include_str!("../../../tests/fixtures/qwen3_8_config.json")).unwrap()
    }
    #[test]
    fn checkpoint_geometry_is_hybrid_and_runtime_only_counts_full_kv_layers() {
        let config = Config::parse(&raw()).unwrap();
        assert_eq!(config.full_layers(), 16);
        assert_eq!(config.num_hidden_layers, 64);
        assert_eq!(config.rotary_dim(), 64);
        let runtime = config.runtime(Kind::BFloat16);
        assert_eq!(runtime.num_layers, 16);
        assert_eq!(runtime.head_dim, 256);
        assert_eq!(runtime.vocab_size, 248320);
    }
    #[test]
    fn rejects_unsupported_computation_and_invalid_hybrid_layout() {
        for (field, value) in [
            ("hidden_act", serde_json::json!("gelu")),
            ("attention_bias", serde_json::json!(true)),
            ("num_attention_heads", serde_json::json!(0)),
            ("layer_types", serde_json::json!(["full_attention"])),
            ("output_gate_type", serde_json::json!("sigmoid")),
            ("attn_output_gate", serde_json::json!(false)),
        ] {
            let mut raw = raw();
            raw["text_config"][field] = value;
            assert!(Config::parse(&raw).is_err(), "{field}");
        }
        let mut raw = raw();
        raw["quantization_config"] = serde_json::json!({});
        assert!(Config::parse(&raw).is_err());
    }
}
