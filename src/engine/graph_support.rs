//! Explicit graph admission after model/backend/dtype/cache initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphPadding {
    ReservedKvPageZero,
    InertTokenRows,
}

#[derive(Debug, Clone, Copy)]
pub struct GraphLimits {
    pub max_batch_size: usize,
    pub max_tokens: usize,
    pub max_context_len: usize,
    pub padding: GraphPadding,
}

#[derive(Debug, Clone, Copy)]
pub enum GraphSupport {
    Unsupported(&'static str),
    Supported(GraphLimits),
}
impl GraphSupport {
    pub fn admits(self, rows: usize, tokens: usize, context: usize) -> bool {
        match self {
            Self::Unsupported(_) => false,
            Self::Supported(limits) => {
                rows > 0
                    && tokens > 0
                    && rows <= limits.max_batch_size
                    && tokens <= limits.max_tokens
                    && context <= limits.max_context_len
            }
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct GraphCapabilities {
    pub decode: GraphSupport,
    pub segmented_prefill: GraphSupport,
}
impl Default for GraphCapabilities {
    fn default() -> Self {
        Self {
            decode: GraphSupport::Unsupported("model has not declared decode graph support"),
            segmented_prefill: GraphSupport::Unsupported(
                "model has not declared segmented prefill graph support",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_is_mode_specific_and_bounded() {
        let disabled = GraphCapabilities::default();
        assert!(!disabled.decode.admits(1, 1, 1));
        assert!(!disabled.segmented_prefill.admits(1, 1, 1));
        let supported = GraphSupport::Supported(GraphLimits {
            max_batch_size: 2,
            max_tokens: 16,
            max_context_len: 64,
            padding: GraphPadding::ReservedKvPageZero,
        });
        assert!(supported.admits(2, 16, 64));
        assert!(!supported.admits(3, 16, 64));
        assert!(!supported.admits(2, 17, 64));
        assert!(!supported.admits(2, 16, 65));
        assert!(!supported.admits(0, 0, 0));
    }
}
