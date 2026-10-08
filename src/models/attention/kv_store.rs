//! CUDA scatter writes K/V together without converting location tensors.
use crate::engine::ModelRunnerError;
use std::ffi::{CStr, c_char, c_void};
use tch::Tensor;

unsafe extern "C" {
    fn sglang_store_kv(
        k: *const c_void,
        v: *const c_void,
        kc: *const c_void,
        vc: *const c_void,
        locations: *const c_void,
        reserved_skip_index: i64,
    ) -> bool;
    fn sglang_store_kv_error() -> *const c_char;
}

pub(super) fn store(
    k: &Tensor,
    v: &Tensor,
    kc: &Tensor,
    vc: &Tensor,
    locations: &Tensor,
    reserved_skip_index: i64,
) -> Result<(), ModelRunnerError> {
    if unsafe {
        sglang_store_kv(
            k.as_ptr().cast(),
            v.as_ptr().cast(),
            kc.as_ptr().cast(),
            vc.as_ptr().cast(),
            locations.as_ptr().cast(),
            reserved_skip_index,
        )
    } {
        return Ok(());
    }
    let error = unsafe { CStr::from_ptr(sglang_store_kv_error()) };
    Err(ModelRunnerError::Model(format!(
        "CUDA KV store failed: {}",
        error.to_string_lossy()
    )))
}
